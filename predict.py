import os
from cog import BasePredictor, Input, Path
from typing import List
import sys
sys.path.append('/content/AniPortrait')
os.chdir('/content/AniPortrait')

import ffmpeg
from datetime import datetime
from pathlib import Path as MyPath
import numpy as np
import cv2
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

from src.utils.mp_utils  import LMKExtractor
from src.utils.draw_util import FaceMeshVisualizer
from src.utils.pose_util import project_points_with_trans

class ChatGPTConfig:
    def __init__(self):
        self.pretrained_base_model_path = './pretrained_model/stable-diffusion-v1-5'
        self.pretrained_vae_path = './pretrained_model/sd-vae-ft-mse'
        self.image_encoder_path = './pretrained_model/image_encoder'
        self.denoising_unet_path = "./pretrained_model/denoising_unet.pth"
        self.reference_unet_path = "./pretrained_model/reference_unet.pth"
        self.pose_guider_path = "./pretrained_model/pose_guider.pth"
        self.motion_module_path = "./pretrained_model/motion_module.pth"
        self.inference_config = "./configs/inference/inference_v2.yaml"
        self.weight_dtype = 'fp16'

class Args:
    def __init__(self):
        self.W = 512
        self.H = 512
        self.L = 64
        self.seed = 42
        self.cfg = 3.5
        self.steps = 25
        self.fps = 30

def cut_video_into_three_equal_parts(input_file):
    (
        ffmpeg
        .input(input_file)
        .filter('crop', 'iw/3', 'ih', x='0', y='0')
        .output('/content/output_part1.mp4')
        .run(overwrite_output=True)
    )
    (
        ffmpeg
        .input(input_file)
        .filter('crop', 'iw/3', 'ih', x='iw/3', y='0')
        .output('/content/output_part2.mp4')
        .run(overwrite_output=True)
    )
    (
        ffmpeg
        .input(input_file)
        .filter('crop', 'iw/3', 'ih', x='iw*2/3', y='0')
        .output('/content/output_part3.mp4')
        .run(overwrite_output=True)
    )
    audio_output = 'audio_from_video.aac'
    ffmpeg.input(input_file).audio.output(audio_output, acodec='copy').run(overwrite_output=True)
    audio = ffmpeg.input(audio_output)
    video1 = ffmpeg.input('/content/output_part1.mp4')
    video2 = ffmpeg.input('/content/output_part2.mp4')
    video3 = ffmpeg.input('/content/output_part3.mp4')
    ffmpeg.concat(video1, audio, v=1, a=1).output('/content/output_part1_a.mp4').run(overwrite_output=True)
    ffmpeg.concat(video2, audio, v=1, a=1).output('/content/output_part2_a.mp4').run(overwrite_output=True)
    ffmpeg.concat(video3, audio, v=1, a=1).output('/content/output_part3_a.mp4').run(overwrite_output=True)

config = ChatGPTConfig()
args = Args()

class Predictor(BasePredictor):
    def setup(self) -> None:

        if config.weight_dtype == "fp16":
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32

        vae = AutoencoderKL.from_pretrained(
            config.pretrained_vae_path,
        ).to("cuda", dtype=weight_dtype)

        reference_unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_base_model_path,
            subfolder="unet",
        ).to(dtype=weight_dtype, device="cuda")

        inference_config_path = config.inference_config
        infer_config = OmegaConf.load(inference_config_path)
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device="cuda")

        pose_guider = PoseGuider(noise_latent_channels=320, use_ca=True).to(device="cuda", dtype=weight_dtype) # not use cross attention

        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            config.image_encoder_path
        ).to(dtype=weight_dtype, device="cuda")

        sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
        scheduler = DDIMScheduler(**sched_kwargs)

        denoising_unet.load_state_dict(
            torch.load(config.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(config.reference_unet_path, map_location="cpu"),
        )
        pose_guider.load_state_dict(
            torch.load(config.pose_guider_path, map_location="cpu"),
        )

        self.pipe = Pose2VideoPipeline(
            vae=vae,
            image_encoder=image_enc,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        self.pipe = self.pipe.to("cuda", dtype=weight_dtype)

        self.lmk_extractor = LMKExtractor()
        self.vis = FaceMeshVisualizer(forehead_edge=False)

    def predict(
        self,
        ref_image_path: Path = Input(description="Input image"),
        source_video_path: Path = Input(description="Input video"),
        width : int = Input(default=512),
        height: int = Input(default=512),
        lenght: int = Input(default=64),
        seed: int = Input(default=42),
        cfg: float = Input(default=3.5),
        steps: int = Input(default=25),
    ) -> List[Path]:
        args.W = width
        args.H = height
        args.L = lenght
        args.seed = seed
        args.cfg = cfg
        args.steps = steps
        args.fps = 30

        generator = torch.manual_seed(args.seed)

        ref_image_pil = Image.open(ref_image_path).convert("RGB")
        ref_image_np = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)
        ref_image_np = cv2.resize(ref_image_np, (args.H, args.W))

        face_result = self.lmk_extractor(ref_image_np)
        assert face_result is not None, "Can not detect a face in the reference image."
        lmks = face_result['lmks'].astype(np.float32)
        ref_pose = self.vis.draw_landmarks((ref_image_np.shape[1], ref_image_np.shape[0]), lmks, normed=True)

        source_images = read_frames(source_video_path)
        src_fps = get_fps(source_video_path)
        print(f"source video has {len(source_images)} frames, with {src_fps} fps")
        pose_transform = transforms.Compose(
            [transforms.Resize((args.H, args.W)), transforms.ToTensor()]
        )

        step = 1
        if src_fps == 60:
            src_fps = 30
            step = 2

        pose_trans_list = []
        verts_list = []
        bs_list = []
        src_tensor_list = []
        for src_image_pil in source_images[: args.L*step: step]:
            src_tensor_list.append(pose_transform(src_image_pil))
            src_img_np = cv2.cvtColor(np.array(src_image_pil), cv2.COLOR_RGB2BGR)
            frame_height, frame_width, _ = src_img_np.shape
            src_img_result = self.lmk_extractor(src_img_np)
            if src_img_result is None:
                break
            pose_trans_list.append(src_img_result['trans_mat'])
            verts_list.append(src_img_result['lmks3d'])
            bs_list.append(src_img_result['bs'])

        pose_arr = np.array(pose_trans_list)
        verts_arr = np.array(verts_list)
        bs_arr = np.array(bs_list)
        min_bs_idx = np.argmin(bs_arr.sum(1))

        # face retarget
        verts_arr = verts_arr - verts_arr[min_bs_idx] + face_result['lmks3d']
        # project 3D mesh to 2D landmark
        projected_vertices = project_points_with_trans(verts_arr, pose_arr, [frame_height, frame_width])

        pose_list = []
        for i, verts in enumerate(projected_vertices):
            lmk_img = self.vis.draw_landmarks((frame_width, frame_height), verts, normed=False)
            pose_image_np = cv2.resize(lmk_img,  (args.W, args.H))
            pose_list.append(pose_image_np)

        pose_list = np.array(pose_list)

        video_length = len(src_tensor_list)

        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(
            0
        )  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=video_length
        )

        src_tensor = torch.stack(src_tensor_list, dim=0)  # (f, c, h, w)
        src_tensor = src_tensor.transpose(0, 1)
        src_tensor = src_tensor.unsqueeze(0)

        video = self.pipe(
            ref_image_pil,
            pose_list,
            ref_pose,
            args.W,
            args.H,
            video_length,
            args.steps,
            args.cfg,
            generator=generator,
        ).videos

        video = torch.cat([ref_image_tensor, video, src_tensor], dim=0)

        save_path = f"/content/noaudio.mp4"
        save_videos_grid(
            video,
            save_path,
            n_rows=3,
            fps=src_fps if args.fps is None else args.fps,
        )

        probe = ffmpeg.probe(save_path)
        duration = float(probe['format']['duration'])
        output_file = "/content/output.mp4"
        audio_stream = ffmpeg.input(source_video_path).audio.filter('atrim', duration=duration)
        video_stream = ffmpeg.input(save_path)
        ffmpeg.output(video_stream, audio_stream, output_file).overwrite_output().run()
        cut_video_into_three_equal_parts(output_file)
        # return Path('/content/output_part2.mp4')
        return [Path("/content/output.mp4"), Path("/content/output_part1_a.mp4"), Path("/content/output_part2_a.mp4"), Path("/content/output_part3_a.mp4")]