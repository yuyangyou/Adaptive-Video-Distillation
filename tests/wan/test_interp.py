import pdb
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from causvid.dmd import DMD
from PIL import Image
import torch
from causvid.data import ODERegressionLMDBDataset
from causvid.models import get_block_class
from causvid.data import TextDataset
from causvid.util import (
    launch_distributed_job,
    prepare_for_saving,
    set_seed, init_logging_folder,
    fsdp_wrap, cycle,
    fsdp_state_dict,
    barrier
)
from diffusers.utils import export_to_video
import torch.distributed as dist
from omegaconf import OmegaConf
from causvid.dmd import DMD
import argparse
import torch
import wandb
import time
import os

import torch



torch.set_grad_enabled(False)
checkpoint_folder = "./interp50M_40k_steps"
config = OmegaConf.load("configs/wan_bidirectional_dmd_from_scratch.yaml")

dmd_model = DMD(config, device="cuda")
dmd_model = dmd_model.to(torch.bfloat16).cuda()
denoising_step_list = [999, 757, 522]
dataset = ODERegressionLMDBDataset("./anVIF/pexels_5s_lmdb", max_pair=int(1e8))

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.batch_size)
dataloader = cycle(dataloader)

state_dict = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']
dmd_model.generator.load_state_dict(state_dict, strict=False)
# unconditional_dict = dmd_model.text_encoder(text_prompts=["色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"] * 1)

# print("Test 1: Backward Simulation")

image_or_video_shape = [1, 21, 16, 60, 104]

batch = next(dataloader)
batch = next(dataloader)

batch = next(dataloader)
batch = next(dataloader)
text_prompts = batch["prompts"]
clean_latent = batch["ode_latent"][:, -1].to(device="cuda", dtype=torch.bfloat16)
# clean_video = (clean_latent * 0.5 + 0.5).clamp(0, 1)

# clean_video = clean_video[0].permute(0, 2, 3, 1).to(dtype=torch.float32).cpu().numpy()
# export_to_video(
#     clean_video, os.path.join("./", f"clean_video.mp4"), fps=16)

clean_latent=clean_latent[:, ::2]
interp_video = dmd_model.generator.interpolate(clean_latent)

interp_video = dmd_model.vae.decode_to_pixel(interp_video)
interp_video = (interp_video * 0.5 + 0.5).clamp(0, 1)

interp_video = interp_video[0].permute(0, 2, 3, 1).to(dtype=torch.float32).cpu().numpy()
export_to_video(
    interp_video, os.path.join("./", f"inter_video.mp4"), fps=16)

