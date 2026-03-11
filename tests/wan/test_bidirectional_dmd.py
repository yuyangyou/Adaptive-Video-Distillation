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

torch.set_grad_enabled(False)
checkpoint_folder = "./wan_bidirectional_dmd_from_scratch/2025-07-14-02-35-24.612658_seed314/checkpoint_model_020000"
config = OmegaConf.load("configs/wan_bidirectional_dmd_from_scratch.yaml")

dmd_model = DMD(config, device="cuda")
dmd_model = dmd_model.to(torch.bfloat16).cuda()

dataset = ODERegressionLMDBDataset("./anVIF/mixkit_latents_lmdb", max_pair=int(1e8))

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=config.batch_size)
dataloader = cycle(dataloader)

# state_dict = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")[
#     'generator']
# dmd_model.generator.load_state_dict(state_dict)
# unconditional_dict = dmd_model.text_encoder(text_prompts=["色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"] * 1)

# print("Test 1: Backward Simulation")

image_or_video_shape = [1, 21, 16, 60, 104]
denoising_step_list=[999, 757]

batch = next(dataloader)
batch = next(dataloader)
text_prompts = batch["prompts"]
clean_latent = batch["ode_latent"][:, -1].to(device="cuda", dtype=torch.bfloat16)
conditional_dict = dmd_model.text_encoder(
    text_prompts=text_prompts)
noisy_image_or_video = torch.randn(image_or_video_shape, device="cuda", dtype=torch.bfloat16)
unconditional_dict = conditional_dict
# for index, current_timestep in enumerate(self.denoising_step_list):
#     pred_image_or_video = dmd_model.generator(
#         noisy_image_or_video=noisy_image_or_video,
#         conditional_dict=conditional_dict,
#         timestep=torch.ones(
#             noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
#     )  # [B, F, C, H, W]

#     if index < len(self.denoising_step_list) - 1:
#         next_timestep = self.denoising_step_list[index + 1] * torch.ones(
#             noise.shape[:2], dtype=torch.long, device=noise.device)
#         interpolation_video = dmd_model.generator.interpolate(pred_image_or_video)
#         noisy_image_or_video = self.scheduler.add_noise(
#             interpolation_video.flatten(0, 1),
#             torch.randn_like(pred_image_or_video.flatten(0, 1)),
#             next_timestep.flatten(0, 1)
#         ).unflatten(0, noise.shape[:2])


# interpolation_input = clean_latent[:, ::2]
# interpolation_output = dmd_model.generator.interpolate(interpolation_input)
# interp_video = dmd_model.vae.decode_to_pixel(interpolation_output)
# interp_video = (interp_video * 0.5 + 0.5).clamp(0, 1)

# origin_video = dmd_model.vae.decode_to_pixel(clean_latent)
# origin_video = (origin_video * 0.5 + 0.5).clamp(0, 1)

# interp_video = interp_video[0].permute(0, 2, 3, 1).to(dtype=torch.float32).cpu().numpy()
# origin_video = origin_video[0].permute(0, 2, 3, 1).to(dtype=torch.float32).cpu().numpy()
# print(f"interp_video{interp_video.shape}")
# print(f"origin_video{origin_video.shape}")
# export_to_video(
#     interp_video, os.path.join("./", f"interp.mp4"), fps=16)
# export_to_video(
#     origin_video, os.path.join("./", f"origin.mp4"), fps=16)


# [B, F, C, H, W] -> [B, C, H, W]

print("Test 2: Generator Loss")
generator_loss, generator_log_dict = dmd_model.generator_loss(
    image_or_video_shape=image_or_video_shape,
    conditional_dict=conditional_dict,
    unconditional_dict=unconditional_dict,
    clean_latent=clean_latent
)

print("Test 3: Critic Loss")
critic_loss, critic_log_dict = dmd_model.critic_loss(
    image_or_video_shape=image_or_video_shape,
    conditional_dict=conditional_dict,
    unconditional_dict=unconditional_dict,
    clean_latent=clean_latent
)

print("Test 4: inter Loss")
inter_loss = dmd_model.interpolation_loss(
    image_or_video_shape=image_or_video_shape,
    conditional_dict=conditional_dict,
    unconditional_dict=unconditional_dict,
    clean_latent=clean_latent
)

for name, param in dmd_model.generator.model.interp.named_parameters():
    print(name, param.shape, param.requires_grad)

print(
    f"Generator Loss: {generator_loss}. dmdtrain_gradient_norm: {generator_log_dict['dmdtrain_gradient_norm']}")

print(
    f"Critic Loss: {critic_loss}.")

print(
    f"inter Loss: {inter_loss}.")
