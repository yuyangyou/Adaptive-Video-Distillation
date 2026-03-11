import os
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from causvid.dmd import DMD
from causvid.data import ODERegressionLMDBDataset
from causvid.util import cycle
import time

from viztracer import VizTracer

torch.set_grad_enabled(False)

# 配置
checkpoint_folder = "./dmd_baseline_official"
config = OmegaConf.load("configs/wan_bidirectional_vif_from_scratch.yaml")
output_dir = "./dmd_moviegen"
os.makedirs(output_dir, exist_ok=True)

# 模型初始化
dmd_model = DMD(config, device="cuda")
dmd_model = dmd_model.to(torch.bfloat16).cuda()
state_dict = torch.load(os.path.join(checkpoint_folder, "model.pt"), map_location="cpu")['generator']
dmd_model.generator.load_state_dict(state_dict, strict=False)


# # 数据集
dataset = ODERegressionLMDBDataset("./data/mixkit_latents_lmdb", max_pair=int(1e8))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
dataloader = cycle(dataloader)

with open(f'./anVIF/sample_dataset/MovieGenVideoBench.txt', 'r') as f:
    prompt_list = f.readlines()
test_prompt_list = [prompt.strip() for prompt in prompt_list]
# 设置降噪时间步
# denoising_step_list = [999, 992, 982, 949, 905, 810]
denoising_step_list = [999, 900, 757, 522]

image_or_video_shape = [1, 21, 16, 60, 104]
half_image_or_video_shape = [1, 11, 16, 60, 104]

tracer = VizTracer()
tracer.start()
# 处理前10条数据

# for i in range(len(test_prompt_list)):
for i in range(50):
    print(f"Processing sample {i + 1}/{len(test_prompt_list)}")

    batch = next(dataloader)
    # text_prompts = batch["prompts"]
    text_prompts = test_prompt_list[i]
    # text_prompts = "A squirrel wearing a tiny aviator hat and goggles, piloting a miniature airplane through a park."
    conditional_dict = dmd_model.text_encoder(text_prompts=text_prompts)

    scheduler = dmd_model.generator.get_scheduler()
    # noisy_image_or_video = torch.randn(image_or_video_shape, device="cuda", dtype=torch.bfloat16)
    noisy_image_or_video = torch.randn(image_or_video_shape, device="cuda", dtype=torch.bfloat16)
    print(f"prompt_{i}:{text_prompts}")
    for index, current_timestep in enumerate(denoising_step_list):
        dit_start_time = time.perf_counter()
        pred_image_or_video = dmd_model.generator(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=torch.ones(
                noisy_image_or_video.shape[:2], dtype=torch.long, device=noisy_image_or_video.device
            ) * current_timestep
        )  # [B, F, C, H, W]
        dit_end_time = time.perf_counter()
        print(f"dit time cost:{dit_end_time - dit_start_time}")
        if index < len(denoising_step_list) - 1:
            # if index == 1:
            #     interp_start_time = time.perf_counter()
            #     interpolation_output = dmd_model.generator.interpolate(pred_image_or_video)
            #     # interpolation_output = pred_image_or_video
            #     interp_end_time = time.perf_counter()
            #     print(f"interp time cost:{interp_end_time - interp_start_time}")
            # else:
            #     interpolation_output = pred_image_or_video
            interpolation_output = pred_image_or_video
            next_timestep = denoising_step_list[index + 1] * torch.ones(
                interpolation_output.shape[:2], dtype=torch.long, device=noisy_image_or_video.device)

            noisy_image_or_video = scheduler.add_noise(
                interpolation_output.flatten(0, 1),
                torch.randn_like(interpolation_output.flatten(0, 1)),
                next_timestep.flatten(0, 1)
            ).unflatten(0, interpolation_output.shape[:2])

    # 解码与保存视频
    interp_video = dmd_model.vae.decode_to_pixel(pred_image_or_video)
    interp_video = (interp_video * 0.5 + 0.5).clamp(0, 1)
    interp_video = interp_video[0].permute(0, 2, 3, 1).to(dtype=torch.float32).cpu().numpy()

    export_to_video(interp_video, os.path.join(output_dir, f"output_{i + 1:02d}.mp4"), fps=16)
    print(f"Saved to {os.path.join(output_dir, f'output_{i + 1:02d}.mp4')}")
# tracer.stop()
# tracer.save(output_file="./full_log.html")

print("All 10 samples processed.")
