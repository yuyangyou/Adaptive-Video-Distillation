import os
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from causvid.dmd import DMD
import time
from tqdm import tqdm  

torch.set_grad_enabled(False)

# 配置
dmd_checkpoint_folder = "./model"
output_dir = "./VBench/vbench_videos/vif_dynamic"
config = OmegaConf.load("configs/wan_bidirectional_vif_from_scratch.yaml")
os.makedirs(output_dir, exist_ok=True)

# 模型初始化
dmd_model = DMD(config, device="cuda")
dmd_model = dmd_model.to(torch.bfloat16).cuda()

state_dict = torch.load(os.path.join(dmd_checkpoint_folder, "model.pt"), map_location="cpu")['generator']
dmd_model.generator.load_state_dict(state_dict, strict=False)

# 设置降噪时间步
denoising_step_list = [999, 900, 757, 522]
image_or_video_shape = [1, 21, 16, 60, 104]

# 输入txt文件
txt_file = "./anVIF/dynamic_degree_prompt.txt"
with open(txt_file, "r", encoding="utf-8") as f:
    filenames = [line.strip() for line in f if line.strip()]

# 遍历txt文件中的每一行，显示进度条
for idx, filename in enumerate(tqdm(filenames, desc="Generating videos", unit="video")):
    # 提取prompt
    prompt = filename.rsplit("-", 1)[0].replace(".mp4", "").strip()

    # ✅ 不同序号对应不同随机种子
    seed = idx
    torch.manual_seed(seed)

    # 编码prompt
    conditional_dict = dmd_model.text_encoder(text_prompts=prompt)

    # 初始化噪声
    scheduler = dmd_model.generator.get_scheduler()
    noisy_image_or_video = torch.randn(image_or_video_shape, device="cuda", dtype=torch.bfloat16)
    noisy_image_or_video = noisy_image_or_video[:, ::2]

    # 逐步去噪
    for step_idx, current_timestep in enumerate(denoising_step_list):
        pred_image_or_video = dmd_model.generator(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=torch.ones(
                noisy_image_or_video.shape[:2], dtype=torch.long, device=noisy_image_or_video.device
            ) * current_timestep
        )  # [B, F, C, H, W]

        if step_idx < len(denoising_step_list) - 1:
            if step_idx == 1:
                interpolation_output = dmd_model.generator.interpolate(pred_image_or_video)
            else:
                interpolation_output = pred_image_or_video

            next_timestep = denoising_step_list[step_idx + 1] * torch.ones(
                interpolation_output.shape[:2], dtype=torch.long, device=noisy_image_or_video.device)

            noisy_image_or_video = scheduler.add_noise(
                interpolation_output.flatten(0, 1),
                torch.randn_like(interpolation_output.flatten(0, 1)),
                next_timestep.flatten(0, 1)
            ).unflatten(0, interpolation_output.shape[:2])

    # 解码并保存
    interp_video = dmd_model.vae.decode_to_pixel(pred_image_or_video)
    interp_video = (interp_video * 0.5 + 0.5).clamp(0, 1)
    interp_video = interp_video[0].permute(0, 2, 3, 1).to(dtype=torch.float32).cpu().numpy()

    save_path = os.path.join(output_dir, filename)
    export_to_video(interp_video, save_path, fps=16)

print("✅ All prompts processed.")
