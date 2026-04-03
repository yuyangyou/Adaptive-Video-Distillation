from causvid.models.wan.wan_wrapper import WanVAEWrapper
from diffusers.utils import export_to_video
import argparse
import torch
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)

args = parser.parse_args()

torch.set_grad_enabled(False)

model = WanVAEWrapper().to(device="cuda:0", dtype=torch.bfloat16)


for index, video_path in enumerate(sorted(glob.glob(args.data_path + "/*.pt"))):
    data = torch.load(video_path)

    prompt = list(data.keys())[0]

    video_latent = data[prompt][:, -1].cuda().to(torch.bfloat16)

    video = model.decode_to_pixel(video_latent)

    video = (video * 0.5 + 0.5).clamp(0, 1)[0].permute(0, 2, 3, 1).cpu().numpy()
    print(index, prompt)
    export_to_video(video, f"ode_output_{index:03d}.mp4", fps=16)
