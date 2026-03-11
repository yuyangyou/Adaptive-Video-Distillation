from causvid.models.wan.wan_wrapper import WanVAEWrapper
from causvid.util import launch_distributed_job
import torch.distributed as dist
import imageio.v3 as iio
from tqdm import tqdm
import argparse
import torch
import json
import math
import os

torch.set_grad_enabled(False)


def video_to_numpy(video_path):
    """
    Reads a video file and returns a NumPy array containing all frames.

    :param video_path: Path to the video file.
    :return: NumPy array of shape (num_frames, height, width, channels)
    """
    return iio.imread(video_path, plugin="pyav")  # Reads the entire video as a NumPy array


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_folder", type=str,
                        help="Path to the folder containing input videos.")
    parser.add_argument("--output_latent_folder", type=str,
                        help="Path to the folder where output latents will be saved.")
    parser.add_argument("--info_path", type=str,
                        help="Path to the info file containing video metadata.")

    args = parser.parse_args()

    # Step 1: Setup the environment
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Step 2: Create the generator
    launch_distributed_job()
    device = torch.cuda.current_device()

    with open(args.info_path, "r") as f:
        video_info = json.load(f)

    model = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)

    video_paths = sorted(list(video_info.keys()))

    os.makedirs(args.output_latent_folder, exist_ok=True)

    for index in tqdm(range(int(math.ceil(len(video_paths) / dist.get_world_size()))), disable=dist.get_rank() != 0):
        global_index = index * dist.get_world_size() + dist.get_rank()
        if global_index >= len(video_paths):
            break

        video_path = video_paths[global_index]
        prompt = video_info[video_path]

        try:
            array = video_to_numpy(os.path.join(
                args.input_video_folder, video_path))
        except:
            print(f"Failed to read video: {video_path}")
            continue

        video_tensor = torch.tensor(array, dtype=torch.float32, device=device).unsqueeze(0).permute(
            0, 4, 1, 2, 3
        ) / 255.0
        video_tensor = video_tensor * 2 - 1
        video_tensor = video_tensor.to(torch.bfloat16)
        encoded_latents = encode(model, video_tensor).transpose(2, 1)

        torch.save(
            {prompt: encoded_latents.cpu().detach()},
            os.path.join(args.output_latent_folder, f"{global_index:08d}.pt")
        )
    dist.barrier()


if __name__ == "__main__":
    main()
