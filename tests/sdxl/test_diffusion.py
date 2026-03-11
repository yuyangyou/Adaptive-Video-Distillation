from causvid.models.sdxl.sdxl_wrapper import SDXLWrapper, SDXLTextEncoder, SDXLVAE
from huggingface_hub import hf_hub_download
from PIL import Image
import torch

torch.set_grad_enabled(False)

model = SDXLWrapper().to("cuda")
encoder = SDXLTextEncoder().to("cuda")
vae = SDXLVAE().to("cuda")

repo_name = "tianweiy/DMD2"
ckpt_name = "dmd2_sdxl_1step_unet_fp16.bin"
model.model.load_state_dict(torch.load(
    hf_hub_download(repo_name, ckpt_name), map_location="cuda"))
model.set_module_grad(
    {
        "model": False
    }
)

conditional_dict = encoder(
    text_prompts=["a photo of a cat"] * 4
)

noise = torch.randn(
    4, 1, 4, 128, 128, generator=torch.Generator().manual_seed(42)
).to("cuda")

timetep = 399 * torch.ones([4, 1], device="cuda", dtype=torch.long)
output = model(noise, conditional_dict, timetep)

# [B, F, C, H, W] -> [B, C, H, W]
images = vae.decode_to_pixel(output).squeeze(1)

images = ((images + 1.0) * 127.5).clamp(0,
                                        255).to(torch.uint8).permute(0, 2, 3, 1)

output_image_list = []
for index, image in enumerate(images):
    output_image_list.append(Image.fromarray(image.cpu().numpy()))

    output_image_list[-1].save(f"output_image_{index}.jpg")
