import pdb
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from causvid.dmd import DMD
from PIL import Image
import torch
from causvid.data import ODERegressionLMDBDataset
from causvid.util import cycle
torch.set_grad_enabled(False)

config = OmegaConf.load("configs/wan_bidirectional_dmd.yaml")

dmd_model = DMD(config, device="cuda")
dmd_model = dmd_model.to(torch.bfloat16).cuda()

conditional_dict = dmd_model.text_encoder(
    text_prompts=[r"""A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."""]
)
dataset = ODERegressionLMDBDataset("./mixkit_latents_lmdb", max_pair=int(1e8))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
dataloader = cycle(dataloader)

unconditional_dict = dmd_model.text_encoder(text_prompts=[config.negative_prompt] * 1)

# print("Test 1: Backward Simulation")

image_or_video_shape = [1, 21, 16, 60, 104]

print("Test 2: Generator Loss")
# generator_loss, generator_log_dict = dmd_model.generator_loss(
#     image_or_video_shape=image_or_video_shape,
#     conditional_dict=conditional_dict,
#     unconditional_dict=unconditional_dict,
#     clean_latent=None
# )
batch = next(dataloader)
text_prompts = batch["prompts"]
video_latent = batch["ode_latent"][:, -1].to(
                device="cuda", dtype=torch.bfloat16)
current_reg_loss, idx = dmd_model.reg_loss(
        image_or_video_shape=image_or_video_shape,
        conditional_dict=conditional_dict,
        unconditional_dict=unconditional_dict,
        clean_latent=video_latent
    )
print(current_reg_loss)
