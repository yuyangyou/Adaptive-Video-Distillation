from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from causvid.dmd import DMD
from PIL import Image
import torch

torch.set_grad_enabled(False)

config = OmegaConf.load("configs/sdxl_8node_dmd_config.yaml")

config.mixed_precision = False

dmd_model = DMD(config, device="cuda")
dmd_model = dmd_model.to("cuda")

repo_name = "tianweiy/DMD2"
ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"

dmd_model.generator.model.load_state_dict(torch.load(
    hf_hub_download(repo_name, ckpt_name), map_location="cuda"))

conditional_dict = dmd_model.text_encoder(
    text_prompts=["a photo of a cat"] * 4
)

unconditional_dict = dmd_model.text_encoder(
    text_prompts=[""] * 4
)

print("Test 1: Backward Simulation")

noise = torch.randn(
    4, 1, 4, 128, 128, generator=torch.Generator().manual_seed(42)
).to("cuda")

image_or_video_shape = [4, 1, 4, 128, 128]


simulated_input = dmd_model._consistency_backward_simulation(
    noise=torch.randn(image_or_video_shape,
                      device="cuda", dtype=torch.float32),
    conditional_dict=conditional_dict
)

# 4 x 1 x 4 x 128 x 128
output = simulated_input[:, -1]
# [B, F, C, H, W] -> [B, C, H, W]
images = dmd_model.vae.decode_to_pixel(output).squeeze(1)

images = ((images + 1.0) * 127.5).clamp(0,
                                        255).to(torch.uint8).permute(0, 2, 3, 1)

output_image_list = []
for index, image in enumerate(images):
    output_image_list.append(Image.fromarray(image.cpu().numpy()))

    output_image_list[-1].save(f"backsim_image_{index}.jpg")

print("Test 2: Generator Loss")
generator_loss, generator_log_dict = dmd_model.generator_loss(
    image_or_video_shape=image_or_video_shape,
    conditional_dict=conditional_dict,
    unconditional_dict=unconditional_dict,
    clean_latent=None
)

print("Test 3: Critic Loss")
critic_loss, critic_log_dict = dmd_model.critic_loss(
    image_or_video_shape=image_or_video_shape,
    conditional_dict=conditional_dict,
    unconditional_dict=unconditional_dict,
    clean_latent=None
)

print(
    f"Generator Loss: {generator_loss}. dmdtrain_gradient_norm: {generator_log_dict['dmdtrain_gradient_norm']}")

print(
    f"Critic Loss: {critic_loss}.")

(dmdtrain_clean_latent, dmdtrain_noisy_latent, dmdtrain_pred_real_image, dmdtrain_pred_fake_image) = (
    generator_log_dict['dmdtrain_clean_latent'],
    generator_log_dict['dmdtrain_noisy_latent'],
    generator_log_dict['dmdtrain_pred_real_image'],
    generator_log_dict['dmdtrain_pred_fake_image']
)

(critictrain_latent, critictrain_noisy_latent, critictrain_pred_image) = (
    critic_log_dict['critictrain_latent'],
    critic_log_dict['critictrain_noisy_latent'],
    critic_log_dict['critictrain_pred_image']
)


dmdtrain_clean_latent_images = dmd_model.vae.decode_to_pixel(
    dmdtrain_clean_latent).squeeze(1)

dmdtrain_noisy_latent_images = dmd_model.vae.decode_to_pixel(
    dmdtrain_noisy_latent).squeeze(1)

dmdtrain_pred_real_image_images = dmd_model.vae.decode_to_pixel(
    dmdtrain_pred_real_image).squeeze(1)

dmdtrain_pred_fake_image_images = dmd_model.vae.decode_to_pixel(
    dmdtrain_pred_fake_image).squeeze(1)

critictrain_latent_images = dmd_model.vae.decode_to_pixel(
    critictrain_latent).squeeze(1)

critictrain_noisy_latent_images = dmd_model.vae.decode_to_pixel(
    critictrain_noisy_latent).squeeze(1)

critictrain_pred_image_images = dmd_model.vae.decode_to_pixel(
    critictrain_pred_image).squeeze(1)

dmdtrain_clean_latent_images = ((dmdtrain_clean_latent_images + 1.0) * 127.5).clamp(0,
                                                                                    255).to(torch.uint8).permute(0, 2, 3, 1)

dmdtrain_noisy_latent_images = ((dmdtrain_noisy_latent_images + 1.0) * 127.5).clamp(0,
                                                                                    255).to(torch.uint8).permute(0, 2, 3, 1)

dmdtrain_pred_real_image_images = ((dmdtrain_pred_real_image_images + 1.0) * 127.5).clamp(0,
                                                                                          255).to(torch.uint8).permute(0, 2, 3, 1)

dmdtrain_pred_fake_image_images = ((dmdtrain_pred_fake_image_images + 1.0) * 127.5).clamp(0,
                                                                                          255).to(torch.uint8).permute(0, 2, 3, 1)

critictrain_latent_images = ((critictrain_latent_images + 1.0) * 127.5).clamp(0,
                                                                              255).to(torch.uint8).permute(0, 2, 3, 1)

critictrain_noisy_latent_images = ((critictrain_noisy_latent_images + 1.0) * 127.5).clamp(0,
                                                                                          255).to(torch.uint8).permute(0, 2, 3, 1)

critictrain_pred_image_images = ((critictrain_pred_image_images + 1.0) * 127.5).clamp(0,
                                                                                      255).to(torch.uint8).permute(0, 2, 3, 1)

for index in range(len(dmdtrain_clean_latent_images)):
    Image.fromarray(dmdtrain_clean_latent_images[index].cpu().numpy()).save(
        f"dmdtrain_clean_latent_image_{index}.jpg")

    Image.fromarray(dmdtrain_noisy_latent_images[index].cpu().numpy()).save(
        f"dmdtrain_noisy_latent_image_{index}.jpg")

    Image.fromarray(dmdtrain_pred_real_image_images[index].cpu().numpy()).save(
        f"dmdtrain_pred_real_image_{index}.jpg")

    Image.fromarray(dmdtrain_pred_fake_image_images[index].cpu().numpy()).save(
        f"dmdtrain_pred_fake_image_{index}.jpg")

    Image.fromarray(critictrain_latent_images[index].cpu().numpy()).save(
        f"critictrain_latent_image_{index}.jpg")

    Image.fromarray(critictrain_noisy_latent_images[index].cpu().numpy()).save(
        f"critictrain_noisy_latent_image_{index}.jpg")

    Image.fromarray(critictrain_pred_image_images[index].cpu().numpy()).save(
        f"critictrain_pred_image_{index}.jpg")
