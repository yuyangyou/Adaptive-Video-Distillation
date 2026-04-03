from causvid.models.wan.wan_wrapper import WanTextEncoder
import torch

torch.set_grad_enabled(False)

model = WanTextEncoder().to(device="cuda:0", dtype=torch.bfloat16)

prompt_list = ["a " * 50] * 10 + ["b " * 25] * 10


print("Test Text Encoder")

encoded_dict = model(prompt_list)

assert encoded_dict['prompt_embeds'].shape[0] == 20 and encoded_dict[
    'prompt_embeds'].shape[1] == 512 and encoded_dict['prompt_embeds'].shape[2] == 4096
