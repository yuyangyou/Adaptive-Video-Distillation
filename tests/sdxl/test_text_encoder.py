from causvid.models.sdxl.sdxl_wrapper import SDXLTextEncoder
import time

model = SDXLTextEncoder()

prompt_list = ["a" * 300] * 20

print("Test Text Tokenizer")

for _ in range(20):
    start = time.time()

    output = model._encode_prompt(prompt_list)

    assert "text_input_ids_one" in output.keys()
    assert "text_input_ids_two" in output

    assert output["text_input_ids_one"].shape[0] == 20 and output["text_input_ids_one"].shape[1] == 77
    assert output["text_input_ids_two"].shape[0] == 20 and output["text_input_ids_two"].shape[1] == 77

    end = time.time()

    print(f"Time taken: {end - start}")

print("Test Text Encoder")

encoded_dict = model(prompt_list)

assert encoded_dict['prompt_embeds'].shape[1] == 77 and encoded_dict['prompt_embeds'].shape[2] == 2048
assert encoded_dict['pooled_prompt_embeds'].shape[0] == 20 and encoded_dict['pooled_prompt_embeds'].shape[1] == 1280
