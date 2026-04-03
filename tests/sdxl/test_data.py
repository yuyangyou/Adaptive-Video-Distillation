from causvid.data import TextDataset
import torch

dataset = TextDataset("sample_dataset/captions_coco14_test.txt")
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=False, drop_last=True)

for batch in dataloader:
    print(
        f"batch element type {type(batch[0])} batch length {len(batch)} batch first element {batch[0]}")
    break
