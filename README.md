# Adaptive Video Distillation
### Mitigating Oversaturation and Temporal Collapse in Few-Step Generation

[Project Page](https://Adaptive-Video-Distillation.github.io/)  

<video width="480" height="270" controls>
  <source src="docs/sample.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

> **Adaptive Video Distillation**  
> Yuyang You*, Yongzhi Li*, Jiahui Li, Yadong Mu, Quan Chen, Peng ...  
> *CVPR 2026*  

---

## Overview

This is the official repository for ADV (Adaptive Video Distillation) — a video model distillation method based on DMD（Distribution Matching Distillation）. It addresses oversaturation and slow-motion issues in video generation model distillation, and is capable of learning from new data during distillation training.


## Environment Setup

```bash
conda create -n AVD python=3.10 -y
conda activate AVD
pip install torch torchvision 
pip install -r requirements.txt 
python setup.py develop
```

Also download the Wan base models from [here](https://github.com/Wan-Video/Wan2.1) and save it to wan_models/Wan2.1-T2V-1.3B/

## Inference Example 

First download the toy checkpoints: [model](https://huggingface.co/adaptive-video-distillation/ADV/tree/main).


### Inference Script

```bash 
python ./tests/wan/test_bidirectional_fewstep.py
```

## Training and Evaluation  

### Dataset Preparation 

We use the [MixKit Dataset](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/all_mixkit) (6K videos) as a toy example for distillation. 

To prepare the dataset, follow these steps. You can also download the final LMDB dataset from [here](https://huggingface.co/tianweiy/CausVid/tree/main/mixkit_latents_lmdb)

```bash
# download and extract video from the Mixkit dataset 
python distillation_data/download_mixkit.py  --local_dir XXX 

# convert the video to 480x832x81 
python distillation_data/process_mixkit.py --input_dir XXX  --output_dir XXX --width 832   --height 480  --fps 16 

# precompute the vae latent 
torchrun --nproc_per_node 8 distillation_data/compute_vae_latent.py --input_video_folder XXX  --output_latent_folder XXX   --info_path sample_dataset/video_mixkit_6484_caption.json

# combined everything into a lmdb dataset 
python causvid/ode_data/create_lmdb_iterative.py   --data_path XXX  --lmdb_path XXX
```

## Training 

Please first modify the wandb account information in the respective config.  

Bidirectional DMD Training

```bash
torchrun --nnodes 1 --nproc_per_node=8 --master_port 29502 \
    causvid/train_distillation_adv.py \
    --config_path configs/wan_bidirectional_dmd.yaml

```

## Citation 

Here is a arxiv version citation bib：

```bib
@inproceedings{you2026adaptive,
  title={Adaptive video distillation: Mitigating oversaturation and temporal collapse in few-step generation},
  author={You, Yuyang and Li, Yongzhi and Li, Jiahui and Mu, Yadong and Chen, Quan and Jiang, Peng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={43429--43439},
  year={2026}
}

```

## Acknowledgments

Our implementation is largely based on the [Causvid](https://github.com/tianweiy/CausVid) and [Wan](https://github.com/Wan-Video/Wan2.1) model suite. Thanks a lot!
