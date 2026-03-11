# Adaptive Video Distillation
### Mitigating Oversaturation and Temporal Collapse in Few-Step Generation

[Project Page](https://yuyangyou.github.io/Adaptive-Video-Distillation.github.io/#)  

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

## Method
![main](docs/main.pdf)

## Environment Setup

```bash
conda create -n causvid python=3.10 -y
conda activate causvid
pip install torch torchvision 
pip install -r requirements.txt 
python setup.py develop
```

Also download the Wan base models from [here](https://github.com/Wan-Video/Wan2.1) and save it to wan_models/Wan2.1-T2V-1.3B/

## Inference Example 

First download the checkpoints: [Autoregressive Model](https://huggingface.co/tianweiy/CausVid/tree/main/autoregressive_checkpoint), [Bidirectional Model 1](https://huggingface.co/tianweiy/CausVid/tree/main/bidirectional_checkpoint1) or [Bidirectional Model 2](https://huggingface.co/tianweiy/CausVid/tree/main/bidirectional_checkpoint2) (performs slightly better). 


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
torchrun --nnodes 8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint $MASTER_ADDR causvid/train_distillation_adv.py \
    --config_path  configs/wan_bidirectional_dmd.yaml 
```



## Citation 

If you find CausVid useful or relevant to your research, please kindly cite our papers:

```bib
@inproceedings{.
}

```

## Acknowledgments

Our implementation is largely based on the [Wan](https://github.com/Wan-Video/Wan2.1) model suite.

