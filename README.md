# Hierarchical Mesh-Based Geometric Reconstruction of Human Pose and Shape

<img style="max-width: 100%;" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/wax.png" alt="Title Overview">

## Overview

We propose a hierarchical mesh-based framework for accurate 3D human pose and shape reconstruction from a single RGB image. A coarse-to-fine alignment strategy integrates multi-level semantic and geometric cues to refine body structure and preserve mesh fidelity. Experiments on 3DPW and COCO demonstrate improved performance over existing methods.

# 👁️💬 Architecture

## 🧾 Framework Overview

The proposed method employs hierarchical mesh alignment with a coarse-to-fine refinement strategy, progressively integrating semantic, structural, and geometric cues to reconstruct accurate 3D human pose and shape.

<img style="max-width: 100%;" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/method.jpg" alt="Overview">

## Demo

Qualitative comparison of video matting results on challenging real-world sequences.

<table>
  <tr>
    <td style="text-align: center;">
      <p>Input Video</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/video1.gif" alt="input video">
    </td>
    <td style="text-align: center;">
      <p>Reconstruction</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/video1_result.gif" alt="Foreground">
    </td>
        <td style="text-align: center;">
      <p>Input Video</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/video2.gif" alt="input video">
    </td>
    <td style="text-align: center;">
      <p>Reconstruction</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/video2_result.gif" alt="Foreground">
    </td>
  </tr>
  <tr>
    <td style="text-align: center;">
      <p>Input Image</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/Picture1.png" alt="input video">
    </td>
    <td style="text-align: center;">
      <p>Input Image</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/Picture3.png" alt="Foreground">
    </td>
    <td style="text-align: center;">
      <p>Input Image</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/Picture5.png" alt="input video">
    </td>
    <td style="text-align: center;">
      <p>Input Image</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/Picture7.png" alt="Foreground">
    </td>
  </tr>
    <tr>
    <td style="text-align: center;">
      <p>Reconstruction</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/Picture2.png" alt="input video">
    </td>
    <td style="text-align: center;">
      <p>Reconstruction</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/Picture4.png" alt="Foreground">
    </td>
        <td style="text-align: center;">
      <p>Reconstruction</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/Picture6.png" alt="input video">
    </td>
    <td style="text-align: center;">
      <p>Reconstruction</p>
      <img width="180" src="https://github.com/rizwanchouhan/hmrs/blob/main/resources/Picture8.png" alt="Foreground">
    </td>
  </tr>
</table>

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone git@github.com:rizwanchouhan/hmrs.git
cd hmrs
```

### 2. Create the Conda Environment

We recommend using Python 3.10.

```bash
conda create -n hmrs python=3.10 -y
conda activate hmrs
```

### 3. Install PyTorch

Install the PyTorch version compatible with your CUDA environment.

For example:

```bash
pip install torch torchvision
```

> For reproducible experiments, please use the PyTorch and CUDA versions specified in the release notes or environment file provided with the repository.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```
---

### Required Files

> Mesh Downsampling and DensePose UV Data
- Execute the following script to download `mesh_downsampling.npz` and DensePose UV data from other repositories:

```
bash fetch_data.sh
```
> SMPL Model Files
- Obtain the SMPL model files from [SMPL](https://smpl.is.tue.mpg.de) and [UP](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). Rename the model files as needed and place them in the `./data/smpl` directory.

> Preprocessed Data from SPIN
- Download the preprocessed data by following the instructions [here](https://github.com/nkolot/SPIN#fetch-data).

> Final Fits Data from SPIN
- Retrieve the final fits data as outlined [here](https://github.com/nkolot/SPIN#final-fits). Important Note: Using [EFT](https://github.com/facebookresearch/eft) fits for training is recommended. Compatible `.npz` files can be found [here](https://cloud.tsinghua.edu.cn/d/635c717375664cd6b3f5)

> Pre-trained Model
- Download the [pre-trained model](https://drive.google.com/file/d/1XMjZBsz-losAilG9ZEZQlZMPmrssDLBg/view?usp=sharing) and place it in the `./data/pretrained_model` directory.
- After gathering these necessary files, your `./data` directory structure should look like this:
```
./data
├── dataset_extras
│   └── .npz files
├── J_regressor_extra.npy
├── J_regressor_h36m.npy
├── mesh_downsampling.npz
├── pretrained_model
│   └── emo-body-lang_checkpoint.pt
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smpl_mean_params.npz
├── final_fits
│   └── .npy files
└── UV_data
    ├── UV_Processed.mat
    └── UV_symmetry_transforms.mat
```

## Preview of Demo Results:

### For Image Input:

```
python3 run_demo.py --checkpoint=data/pretrained_model/emo_body_lang_checkpoint.pt --img_file input/Picture5.png
```

<p align="center">
    <img style="max-width: 100%;" src="https://github.com/swerizwan/hmrs/blob/main/resources/image.png" alt="Overview">
</p>

### For Video Input:

```
python3 run_demo.py --checkpoint=data/pretrained_model/emo_body_lang_checkpoint.pt --vid_file input/dancer.mp4
```

<p align="center">
    <img style="max-width: 100%;" src="https://github.com/swerizwan/hmrs/blob/main/resources/image.gif" alt="Overview">
</p>


## Evaluation

### COCO

1. Download the preprocessed data [coco_2014_val.npz](https://drive.google.com/file/d/1ew77AaaOT3SAF0fZpfPrg02P5c9bzTHe/view?usp=sharing). Put it into the `./data/dataset_extras` directory. 

2. Run the COCO evaluation code.
```
python3 coco.py --checkpoint=data/pretrained_model/emo_body_lang_checkpoint.pt
```

### 3DPW

Run the evaluation code. Using `--dataset` to specify the evaluation dataset.
```
python3 main.py --checkpoint=data/pretrained_model/emo_body_lang_checkpoint.pt --dataset=3dpw --log_freq=20
```

## Training

We can monitor the training process by setting up a TensorBoard in the directory `./logs`.

```
CUDA_VISIBLE_DEVICES=0 python3 trainer.py --regressor emo_body_lang --single_dataset --misc TRAIN.BATCH_SIZE 64
```
