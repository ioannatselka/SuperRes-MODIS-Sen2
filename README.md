# SuperRes-MODIS-Sen2

*A Deep Learning Approach to Image Resolution Enhancement: Super-Resolution of Multispectral MODIS Imagery Utilizing Sentinel-2 Products (Tselka I., 2024)*

---

## Overview

This repository contains the full implementation of my Master’s Thesis, which explores the use of deep learning models to enhance the spatial resolution of Earth Observation (EO) imagery.  
You can access the full thesis [here](https://dspace.lib.ntua.gr/xmlui/handle/123456789/60512).


---

## Models Implemented

The following state-of-the-art Super-Resolution models are implemented and evaluated:

- **SRCNN**: Super-Resolution Convolutional Neural Network
- **FSRCNN**: Fast SRCNN
- **EDSR**: Enhanced Deep Super-Resolution
- **ESPCN**: Efficient Sub-Pixel Convolutional Network
- **MRUNet**: Multi-Resolution U-Net (custom)
- **VDSR**: Very Deep Super-Resolution

---

## Experimental Setup

![image](https://github.com/user-attachments/assets/d2e33262-a516-45a9-8d8d-ae33f9b00134)


- **Dataset**: The FLOGA dataset pairing MODIS and Sentinel-2 imagery, preprocessed and aligned. You can access the dataset [here](https://github.com/Orion-AI-Lab/FLOGA).
- **Super-Resolution Scales**:
  - `x2`: Upsampling from 500m → 240m
  - `x4`: Upsampling from 500m → 120m
  - `x8`: Upsampling from 500m → 60m
- **Evaluation Metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)

---

## Results

### 🔹 Scaling Factor ×8 (Target Resolution: 60m)

![Super-Resolution Comparison (x8)](https://github.com/user-attachments/assets/0c6a7619-7161-468b-af04-68f42a663fdc)

---

### 🔹 Scaling Factor ×4 (Target Resolution: 120m)

![image](https://github.com/user-attachments/assets/a3ce96e8-05c7-47f1-9494-f78f9fd0596f)

---

### 🔹 Scaling Factor ×2 (Target Resolution: 240m)

![image](https://github.com/user-attachments/assets/a3308e2e-a7cc-4342-89ce-27f766066863)

---

## Brief discussion

#### Model Ranking Summary (by PSNR/SSIM)

| Rank | Model   | Notes |
|------|---------|-------|
|  1   | MRUNeT  | Best overall across all scales and bands |
|  2   | VDSR    | Strong in SWIR/NIR; good general performance |
|  3   | EDSR    | Competitive on ×2 and ×4 scales |

> 📌 Based on both visual and quantitative evaluations, **MRUNeT** is the most effective model for super-resolving MODIS imagery in this study.

---

## Repository Structure

<pre>
├── configs.json # Configuration file for datasets and models
├── dataset_utils.py # FLOGA dataset set-up & creation of HR-LR image pairs
├── models/ # All SR model implementations
├── run_experiment.py # Model training script
├── run_inference.py # Inference script with target resolution 60m
├── run_inferencex4.py # Inference script with target resolution 120m
├── run_inferencex2.py # Inference script with target resolution 240m
├── visualize.py # Compare outputs from multiple models
├── utils.py # Shared helpers (normalization, config loading, etc.)
└── README.md
<pre/>

