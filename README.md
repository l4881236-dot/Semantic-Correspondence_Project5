# Semantic Correspondence with Visual Foundation Models

This repository contains the official implementation for the course project **"Semantic Correspondence with Visual Foundation Models"**.

We explore the capabilities of Foundation Models (DINOv2, SAM, Stable Diffusion) on the semantic correspondence task and propose a **Light Fine-tuning** strategy that significantly improves performance on the **SPair-71k** benchmark.

##  Team Members
- **Student 1**: s346897 Jinyu Ai
- **Student 2**: s336755 Hao Li
- **Student 3**: s348266 Shixu Zhang

---

##  Setup & Installation

### 1. Install Dependencies
Ensure you have Python 3.8+ and PyTorch installed. Then run:

```bash
pip install -r requirements.txt
###2. Dataset Preparation
# 1. Download the dataset
wget [http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz](http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz)

# 2. Extract it (This will create a folder named 'SPair-71k')
tar -xvf SPair-71k.tar.gz
###3. Download Model Checkpoints
# Download SAM weights to the root directory
wget [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
