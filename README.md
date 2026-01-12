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

### 2. Dataset Preparation (Crucial ⚠️)
The project relies on the **SPair-71k** dataset. Please execute the following commands in the root directory of this project to download and set it up automatically.

```bash
# 1. Download the SPair-71k dataset (approx. 1.2GB)
wget [http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz](http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz)

# 2. Extract the dataset
# This will create a folder named 'SPair-71k' containing JPEGImages and PairAnnotation
tar -xvf SPair-71k.tar.gz

# 3. (Optional) Clean up the compressed file
rm SPair-71k.tar.gz

```bash
pip install -r requirements.txt
