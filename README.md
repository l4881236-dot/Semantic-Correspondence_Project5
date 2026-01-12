# Semantic-Correspondence_Project5
s346897_s348266_s336755_project5
# Semantic Correspondence with Visual Foundation Models

This repository contains the official implementation for the course project **"Semantic Correspondence with Visual Foundation Models"**.

We explore the capabilities of Foundation Models (DINOv2, SAM, Stable Diffusion) on the semantic correspondence task and propose a **Light Fine-tuning** strategy that significantly improves performance on the **SPair-71k** benchmark.

##  Team Members
- **Student 1**: s346897 Jinyu Ai 
- **Student 2**: s348266 Shixu Zhang 
- **Student 3**: s336755 Hao Li 

---

##  Setup & Installation

### 1. Install Dependencies
Ensure you have Python 3.8+ and PyTorch installed. Then run:

```bash
pip install -r requirements.txt
# Download the dataset
wget [http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz](http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz)

# Extract (This will create a 'SPair-71k' folder)
tar -xvf SPair-71k.tar.gz
