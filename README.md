# Semantic Correspondence with Visual Foundation Models

## Team Members
- **Student 1**: s346897 Jinyu Ai
- **Student 2**: s336755 Hao Li
- **Student 3**: s348266 Shixu Zhang

## Links
- **Official Dataset (SPair-71k)**: [https://cvlab.postech.ac.kr/research/SPair-71k/](https://cvlab.postech.ac.kr/research/SPair-71k/)

---

## 🛠️ Project Setup

To reproduce our results, please follow the installation steps below.

### 1. Install Dependencies

pip install -r requirements.txt
2. Download Dataset
The project relies on the SPair-71k dataset. Run the following commands to download and extract it to the root directory:
# Download dataset (approx. 1.2GB)
wget [http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz](http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz)

# Extract (This creates the 'SPair-71k' folder)
tar -xvf SPair-71k.tar.gz

# Cleanup compressed file
rm SPair-71k.tar.gz
3. Download Model Checkpoints
Required for Step 1 (SAM) evaluation.
# Download SAM ViT-B weights to the root directory
wget [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
