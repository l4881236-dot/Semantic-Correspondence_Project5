# Semantic Correspondence with Visual Foundation Models

## Team Members
- **Student 1**: s346897 Jinyu Ai
- **Student 2**: s336755 Hao Li
- **Student 3**: s348266 Shixu Zhang

## Links
- **Official Dataset (SPair-71k)**: [https://cvlab.postech.ac.kr/research/SPair-71k/](https://cvlab.postech.ac.kr/research/SPair-71k/)
## Run
Step 1: Zero-shot Baselines
Evaluate the off-the-shelf performance of Foundation Models.
python Step1_DINOv2.py  # DINOv2 (Baseline)
python Step1_DINOv3.py  # DINOv3
python Step1_SAM.py     # SAM (Segment Anything)
Step 2: Light Fine-tuning (Main Method)
Train the model using our Layer-wise Unfreezing strategy and evaluate PCK.
# Note: This requires a GPU.
python Step2.py
Step 3: Visualization
Visualize the keypoint predictions between source and target images.
python Step3.py
Step 4: Extension
Evaluate features from Stable Diffusion (Generative Model).
python Step4.SD.py
