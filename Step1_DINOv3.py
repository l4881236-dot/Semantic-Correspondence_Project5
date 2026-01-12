import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModel

# Step1: DINOv3
#----------------------------------------------------------
# Basic configurations
ROOT = Path("SPair-71k")          # Where is your SPAIR-71k dataset?
SPLIT = "test"                   # Which one you want to test:trn / val / test

MAX_PAIRS = None                # Choose to limit number of pairs for quick test, or None for all

DINO_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
PATCH_SIZE = 16                  # DINOv3 ViT-S/16 uses 16x16 patches
RESIZE_TO = 512                  # must be multiple of PATCH_SIZE
THRESHOLDS = (0.05, 0.1, 0.2)


#----------------------------------------------------------
# ===============Device selection===================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

#----------------------------------------------------------
# ===============Define utility functions=====================
# Load json files
def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

# Get image paths from a json file
def image_path(cat,name):
    p = ROOT / "JPEGImages" / cat / name
    if p.exists():
        return p
    raise FileNotFoundError(f"Image not found: {p}")

# Load image and convert to RGB
def load_image(cat,name):
    return Image.open(image_path(cat,name)).convert("RGB")

# Find all pair json files in a split and return sorted list
def list_pair_jsons(split):
    d = ROOT / "PairAnnotation" / split
    files = sorted(d.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No json files found in: {d}")
    return files

# Resize keypoints
def scale_kps(kps, orig_size, new_size):
    W0, H0 = orig_size[0], orig_size[1]
    W1, H1 = new_size
    out = []
    for x, y in kps:
        out.append((x * W1 / W0, y * H1 / H0))
    return out

# Resize bounding box 
def scale_bbox(bbox, orig_size, new_size):
    xmin, ymin, xmax, ymax = bbox
    W0, H0 = orig_size[0], orig_size[1]
    W1, H1 = new_size
    sx = W1 / W0
    sy = H1 / H0
    return [xmin * sx, ymin * sy, xmax * sx, ymax * sy]

# Calculate bounding box norm
def bbox_norm_from_trg_bbox(trg_bbox_resized):
    xmin, ymin, xmax, ymax = trg_bbox_resized
    w = max(1e-6, (xmax - xmin))
    h = max(1e-6, (ymax - ymin))
    return max(w, h)

# Conversion between pixel and patch coordinates
def pixel_to_patch(x, y, patch_size):
    return int(x // patch_size), int(y // patch_size)
# Clamp patch coordinates to valid range
def clamp_patch(px, py, Wf, Hf):
    px = max(0, min(px, Wf - 1))
    py = max(0, min(py, Hf - 1))
    return px, py

def patch_to_pixel_center(px, py, patch_size):
    return (px + 0.5) * patch_size, (py + 0.5) * patch_size # 0.5 for locating patch center

# Cosine similarity
def cosine_sim_map(fs, Ft):
    fs = F.normalize(fs, dim=0)  # normalize source feature vector
    Ft = F.normalize(Ft, dim=-1)  # normalize each patch feature vector
    return torch.einsum("c,hwc->hw", fs, Ft) # compute dot product between fs and Ft for each patch

# Scoring for Difficulty level
def difficulty_score(pair: dict) -> int:
    v = int(pair.get("viewpoint_variation", 0))   # 0/1/2
    s = int(pair.get("scale_variation", 0))       # 0/1/2
    t = int(pair.get("truncation", 0))            # 0/1
    o = int(pair.get("occlusion", 0))             # 0/1
    return v + s + t + o  # 0~6


#----------------------------------------------------------
# Load DINOv3 model and processor
@torch.no_grad()
def extract_dinov3_dense_hf(model, processor, img_pil, resize_to, device):
    img = img_pil.resize((resize_to, resize_to))

    inputs = processor(
        images=img,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model(**inputs)

    # (1, 1 + R + Npatch, C)
    x = out.last_hidden_state
    tokens_after_cls = x[:, 1:, :]  # (1, R+Npatch, C)

    _, _, H, W = inputs["pixel_values"].shape
    Hf = H // PATCH_SIZE
    Wf = W // PATCH_SIZE
    Npatch = Hf * Wf

    total_after_cls = tokens_after_cls.shape[1]
    R = total_after_cls - Npatch  # register tokens
    if R < 0:
        raise RuntimeError(
            f"Token count smaller than expected patches. Got {total_after_cls}, expected >= {Npatch}. "
            f"(H,W)=({H},{W}), patch={PATCH_SIZE}"
        )

    patch_tokens = tokens_after_cls[:, R:R + Npatch, :]  # (1, Npatch, C)
    C = patch_tokens.shape[-1]
    feat = patch_tokens[0].reshape(Hf, Wf, C).contiguous()
    return feat


#----------------------------------------------------------
# ==========Define statistics functions=========================
# Create a new empty stats bucket
def new_bucket():
    return {
        "pairs_used": 0,                       
        "kp_total": 0,                         
        "kp_correct": {T: 0 for T in THRESHOLDS},
        "img_pck": {T: [] for T in THRESHOLDS} 
    }

# Add pair result to a bucket
def add_pair_result(bucket, pair_correct, pair_total):
    bucket["pairs_used"] += 1
    bucket["kp_total"] += pair_total
    for T in THRESHOLDS:
        bucket["kp_correct"][T] += pair_correct[T]
        bucket["img_pck"][T].append(pair_correct[T] / pair_total)

# Print bucket report
def bucket_report(title, bucket):
    print(f"\n--- {title} ---")
    print(f"Used pairs: {bucket['pairs_used']}")
    print(f"Valid keypoints total: {bucket['kp_total']}")

    print("\nPer-keypoint PCK (global over keypoints):")
    for T in THRESHOLDS:
        pck = bucket["kp_correct"][T] / bucket["kp_total"] if bucket["kp_total"] > 0 else float("nan")
        print(f"  PCK@{T}: {pck:.4f}")

    print("\nPer-image PCK (mean over pairs):")
    for T in THRESHOLDS:
        vals = bucket["img_pck"][T]
        if len(vals) == 0:
            print(f"  PCK@{T}: nan (no pairs)")
        else:
            print(f"  PCK@{T}: {mean(vals):.4f} ")


#----------------------------------------------------------
#main
def evaluate_full():
    pair_files = list_pair_jsons(SPLIT)

    if MAX_PAIRS is not None:
        pair_files = pair_files[:MAX_PAIRS]

    print(f"Split={SPLIT}  Num_json={len(pair_files)}")
    print(f"Device={DEVICE}  RESIZE_TO={RESIZE_TO}  PATCH_SIZE={PATCH_SIZE}")
    print(f"Model={DINO_NAME}")

    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(DINO_NAME, token=True)
    model = AutoModel.from_pretrained(DINO_NAME, token=True).to(DEVICE).float().eval()

    # Initialize buckets for statistics
    global_bucket = new_bucket()
    cat_buckets = defaultdict(new_bucket)
    diff_buckets = defaultdict(new_bucket)

    for pair_path in tqdm(pair_files, desc=f"Evaluating {SPLIT}"):
        # Load pair json
        pair = load_json(pair_path)
        cat = pair["category"]

        # Get difficulty level 0~6
        dlevel = difficulty_score(pair)

        # Load source and target images
        src_img = load_image(cat, pair["src_imname"])
        trg_img = load_image(cat, pair["trg_imname"])

        # Resize keypoints
        src_kps_r = scale_kps(pair["src_kps"], pair["src_imsize"], (RESIZE_TO, RESIZE_TO))
        trg_kps_r = scale_kps(pair["trg_kps"], pair["trg_imsize"], (RESIZE_TO, RESIZE_TO))

        # Normalizer
        trg_bbox_r = scale_bbox(pair["trg_bndbox"], pair["trg_imsize"], (RESIZE_TO, RESIZE_TO))
        norm = bbox_norm_from_trg_bbox(trg_bbox_r)

        # Extract DINOv3 dense features
        Fs = extract_dinov3_dense_hf(model, processor, src_img, RESIZE_TO, DEVICE)
        Ft = extract_dinov3_dense_hf(model, processor, trg_img, RESIZE_TO, DEVICE)
        Hf, Wf, _ = Ft.shape

        # Starting statistics for this pair
        pair_correct = {T: 0 for T in THRESHOLDS}
        pair_total = 0

        for (x_s, y_s), (x_t_gt, y_t_gt) in zip(src_kps_r, trg_kps_r):
            if x_s < 0 or y_s < 0 or x_t_gt < 0 or y_t_gt < 0:
                continue # skip invalid keypoints

            px_s, py_s = pixel_to_patch(x_s, y_s, PATCH_SIZE)
            px_s, py_s = clamp_patch(px_s, py_s, Wf, Hf)
            fs = Fs[py_s, px_s]  # Source feature vector from DiNOv3

            sim_map = cosine_sim_map(fs, Ft)
            best = int(torch.argmax(sim_map.view(-1))) # torch.argmax returns the indices of the maximum value of all elements in the input tensor.
            py_p, px_p = best // Wf, best % Wf #recover 2d coords
            x_t_pred, y_t_pred = patch_to_pixel_center(px_p, py_p, PATCH_SIZE)

            d = ((x_t_pred - x_t_gt) ** 2 + (y_t_pred - y_t_gt) ** 2) ** 0.5

            for T in THRESHOLDS:
                if (d / norm) < T:
                    pair_correct[T] += 1

            pair_total += 1

        # Update statistics buckets
        if pair_total > 0:
            add_pair_result(global_bucket, pair_correct, pair_total)
            add_pair_result(cat_buckets[cat], pair_correct, pair_total)
            add_pair_result(diff_buckets[dlevel], pair_correct, pair_total)

    # Output
    print("\n==================== ALL PAIRS (GLOBAL) ====================")
    bucket_report("All pairs", global_bucket)

    print("\n==================== PER-CATEGORY ====================")
    for cat in sorted(cat_buckets.keys()):
        b = cat_buckets[cat]

        print(f"\nCategory: {cat}")
        for T in THRESHOLDS:
            kp_pck = b["kp_correct"][T] / b["kp_total"] if b["kp_total"] > 0 else float("nan")
            img_vals = b["img_pck"][T]
            img_mean = mean(img_vals) if len(img_vals) else float("nan")
            print(f"  PCK@{T}: per-keypoint={kp_pck:.4f} | per-image={img_mean:.4f}")

    # If you want you can indent from here to print for difficulty of category
    print("\n==================== PER-DIFFICULTY (0 easiest -> 6 hardest) ====================")
    for d in range(0, 7):
        b = diff_buckets[d]
        if b["pairs_used"] == 0:
            print(f"\nDifficulty {d}: no data")
            continue
        print(f"\nDifficulty {d}: pairs={b['pairs_used']}  keypoints={b['kp_total']}")
        for T in THRESHOLDS:
            kp_pck = b["kp_correct"][T] / b["kp_total"] if b["kp_total"] > 0 else float("nan")
            img_vals = b["img_pck"][T]
            img_mean = mean(img_vals) if len(img_vals) else float("nan")
            print(f"  PCK@{T}: per-keypoint={kp_pck:.4f} | per-image={img_mean:.4f}")

    print("\nDone.")



if __name__ == "__main__":
    evaluate_full()


