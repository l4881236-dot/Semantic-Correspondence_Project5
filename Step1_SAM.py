import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from segment_anything import sam_model_registry


# ----------------------------------------------------------
# Basic configurations
ROOT = Path("SPair-71k")
SPLIT = "test"
MAX_PAIRS = None

SAM_TYPE = "vit_b"
SAM_CKPT = "sam_vit_b_01ec64.pth"

IMG_SIZE = 1024          
PATCH_SIZE = 16
THRESHOLDS = (0.05, 0.1, 0.2)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------
# Dataset utils
def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def image_path(cat, name):
    return ROOT / "JPEGImages" / cat / name

def load_image(cat, name):
    return Image.open(image_path(cat, name)).convert("RGB")

def list_pair_jsons(split):
    return sorted((ROOT / "PairAnnotation" / split).glob("*.json"))

# SPair-71k imsize = [H, W, C]
def scale_kps(kps, orig_size, new_size):
    H0, W0 = orig_size[:2]
    W1, H1 = new_size
    return [(x * W1 / W0, y * H1 / H0) for x, y in kps]

def scale_bbox(bbox, orig_size, new_size):
    xmin, ymin, xmax, ymax = bbox
    H0, W0 = orig_size[:2]
    W1, H1 = new_size
    sx, sy = W1 / W0, H1 / H0
    return [xmin * sx, ymin * sy, xmax * sx, ymax * sy]

def bbox_norm_from_trg_bbox(b):
    xmin, ymin, xmax, ymax = b
    return max(1e-6, xmax - xmin, ymax - ymin)


# ----------------------------------------------------------
# Geometry utils
def pixel_to_patch(x, y):
    return int(x // PATCH_SIZE), int(y // PATCH_SIZE)

def clamp_patch(px, py, Wf, Hf):
    return max(0, min(px, Wf - 1)), max(0, min(py, Hf - 1))

def patch_to_pixel_center(px, py):
    return (px + 0.5) * PATCH_SIZE, (py + 0.5) * PATCH_SIZE


# ----------------------------------------------------------
# ✅ SAM feature extraction (NO ResizeLongestSide, NO pad)
@torch.no_grad()
def extract_sam_dense(sam, img_pil):
    # 强制 resize 到 1024x1024
    img = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img = np.array(img).astype(np.float32)

    img = torch.from_numpy(img).permute(2, 0, 1).to(DEVICE)

    # SAM normalization (官方)
    mean = torch.tensor([123.675, 116.28, 103.53], device=DEVICE).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], device=DEVICE).view(3, 1, 1)
    img = (img - mean) / std

    img = img.unsqueeze(0)  # (1,3,1024,1024)

    feat = sam.image_encoder(img)      # (1,256,64,64)
    feat = feat[0].permute(1, 2, 0)    # (64,64,256)

    # ⭐ patch-level L2 norm
    feat = F.normalize(feat, dim=-1)

    return feat


# ----------------------------------------------------------
# Stats
def new_bucket():
    return {
        "pairs": 0,
        "kp_total": 0,
        "kp_correct": {T: 0 for T in THRESHOLDS},
    }

def add_pair(bucket, correct, total):
    bucket["pairs"] += 1
    bucket["kp_total"] += total
    for T in THRESHOLDS:
        bucket["kp_correct"][T] += correct[T]

def report(bucket):
    for T in THRESHOLDS:
        print(f"PCK@{T}: {bucket['kp_correct'][T] / bucket['kp_total']:.4f}")


# ----------------------------------------------------------
# Main
def evaluate_full():
    pair_files = list_pair_jsons(SPLIT)
    if MAX_PAIRS:
        pair_files = pair_files[:MAX_PAIRS]

    print(f"Split={SPLIT} | Pairs={len(pair_files)}")
    print(f"Backbone=SAM-{SAM_TYPE}")
    print(f"Device={DEVICE}")

    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT)
    sam.to(DEVICE).eval()

    global_bucket = new_bucket()

    for pair_path in tqdm(pair_files):
        pair = load_json(pair_path)
        cat = pair["category"]

        src_img = load_image(cat, pair["src_imname"])
        trg_img = load_image(cat, pair["trg_imname"])

        src_kps = scale_kps(pair["src_kps"], pair["src_imsize"], (IMG_SIZE, IMG_SIZE))
        trg_kps = scale_kps(pair["trg_kps"], pair["trg_imsize"], (IMG_SIZE, IMG_SIZE))
        trg_bbox = scale_bbox(pair["trg_bndbox"], pair["trg_imsize"], (IMG_SIZE, IMG_SIZE))
        norm = bbox_norm_from_trg_bbox(trg_bbox)

        Fs = extract_sam_dense(sam, src_img)
        Ft = extract_sam_dense(sam, trg_img)
        Hf, Wf, _ = Ft.shape  

        pair_correct = {T: 0 for T in THRESHOLDS}
        pair_total = 0

        for (xs, ys), (xt, yt) in zip(src_kps, trg_kps):
            if xs < 0 or ys < 0 or xt < 0 or yt < 0:
                continue

            px, py = pixel_to_patch(xs, ys)
            px, py = clamp_patch(px, py, Wf, Hf)

            fs = Fs[py, px]
            sim = torch.einsum("c,hwc->hw", fs, Ft)

            best = torch.argmax(sim)
            py_p, px_p = best // Wf, best % Wf
            xp, yp = patch_to_pixel_center(px_p, py_p)

            d = ((xp - xt) ** 2 + (yp - yt) ** 2) ** 0.5
            for T in THRESHOLDS:
                if d / norm < T:
                    pair_correct[T] += 1

            pair_total += 1

        if pair_total > 0:
            add_pair(global_bucket, pair_correct, pair_total)

    print("\n================ GLOBAL =================")
    report(global_bucket)
    print("Done.")


if __name__ == "__main__":
    evaluate_full()
