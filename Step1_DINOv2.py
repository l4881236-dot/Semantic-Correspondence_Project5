import json
import random
from pathlib import Path
from collections import defaultdict
from statistics import mean

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


# ----------------------------------------------------------
# Basic configurations
ROOT = Path("SPair-71k")
SPLIT = "test"

DINO_NAME = "dinov2_vitb14"
PATCH_SIZE = 14
RESIZE_TO = 518
THRESHOLDS = (0.05, 0.1, 0.2)

NUM_SAMPLES = None      



# ----------------------------------------------------------
# 🔥 FORCE GPU (CUDA)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is NOT available!")

DEVICE = torch.device("cuda")
print("✅ Using GPU:", torch.cuda.get_device_name(0))


# ----------------------------------------------------------
# Image transform (DINOv2 standard)
to_tensor = transforms.Compose([
    transforms.Resize((RESIZE_TO, RESIZE_TO)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


# ----------------------------------------------------------
# Utility functions
def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def image_path(cat, name):
    p = ROOT / "JPEGImages" / cat / name
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_image(cat, name):
    return Image.open(image_path(cat, name)).convert("RGB")


def list_pair_jsons(split):
    d = ROOT / "PairAnnotation" / split
    files = sorted(d.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No json files in {d}")
    return files


# -------------------- SPair-71k FIX --------------------
def scale_kps(kps, orig_size, new_size):
    W0, H0 = orig_size[0], orig_size[1]
    W1, H1 = new_size
    return [(x * W1 / W0, y * H1 / H0) for x, y in kps]


def scale_bbox(bbox, orig_size, new_size):
    xmin, ymin, xmax, ymax = bbox
    W0, H0 = orig_size[0], orig_size[1]
    W1, H1 = new_size
    sx, sy = W1 / W0, H1 / H0
    return [xmin * sx, ymin * sy, xmax * sx, ymax * sy]


def bbox_norm_from_trg_bbox(b):
    xmin, ymin, xmax, ymax = b
    return max(xmax - xmin, ymax - ymin, 1e-6)


def pixel_to_patch(x, y):
    return int(x // PATCH_SIZE), int(y // PATCH_SIZE)


def clamp_patch(px, py, Wf, Hf):
    return max(0, min(px, Wf - 1)), max(0, min(py, Hf - 1))


def patch_to_pixel_center(px, py):
    return (px + 0.5) * PATCH_SIZE, (py + 0.5) * PATCH_SIZE


def cosine_sim_map(fs, Ft):
    fs = F.normalize(fs, dim=0)
    Ft = F.normalize(Ft, dim=-1)
    return torch.einsum("c,hwc->hw", fs, Ft)


def difficulty_score(pair):
    return (
        int(pair.get("viewpoint_variation", 0)) +
        int(pair.get("scale_variation", 0)) +
        int(pair.get("truncation", 0)) +
        int(pair.get("occlusion", 0))
    )


# ----------------------------------------------------------
# DINOv2 dense feature extraction
@torch.no_grad()
def extract_dinov2_dense(model, img_pil):
    x = to_tensor(img_pil).unsqueeze(0).to(DEVICE)
    out = model.forward_features(x)
    pt = out["x_norm_patchtokens"]

    N = pt.shape[1]
    C = pt.shape[2]
    Hf = Wf = int(N ** 0.5)

    return pt[0].reshape(Hf, Wf, C).contiguous()


# ----------------------------------------------------------
# Statistics helpers
def new_bucket():
    return {
        "pairs_used": 0,
        "kp_total": 0,
        "kp_correct": {T: 0 for T in THRESHOLDS},
        "img_pck": {T: [] for T in THRESHOLDS},
    }


def add_pair_result(bucket, pair_correct, pair_total):
    bucket["pairs_used"] += 1
    bucket["kp_total"] += pair_total
    for T in THRESHOLDS:
        bucket["kp_correct"][T] += pair_correct[T]
        bucket["img_pck"][T].append(pair_correct[T] / pair_total)


def bucket_report(title, bucket):
    print(f"\n--- {title} ---")
    print(f"Pairs used: {bucket['pairs_used']}")
    print(f"Keypoints total: {bucket['kp_total']}")
    for T in THRESHOLDS:
        print(f"  PCK@{T}: {bucket['kp_correct'][T] / bucket['kp_total']:.4f}")


# ----------------------------------------------------------
#  Main evaluation
def evaluate_full():
    pair_files = list_pair_jsons(SPLIT)

    if NUM_SAMPLES is not None:
        pair_files = pair_files[:NUM_SAMPLES]

    print(f"Split={SPLIT} | Pairs={len(pair_files)}")
    print(f"Model={DINO_NAME} | Device={DEVICE}")

    model = torch.hub.load(
        "facebookresearch/dinov2",
        DINO_NAME
    ).to(DEVICE).eval()

    global_bucket = new_bucket()

    for pair_path in tqdm(pair_files, desc="Evaluating"):
        pair = load_json(pair_path)
        cat = pair["category"]

        src_img = load_image(cat, pair["src_imname"])
        trg_img = load_image(cat, pair["trg_imname"])

        src_kps = scale_kps(pair["src_kps"], pair["src_imsize"], (RESIZE_TO, RESIZE_TO))
        trg_kps = scale_kps(pair["trg_kps"], pair["trg_imsize"], (RESIZE_TO, RESIZE_TO))

        trg_bbox = scale_bbox(pair["trg_bndbox"], pair["trg_imsize"], (RESIZE_TO, RESIZE_TO))
        norm = bbox_norm_from_trg_bbox(trg_bbox)

        Fs = extract_dinov2_dense(model, src_img)
        Ft = extract_dinov2_dense(model, trg_img)
        Hf, Wf, _ = Ft.shape

        pair_correct = {T: 0 for T in THRESHOLDS}
        pair_total = 0

        for (xs, ys), (xt, yt) in zip(src_kps, trg_kps):
            if xs < 0 or ys < 0 or xt < 0 or yt < 0:
                continue

            px, py = pixel_to_patch(xs, ys)
            px, py = clamp_patch(px, py, Wf, Hf)

            fs = Fs[py, px]
            sim = cosine_sim_map(fs, Ft)

            best = int(torch.argmax(sim.view(-1)))
            py_p, px_p = best // Wf, best % Wf
            x_pred, y_pred = patch_to_pixel_center(px_p, py_p)

            d = ((x_pred - xt) ** 2 + (y_pred - yt) ** 2) ** 0.5

            for T in THRESHOLDS:
                if (d / norm) < T:
                    pair_correct[T] += 1

            pair_total += 1

        if pair_total > 0:
            add_pair_result(global_bucket, pair_correct, pair_total)

    print("\n================ GLOBAL =================")
    bucket_report("All pairs", global_bucket)
    print("\nDone.")


if __name__ == "__main__":
    evaluate_full()
