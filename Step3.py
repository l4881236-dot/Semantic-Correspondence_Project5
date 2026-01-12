import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Step3: soft-argmax keypoint matching with DINOv2 features
#----------------------------------------------------------
# Basic configurations
ROOT = Path("SPair-71k")
SPLIT = "test"

DINO_NAME = "dinov2_vitb14"
PATCH_SIZE = 14
RESIZE_TO = 518
THRESHOLDS = (0.05, 0.1, 0.2)

WINDOW_RADIUS = 2
BETA = 10.0


#----------------------------------------------------------
# ===============Device====================================
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is NOT available!")

DEVICE = torch.device("cuda")
print("Using GPU:", torch.cuda.get_device_name(0))


# ----------------------------------------------------------
# Image transforms
to_tensor = transforms.Compose([
    transforms.Resize((RESIZE_TO, RESIZE_TO)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


#----------------------------------------------------------
# ===============Define utility functions=====================
def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def image_path(cat, name):
    return ROOT / "JPEGImages" / cat / name


def load_image(cat, name):
    return Image.open(image_path(cat, name)).convert("RGB")


def list_pair_jsons(split):
    return sorted((ROOT / "PairAnnotation" / split).glob("*.json"))


# ----------------------------------------------------------
# ===============Resize pictures=====================
def scale_kps(kps, orig_size, new_size):
    
    W0, H0 = orig_size[:2]   
    W1, H1 = new_size
    return [(x * W1 / W0, y * H1 / H0) for x, y in kps]


def scale_bbox(bbox, orig_size, new_size):
    xmin, ymin, xmax, ymax = bbox
    W0, H0 = orig_size[:2]   
    W1, H1 = new_size
    return [
        xmin * W1 / W0,
        ymin * H1 / H0,
        xmax * W1 / W0,
        ymax * H1 / H0,
    ]


def bbox_norm_from_trg_bbox(b):
    xmin, ymin, xmax, ymax = b
    return max(xmax - xmin, ymax - ymin, 1e-6)


def pixel_to_patch(x, y):
    return int(x // PATCH_SIZE), int(y // PATCH_SIZE)


def clamp_patch(px, py, Wf, Hf):
    return max(0, min(px, Wf - 1)), max(0, min(py, Hf - 1))


def patch_to_pixel_center(px, py):
    return (px + 0.5) * PATCH_SIZE, (py + 0.5) * PATCH_SIZE


# Cosine similarity
def cosine_sim_map(fs, Ft):
    fs = F.normalize(fs, dim=0)
    Ft = F.normalize(Ft, dim=-1)
    return torch.einsum("c,hwc->hw", fs, Ft)


# ----------------------------------------------------------
# =================WINDOW SOFT-ARGMAX=======================

def window_soft_argmax(sim, peak_xy, radius, beta):
    Hf, Wf = sim.shape
    px, py = peak_xy

    x0 = max(px - radius, 0)
    x1 = min(px + radius + 1, Wf)
    y0 = max(py - radius, 0)
    y1 = min(py + radius + 1, Hf)

    window = sim[y0:y1, x0:x1].reshape(-1)
    weights = torch.softmax(beta * window, dim=0)

    xs = torch.arange(x0, x1, device=sim.device).repeat(y1 - y0)
    ys = torch.arange(y0, y1, device=sim.device).unsqueeze(1).repeat(
        1, x1 - x0
    ).reshape(-1)

    px_ref = (weights * xs).sum()
    py_ref = (weights * ys).sum()

    return px_ref, py_ref


#----------------------------------------------------------
# Load DINOv2 model
@torch.no_grad()
def extract_dinov2_dense(model, img_pil):
    x = to_tensor(img_pil).unsqueeze(0).to(DEVICE)
    pt = model.forward_features(x)["x_norm_patchtokens"]
    Hf = Wf = int(pt.shape[1] ** 0.5)
    return pt[0].reshape(Hf, Wf, -1).contiguous()


#----------------------------------------------------------
# ==========Define statistics functions=========================
# Create a new empty stats bucket
def new_bucket():
    return {T: {"correct": 0, "total": 0} for T in THRESHOLDS}


def update_bucket(bucket, d, norm):
    for T in THRESHOLDS:
        if (d / norm) < T:
            bucket[T]["correct"] += 1
        bucket[T]["total"] += 1


def print_bucket(title, bucket):
    print(f"\n--- {title} ---")
    for T in THRESHOLDS:
        pck = bucket[T]["correct"] / bucket[T]["total"]
        print(f"PCK@{T}: {pck:.4f}")


#----------------------------------------------------------
#main
def evaluate():
    pair_files = list_pair_jsons(SPLIT)

   
    print(f"Total test pairs = {len(pair_files)}\n")

    model = torch.hub.load(
        "facebookresearch/dinov2", DINO_NAME
    ).to(DEVICE).eval()

    bucket_argmax = new_bucket()
    bucket_soft = new_bucket()

    for pair_path in tqdm(pair_files, desc="Evaluating"):
        pair = load_json(pair_path)
        cat = pair["category"]

        src = load_image(cat, pair["src_imname"])
        trg = load_image(cat, pair["trg_imname"])

        src_kps = scale_kps(
            pair["src_kps"], pair["src_imsize"], (RESIZE_TO, RESIZE_TO)
        )
        trg_kps = scale_kps(
            pair["trg_kps"], pair["trg_imsize"], (RESIZE_TO, RESIZE_TO)
        )

        norm = bbox_norm_from_trg_bbox(
            scale_bbox(
                pair["trg_bndbox"],
                pair["trg_imsize"],
                (RESIZE_TO, RESIZE_TO),
            )
        )

        Fs = extract_dinov2_dense(model, src)
        Ft = extract_dinov2_dense(model, trg)
        Hf, Wf, _ = Ft.shape

        for (xs, ys), (xt, yt) in zip(src_kps, trg_kps):
            if xs < 0 or ys < 0 or xt < 0 or yt < 0:
                continue

            px, py = clamp_patch(
                *pixel_to_patch(xs, ys), Wf, Hf
            )
            fs = Fs[py, px]
            sim = cosine_sim_map(fs, Ft)

            # ---------- ARGMAX ----------
            best = int(torch.argmax(sim))
            py_p, px_p = best // Wf, best % Wf
            x_a, y_a = patch_to_pixel_center(px_p, py_p)
            d_a = ((x_a - xt) ** 2 + (y_a - yt) ** 2) ** 0.5
            update_bucket(bucket_argmax, d_a, norm)

            # ---------- WINDOW SOFT-ARGMAX ----------
            px_s, py_s = window_soft_argmax(
                sim, (px_p, py_p), WINDOW_RADIUS, BETA
            )
            x_s = (px_s + 0.5) * PATCH_SIZE
            y_s = (py_s + 0.5) * PATCH_SIZE
            d_s = ((x_s - xt) ** 2 + (y_s - yt) ** 2) ** 0.5
            update_bucket(bucket_soft, d_s, norm)

    print("\n================ RESULT COMPARISON ================")
    print_bucket("ARGMAX", bucket_argmax)
    print_bucket("WINDOW SOFT-ARGMAX", bucket_soft)


if __name__ == "__main__":
    evaluate()
