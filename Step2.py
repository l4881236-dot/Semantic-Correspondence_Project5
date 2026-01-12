# merged_unfreeze_dinov2.py
import json
import random
import gc 
from pathlib import Path
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModel


# =============================================================================
# CONFIG 
ROOT = Path("SPair-71k")  
MODEL_NAME = "facebook/dinov2-base"

TRAIN_SPLIT = "trn"
EVAL_SPLIT = "val"   
THRESHOLDS = (0.05, 0.1, 0.2)

# DINOv2 ViT-B/14
PATCH_SIZE = 14
RESIZE_TO = 336  # 336/14=24

# progressive targets (cumulative)
TRAIN_TARGETS = [60000]
UNFREEZE_LIST = [1, 2, 4]

# training hyperparams
EPOCHS_PER_STAGE = 1     
LR = 1e-5
WEIGHT_DECAY = 1e-4
TEMP = 0.1

SEED = 42  # Fix random seed to ensure reproducible results
BATCH_SIZE = 8       
NUM_WORKERS = 4          

# speed controls
MAX_EVAL_PAIRS = None      
MAX_KP_PER_PAIR = None    
# =============================================================================

# DEVICE SETUP 
# -----------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("⚠️ CPU (slow)")

# =============================================================================
# IO helpers
def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def list_pair_jsons(split: str):
    d = ROOT / "PairAnnotation" / split
    files = sorted(d.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No pair jsons found in: {d}")
    return files

def image_path(cat: str, name: str):
    p = ROOT / "JPEGImages" / cat / name
    if p.exists():
        return p
    p2 = ROOT / "JPEGImages" / name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Image not found: {p}")

def load_image(cat: str, name: str):
    return Image.open(image_path(cat, name)).convert("RGB")


# =============================================================================
# geometry / scaling
def scale_kps(kps, orig_size, new_size):
    W0, H0 = float(orig_size[0]), float(orig_size[1])
    W1, H1 = float(new_size[0]), float(new_size[1])
    sx, sy = W1 / W0, H1 / H0

    out = []
    for x, y in kps:
        if x < 0 or y < 0:
            out.append([-1.0, -1.0])
        else:
            out.append([x * sx, y * sy])
    return out

def scale_bbox(bbox, orig_size, new_size):
    xmin, ymin, xmax, ymax = bbox
    W0, H0 = float(orig_size[0]), float(orig_size[1])
    W1, H1 = float(new_size[0]), float(new_size[1])
    sx, sy = W1 / W0, H1 / H0
    return [xmin * sx, ymin * sy, xmax * sx, ymax * sy]

def bbox_norm_from_trg_bbox(trg_bbox_resized):
    xmin, ymin, xmax, ymax = trg_bbox_resized
    w = max(1e-6, xmax - xmin)
    h = max(1e-6, ymax - ymin)
    return max(w, h)

def pixel_to_patch(x, y, patch_size):
    return int(x // patch_size), int(y // patch_size)

def clamp_patch(px, py, Wf, Hf):
    px = max(0, min(px, Wf - 1))
    py = max(0, min(py, Hf - 1))
    return px, py

def patch_to_pixel_center(px, py, patch_size):
    return (px + 0.5) * patch_size, (py + 0.5) * patch_size


# =============================================================================
# features + similarity
def cosine_sim_map(fs, Ft):
    fs = F.normalize(fs, dim=0)
    Ft_n = F.normalize(Ft, dim=-1)
    return torch.einsum("c,hwc->hw", fs, Ft_n)

def extract_dinov2_dense(model, processor, img_pil, resize_to, device, no_grad: bool):
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        img = img_pil.resize((resize_to, resize_to))
        inputs = processor(
            images=img,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = model(**inputs)
        x = out.last_hidden_state  # (1, 1+tokens, C)
        tokens_after_cls = x[:, 1:, :]

        _, _, H, W = inputs["pixel_values"].shape
        Hf = H // PATCH_SIZE
        Wf = W // PATCH_SIZE
        Npatch = Hf * Wf

        total_after_cls = tokens_after_cls.shape[1]
        R = total_after_cls - Npatch
        if R < 0:
            raise RuntimeError(f"Token count smaller than patches: total_after_cls={total_after_cls}, Npatch={Npatch}")

        patch_tokens = tokens_after_cls[:, R:R + Npatch, :]
        C = patch_tokens.shape[-1]
        feat = patch_tokens[0].reshape(Hf, Wf, C).contiguous()
        return feat


# =============================================================================
# GLOBAL PCK bucket
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

def print_global(title, bucket):
    print(f"\n--- {title} ---")
    print(f"pairs_used={bucket['pairs_used']}  kp_total={bucket['kp_total']}")
    for T in THRESHOLDS:
        kp_pck = bucket["kp_correct"][T] / bucket["kp_total"] if bucket["kp_total"] > 0 else float("nan")
        img_vals = bucket["img_pck"][T]
        img_mean = mean(img_vals) if len(img_vals) else float("nan")
        print(f"PCK@{T}: per-keypoint={kp_pck:.4f} | per-image={img_mean:.4f}")


# =============================================================================
# freeze/unfreeze
def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_last_blocks_auto(model: nn.Module, n_blocks: int, verbose: bool = True):
    modulelists = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.ModuleList)]
    if not modulelists:
        raise RuntimeError("Cannot locate transformer blocks: no ModuleList found.")
    modulelists.sort(key=lambda x: len(x[1]), reverse=True)
    chosen_name, blocks = modulelists[0]
    if verbose:
        print(f"[unfreeze] Using ModuleList '{chosen_name}' (len={len(blocks)}) -> unfreeze last {n_blocks}")
    for blk in blocks[-n_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    return chosen_name, len(blocks)

def print_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable}/{total} ({100*trainable/total:.4f}%)")


# =============================================================================
# loss
def kp_ce_loss(sim_map, x_t_gt, y_t_gt, Wf, Hf, temp: float):
    px_gt, py_gt = pixel_to_patch(x_t_gt, y_t_gt, PATCH_SIZE)
    px_gt, py_gt = clamp_patch(px_gt, py_gt, Wf, Hf)
    gt_index = py_gt * Wf + px_gt
    logits = (sim_map.reshape(-1) / temp).unsqueeze(0)  # [1, Hf*Wf]
    target = torch.tensor([gt_index], device=sim_map.device)
    return F.cross_entropy(logits, target)


# =============================================================================
# training / eval
def build_train_subset(all_train_files, max_n: int, seed: int):
    rng = random.Random(seed)
    files = list(all_train_files)
    rng.shuffle(files)
    return files[:max_n]

def train_on_pairs(model, processor, pair_files, stage_name: str):
    """Train EPOCHS_PER_STAGE epochs on given pair_files."""
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    for ep in range(1, EPOCHS_PER_STAGE + 1):
        running = 0.0
        used = 0

        for pair_path in tqdm(pair_files, desc=f"Train {stage_name} | epoch {ep}/{EPOCHS_PER_STAGE}"):
            pair = load_json(pair_path)
            cat = pair["category"]

            src_img = load_image(cat, pair["src_imname"])
            trg_img = load_image(cat, pair["trg_imname"])

            src_kps_r = scale_kps(pair["src_kps"], pair["src_imsize"], (RESIZE_TO, RESIZE_TO))
            trg_kps_r = scale_kps(pair["trg_kps"], pair["trg_imsize"], (RESIZE_TO, RESIZE_TO))

            Fs = extract_dinov2_dense(model, processor, src_img, RESIZE_TO, DEVICE, no_grad=False)
            Ft = extract_dinov2_dense(model, processor, trg_img, RESIZE_TO, DEVICE, no_grad=False)
            Hf, Wf, _ = Ft.shape

            loss_sum = 0.0
            kp_count = 0

            for (x_s, y_s), (x_t_gt, y_t_gt) in zip(src_kps_r, trg_kps_r):
                if MAX_KP_PER_PAIR is not None and kp_count >= MAX_KP_PER_PAIR:
                    break
                if x_s < 0 or y_s < 0 or x_t_gt < 0 or y_t_gt < 0:
                    continue

                px_s, py_s = pixel_to_patch(x_s, y_s, PATCH_SIZE)
                px_s, py_s = clamp_patch(px_s, py_s, Wf, Hf)
                fs = Fs[py_s, px_s]

                sim_map = cosine_sim_map(fs, Ft)
                loss_kp = kp_ce_loss(sim_map, x_t_gt, y_t_gt, Wf, Hf, temp=TEMP)

                loss_sum = loss_sum + loss_kp
                kp_count += 1

            if kp_count == 0:
                continue

            loss = loss_sum / kp_count
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu())
            used += 1

        avg_loss = running / max(1, used)
        print(f"\nEpoch loss ({stage_name}) = {avg_loss:.4f}")

@torch.no_grad()
def evaluate_global(model, processor, split: str, max_pairs=None):
    model.eval()
    pair_files = list_pair_jsons(split)
    if max_pairs is not None:
        pair_files = pair_files[:max_pairs]

    bucket = new_bucket()

    for pair_path in tqdm(pair_files, desc=f"Eval {split}"):
        pair = load_json(pair_path)
        cat = pair["category"]

        src_img = load_image(cat, pair["src_imname"])
        trg_img = load_image(cat, pair["trg_imname"])

        src_kps_r = scale_kps(pair["src_kps"], pair["src_imsize"], (RESIZE_TO, RESIZE_TO))
        trg_kps_r = scale_kps(pair["trg_kps"], pair["trg_imsize"], (RESIZE_TO, RESIZE_TO))

        trg_bbox_r = scale_bbox(pair["trg_bndbox"], pair["trg_imsize"], (RESIZE_TO, RESIZE_TO))
        norm = bbox_norm_from_trg_bbox(trg_bbox_r)

        Fs = extract_dinov2_dense(model, processor, src_img, RESIZE_TO, DEVICE, no_grad=True)
        Ft = extract_dinov2_dense(model, processor, trg_img, RESIZE_TO, DEVICE, no_grad=True)
        Hf, Wf, _ = Ft.shape

        pair_correct = {T: 0 for T in THRESHOLDS}
        pair_total = 0

        for (x_s, y_s), (x_t_gt, y_t_gt) in zip(src_kps_r, trg_kps_r):
            if x_s < 0 or y_s < 0 or x_t_gt < 0 or y_t_gt < 0:
                continue

            px_s, py_s = pixel_to_patch(x_s, y_s, PATCH_SIZE)
            px_s, py_s = clamp_patch(px_s, py_s, Wf, Hf)
            fs = Fs[py_s, px_s]

            sim_map = cosine_sim_map(fs, Ft)
            best = int(torch.argmax(sim_map.reshape(-1)))
            py_p, px_p = best // Wf, best % Wf
            x_t_pred, y_t_pred = patch_to_pixel_center(px_p, py_p, PATCH_SIZE)

            d = ((x_t_pred - x_t_gt) ** 2 + (y_t_pred - y_t_gt) ** 2) ** 0.5
            for T in THRESHOLDS:
                if (d / norm) < T:
                    pair_correct[T] += 1
            pair_total += 1

        if pair_total > 0:
            add_pair_result(bucket, pair_correct, pair_total)

    return bucket


# =============================================================================
def run_progressive_for_unfreeze(unfreeze_last: int):
    print("\n" + "=" * 92)
    print(f"SETTING: DINOv2 | unfreeze_last={unfreeze_last} | stages={TRAIN_TARGETS} | epochs_per_stage={EPOCHS_PER_STAGE}")
    print("=" * 92)

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).float()

    freeze_all(model)
    unfreeze_last_blocks_auto(model, unfreeze_last, verbose=True)
    print_trainable(model)

    # Build a single deterministic subset ordering up to max target (so stages are cumulative)
    all_train = list_pair_jsons(TRAIN_SPLIT)
    max_target = max(TRAIN_TARGETS)
    ordered_subset = build_train_subset(all_train, max_target, seed=SEED)

    # 0) baseline eval (only ONCE)
    base_bucket = evaluate_global(model, processor, EVAL_SPLIT, max_pairs=MAX_EVAL_PAIRS)
    print_global(f"GLOBAL BASELINE | {EVAL_SPLIT}", base_bucket)

    # progressive stages
    # prev = 0
    for target in TRAIN_TARGETS:
        stage_pairs = ordered_subset[:target]  # incremental chunk
        #prev = target

        # Train on incremental chunk
        train_on_pairs(model, processor, stage_pairs, stage_name=f"{target}")

        # Eval after reaching this target
        after_bucket = evaluate_global(model, processor, EVAL_SPLIT, max_pairs=MAX_EVAL_PAIRS)
        print_global(f"GLOBAL AFTER train_pairs={target} | {EVAL_SPLIT}", after_bucket)
    

    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    print(f"Device: {DEVICE}")
    print(f"ROOT: {ROOT.resolve()}")
    print(f"MODEL: {MODEL_NAME}")
    print(f"RESIZE_TO={RESIZE_TO}, PATCH_SIZE={PATCH_SIZE}, MAX_EVAL_PAIRS={MAX_EVAL_PAIRS}")
    
    for uf in UNFREEZE_LIST:
        run_progressive_for_unfreeze(uf)


if __name__ == "__main__":
    main()