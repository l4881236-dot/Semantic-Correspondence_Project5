import os
import json
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


# Step4: SD Backbone + Evaluation
#----------------------------------------------------------
# Basic configurations
ROOT = Path("SPair-71k")          
SPLIT = "test"                   
RESIZE_TO = 512                  
PATCH_SIZE = 8                   
THRESHOLDS = [0.05, 0.1, 0.2]    

# SD backbone
SD_MODE = "unet_mid"           
UNET_T_INDICES = [200, 400, 600] # multi-timestep mean 

MAX_PAIRS = None               # set int for quick debug 
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32


#----------------------------------------------------------
# ===============Define utility functions=====================
def read_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def list_pair_jsons(split: str):
    
    candidates = [
        ROOT / "PairAnnotation" / split,
        ROOT / "PairAnnotation" / f"{split}",
        ROOT / "pair_annotation" / split,
        ROOT / "pair_annotations" / split,
    ]
    for d in candidates:
        if d.exists():
            return sorted(d.glob("*.json"))
    # fallback: search
    found = list(ROOT.rglob(f"{split}/*.json"))
    if not found:
        raise FileNotFoundError(f"Cannot find pair annotation folder for split='{split}' under {ROOT}")
    return sorted(found)

def resolve_image_path(root: Path, imname: str, pair_json_path: Path | None = None):
    
    p1 = root / "JPEGImages" / imname
    if p1.exists():
        return p1

    cat = None
    if pair_json_path is not None:
        base = pair_json_path.name
        if ":" in base:
            cat = base.split(":")[-1].replace(".json", "")
    if cat is not None:
        p2 = root / "JPEGImages" / cat / imname
        if p2.exists():
            return p2

    jpg_root = root / "JPEGImages"
    target_name = Path(imname).name
    for dirpath, _, filenames in os.walk(jpg_root):
        if target_name in filenames:
            return Path(dirpath) / target_name

    raise FileNotFoundError(f"Cannot resolve image path: {imname}")

def load_pair(pair_json: Path):
    pair = read_json(pair_json)

    # Typical keys in SPair-71k pair JSON
    src_im = pair.get("src_imname") or pair.get("src_img") or pair.get("src")
    trg_im = pair.get("trg_imname") or pair.get("trg_img") or pair.get("trg")
    
    # [FIX] Extract category
    cat = pair.get("category", "unknown")

    if src_im is None or trg_im is None:
        raise KeyError(f"Missing src/trg image name in {pair_json}")

    src_path = resolve_image_path(ROOT, src_im, pair_json)
    trg_path = resolve_image_path(ROOT, trg_im, pair_json)

    # Keypoints: often stored as 2xN lists
    src_kps = pair.get("src_kps")
    trg_kps = pair.get("trg_kps")
    if src_kps is None or trg_kps is None:
        raise KeyError(f"Missing src_kps/trg_kps in {pair_json}")

    src_kps = np.array(src_kps, dtype=np.float32)  # (2,N) or (N,2)
    trg_kps = np.array(trg_kps, dtype=np.float32)

    if src_kps.shape[0] == 2 and src_kps.ndim == 2:
        src_kps = src_kps.T  # (N,2)
    if trg_kps.shape[0] == 2 and trg_kps.ndim == 2:
        trg_kps = trg_kps.T  # (N,2)

    # visibility: some releases use 1xN or N list
    # if coords are -1, treat as invalid
    valid = (src_kps[:, 0] >= 0) & (src_kps[:, 1] >= 0) & (trg_kps[:, 0] >= 0) & (trg_kps[:, 1] >= 0)

    # PCK threshold in pixels: prefer pckthres if provided
    pckthres = pair.get("pckthres", None)
    if pckthres is None:
        # compute from target bbox if available
        bbox = pair.get("trg_bbox") or pair.get("trg_bndbox") or pair.get("trg_box")
        if bbox is not None and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            pckthres = max(float(x2 - x1), float(y2 - y1))
        else:
            # worst-case fallback: use resize size
            pckthres = float(RESIZE_TO)

    # difficulty attributes (often in SPair)
    dscore = int(pair.get("viewpoint_variation", 0)) + int(pair.get("scale_variation", 0)) \
             + int(pair.get("truncation", 0)) + int(pair.get("occlusion", 0))
    # clamp to 0..7
    dscore = max(0, min(7, dscore))

    # [FIX] Return cat at the end
    return src_path, trg_path, src_kps, trg_kps, valid, float(pckthres), dscore, cat


# ------------------------------------------
# SD Backbone (multi-timestep)
class SDBackbone:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = DEVICE,
        dtype: torch.dtype = DTYPE,
        mode: str = SD_MODE,
        input_size: int = RESIZE_TO,
        unet_t_indices=None,
    ):
        assert mode in ["vae", "unet_mid"], "mode must be 'vae' or 'unet_mid'"
        assert input_size % 8 == 0, "input_size must be multiple of 8"

        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.input_size = input_size

        if unet_t_indices is None:
            unet_t_indices = UNET_T_INDICES
        self.unet_t_indices = list(unet_t_indices)

        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to(device)
        pipe.safety_checker = None
        pipe.enable_attention_slicing()

        self.pipe = pipe
        self.vae = pipe.vae.eval()
        self.unet = pipe.unet.eval()
        self.scheduler = pipe.scheduler

        # text encoder (for unconditional embedding); required for UNet cross-attn
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        # cache unconditional (empty prompt) embedding once
        with torch.no_grad():
            tok = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tok.input_ids.to(self.device)
            self._empty_text_emb = self.text_encoder(input_ids)[0].to(self.dtype)

        for m in [self.vae, self.unet]:
            for p in m.parameters():
                p.requires_grad_(False)

        # cache timesteps once
        self.scheduler.set_timesteps(1000, device=self.device)
        self._timesteps = self.scheduler.timesteps

    @torch.no_grad()
    def _preprocess(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((self.input_size, self.input_size), Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        x = x * 2.0 - 1.0
        return x.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def _vae_encode(self, x: torch.Tensor) -> torch.Tensor:
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def extract(self, img: Image.Image) -> torch.Tensor:
        """
        return (C, Hf, Wf) normalized
        """
        x = self._preprocess(img)
        latents = self._vae_encode(x)  # (1,4,64,64)

        if self.mode == "vae":
            fmap = latents[0].float()
            return F.normalize(fmap, dim=0)

        # unet_mid multi-t
        feats_sum = None
        for t_idx in self.unet_t_indices:
            t = self._timesteps[int(t_idx)].to(self.device)
            noise = torch.randn_like(latents)
            noisy = self.scheduler.add_noise(latents, noise, t)

            feats = {}
            def hook_mid(_m, _inp, out):
                feats["mid"] = out

            h = self.unet.mid_block.register_forward_hook(hook_mid)
            _ = self.unet(noisy, t, encoder_hidden_states=self._empty_text_emb)
            h.remove()

            fmap = feats["mid"][0].float()  # (C,Hf,Wf)
            fmap = F.normalize(fmap, dim=0)
            feats_sum = fmap if feats_sum is None else (feats_sum + fmap)

        fmap = feats_sum / float(len(self.unet_t_indices))
        return F.normalize(fmap, dim=0)


# ---------------------------------
# Matching

def pixel_to_feat_xy(x_pix: float, y_pix: float, resize_to: int, Hf: int, Wf: int):
    """Map pixel coords (in the resized image space) to feature-map coords.

    We use align_corners=True in grid_sample, so we should map pixel centers consistently:
      x_pix in [0, resize_to-1]  ->  x_f in [0, Wf-1]
      y_pix in [0, resize_to-1]  ->  y_f in [0, Hf-1]
    """
    if resize_to <= 1:
        return 0.0, 0.0
    x = (x_pix / (resize_to - 1.0)) * (Wf - 1.0)
    y = (y_pix / (resize_to - 1.0)) * (Hf - 1.0)
    # clamp
    x = max(0.0, min(Wf - 1.0, x))
    y = max(0.0, min(Hf - 1.0, y))
    return x, y

def feat_to_pixel_xy(x_f: float, y_f: float, resize_to: int, Hf: int, Wf: int):
    """Inverse mapping of pixel_to_feat_xy."""
    if Wf <= 1 or Hf <= 1:
        return 0.0, 0.0
    x = (x_f / (Wf - 1.0)) * (resize_to - 1.0)
    y = (y_f / (Hf - 1.0)) * (resize_to - 1.0)
    return x, y

def sample_feat_bilinear(Fc: torch.Tensor, x: float, y: float) -> torch.Tensor:
    
    C, H, W = Fc.shape
    gx = (x / (W - 1)) * 2 - 1
    gy = (y / (H - 1)) * 2 - 1
    grid = torch.tensor([[[[gx, gy]]]], device=Fc.device, dtype=Fc.dtype)
    v = F.grid_sample(Fc.unsqueeze(0), grid, mode="bilinear", align_corners=True)
    return v[0, :, 0, 0]

def cosine_sim_map(f_src: torch.Tensor, F_tgt: torch.Tensor) -> torch.Tensor:
    # both normalized => dot
    return (F_tgt * f_src[:, None, None]).sum(dim=0)

def argmax_2d(sim: torch.Tensor):
    H, W = sim.shape
    idx = int(torch.argmax(sim).item())
    y = idx // W
    x = idx % W
    return float(x), float(y)


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

@torch.no_grad()
def evaluate_full():
    pair_files = list_pair_jsons(SPLIT)
    if MAX_PAIRS is not None:
        pair_files = pair_files[:MAX_PAIRS]

    print(f"Split={SPLIT}  Num_json={len(pair_files)}")
    print(f"Device={DEVICE}  RESIZE_TO={RESIZE_TO}  PATCH_SIZE={PATCH_SIZE}")
    print(f"SD_MODE={SD_MODE}  UNET_T_INDICES={UNET_T_INDICES}")

    backbone = SDBackbone()

    global_bucket = new_bucket()
    cat_buckets = defaultdict(new_bucket)

    for pj in tqdm(pair_files, desc=f"Eval {SPLIT}"):
        try:
            # [FIX] now receives 'cat'
            src_path, trg_path, src_kps, trg_kps, valid, pckthres, dscore, cat = load_pair(pj)
        except Exception as e:
            # skip broken file
            # print(f"Error loading {pj}: {e}")
            continue

        img_s = Image.open(src_path).convert("RGB")
        img_t = Image.open(trg_path).convert("RGB")

        # IMPORTANT: SPair-71k keypoints/bboxes are in the ORIGINAL image coordinate system.
        # Our backbone always resizes inputs to RESIZE_TO x RESIZE_TO internally.
        # Therefore, we must rescale both keypoints and the PCK threshold to the resized space
        # before computing distances.
        Ws0, Hs0 = img_s.size
        Wt0, Ht0 = img_t.size
        sx_s = RESIZE_TO / float(Ws0) if Ws0 > 0 else 1.0
        sy_s = RESIZE_TO / float(Hs0) if Hs0 > 0 else 1.0
        sx_t = RESIZE_TO / float(Wt0) if Wt0 > 0 else 1.0
        sy_t = RESIZE_TO / float(Ht0) if Ht0 > 0 else 1.0

        # rescale PCK threshold (defined in target-image pixels) to resized target scale
        pckthres_r = float(pckthres) * max(sx_t, sy_t)

        F_s = backbone.extract(img_s)  # (C,Hf,Wf)
        F_t = backbone.extract(img_t)

        C, Hf, Wf = F_s.shape

        pair_total = int(valid.sum())
        if pair_total == 0:
            continue

        pair_correct = {T: 0 for T in THRESHOLDS}

        # for each keypoint
        for (xs, ys, xt, yt, ok) in zip(src_kps[:,0], src_kps[:,1], trg_kps[:,0], trg_kps[:,1], valid):
            if not ok:
                continue

            # rescale keypoints to resized-image coordinates
            xs_r = float(xs) * sx_s
            ys_r = float(ys) * sy_s
            xt_r = float(xt) * sx_t
            yt_r = float(yt) * sy_t

            # map source kp to feature coords
            x_f, y_f = pixel_to_feat_xy(xs_r, ys_r, RESIZE_TO, Hf, Wf)
            f = sample_feat_bilinear(F_s, x_f, y_f)
            f = F.normalize(f, dim=0)

            sim = cosine_sim_map(f, F_t)  # (Hf,Wf)
            xb, yb = argmax_2d(sim)

            x_pred, y_pred = feat_to_pixel_xy(xb, yb, RESIZE_TO, Hf, Wf)

            # measure distance in pixel space (on resized images)
            dx = x_pred - xt_r
            dy = y_pred - yt_r
            dist = math.sqrt(dx*dx + dy*dy)

            for T in THRESHOLDS:
                if dist <= T * pckthres_r:
                    pair_correct[T] += 1

        # add to stats
        if pair_total > 0:
            add_pair_result(global_bucket, pair_correct, pair_total)
            add_pair_result(cat_buckets[cat], pair_correct, pair_total)
            

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


if __name__ == "__main__":
    evaluate_full()