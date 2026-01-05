#!/usr/bin/env python3
"""
Snapshot-style RGB-D training script (single file, no external imports).

- Config block at top (paths/hparams).
- COCO loader that keeps diameter + per-annotation depth_file, and remaps category_id contiguously.
- RGB-D mapper: no geometric augs, photometric only on RGB; depth in mm as 4th channel.
- ResNet50-FPN backbone patched for 4-channel input (early fusion, robust to re-runs).
- Inline DiameterROIHeads (@configurable): regress normalized diameter from RoI features + geometric cues
  [log_w, log_h, log_z, geom_mm_norm].
- Inline PeriodicValHook: MAE/RMSE on the validation set using GT boxes.

Assumptions:
- COCO annotations include: bbox, category_id, diameter_gt (or diameter_mm), depth_file
- Depth images are Z16 PNGs in millimetres.
"""

import os, json, random, math
from typing import Any, Dict, List
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F

from detectron2 import model_zoo
from detectron2.config import get_cfg, configurable
from detectron2.engine import DefaultTrainer, HookBase, default_setup
from detectron2.data import (
    DatasetCatalog, MetadataCatalog,
    build_detection_train_loader, build_detection_test_loader,
    detection_utils as utils, transforms as T,
)
from detectron2.structures import Instances, Boxes, BoxMode
from detectron2.utils.events import CommonMetricPrinter, JSONWriter

from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import BasicStem
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROIHeads, ROI_HEADS_REGISTRY
from detectron2.layers import Conv2d
from detectron2.checkpoint import DetectionCheckpointer


# =========================================================
# ===================== CONFIG BLOCK ======================
# =========================================================
TRAIN_JSON   = "/content/drive/MyDrive/snapshots_dataset/annotations/instances_Train.json"
EVAL_JSON    = "/content/drive/MyDrive/snapshots_dataset/annotations/instances_Eval.json"
IMG_ROOT     = "/content/drive/MyDrive/snapshots_dataset/rgb"
DEPTH_ROOT   = "/content/drive/MyDrive/snapshots_dataset/depth"

OUTPUT_DIR   = "/content/drive/MyDrive/snapshots_dataset/output_rgbd"
MAX_ITER     = 32000
BATCH        = 2
BASE_LR      = 2e-4
WARMUP_FRAC  = 0.02
EVAL_PERIOD  = 1000
SEED         = 42

# Camera / geometry
FX_PX        = 600.0
SCALE_MM     = 70.0

# Augmentations: snapshot style (no geometry, solo photometric su RGB)
ENABLE_PHOTOMETRIC = True

# 4th channel normalization (depth in mm) — dai tuoi valori stimati
PIXEL_MEAN = [103.53, 116.28, 123.675, 192.723]
PIXEL_STD  = [57.375, 57.12,  58.395,  80.738]

# ROI head settings
POOLER_RESOLUTION = 14
HIDDEN_DIM = 1024
HIDDEN_DIM_2 = 256
HUBER_DELTA = 1.0           # Huber loss delta su unità normalizzate
RESIDUAL_L2 = 0.0           # >0 per spingere residual piccolo rispetto al prior geometrico
DROPOUT_P   = 0.0
# =========================================================


# ===== category mapping (global) =====
_ID_TO_CONTIG = None        # es. {1:0, 3:1, ...}
_THING_CLASSES = None       # es. ["apple"] in ordine contiguo

def _build_category_mapping(json_path: str):
    """Legge categories dal COCO e costruisce:
       - _ID_TO_CONTIG: mappa id COCO -> indice contiguo [0..K-1]
       - _THING_CLASSES: nomi in ordine contiguo
    """
    global _ID_TO_CONTIG, _THING_CLASSES
    with open(json_path, "r") as f:
        coco = json.load(f)

    cats = coco.get("categories", [])
    if not cats:
        # fallback: singola classe 'apple'
        _ID_TO_CONTIG = {0: 0, 1: 0}
        _THING_CLASSES = ["apple"]
        return

    cats_sorted = sorted(cats, key=lambda c: c.get("id", 0))
    _ID_TO_CONTIG = {c["id"]: i for i, c in enumerate(cats_sorted)}
    _THING_CLASSES = [c.get("name", str(c["id"])) for c in cats_sorted]


# ---------------------- dataset loader ----------------------
def load_coco_with_diameter(json_path: str, img_root: str) -> List[Dict[str, Any]]:
    with open(json_path, "r") as f:
        coco = json.load(f)

    imgs = {im["id"]: im for im in coco["images"]}
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if "bbox" not in ann or len(ann["bbox"]) != 4:
            continue

        if "diameter_mm" in ann:
            dmm = float(ann["diameter_mm"])
        elif "diameter_gt" in ann:
            dmm = float(ann["diameter_gt"])
        else:
            continue

        # ---- remap category_id to contiguous ----
        raw_cid = ann.get("category_id", 0)
        cid = _ID_TO_CONTIG.get(raw_cid, 0) if _ID_TO_CONTIG is not None else 0

        depth_file = ann.get("depth_file", None)
        anns_by_img.setdefault(ann["image_id"], []).append({
            "bbox": ann["bbox"],
            "category_id": cid,
            "diameter_mm": dmm,
            "depth_file": depth_file,
        })

    dataset = []
    for img_id, im in imgs.items():
        file_name = os.path.join(img_root, im["file_name"])
        rec = {
            "file_name": file_name,
            "image_id": img_id,
            "height": im["height"],
            "width":  im["width"],
            "annotations": anns_by_img.get(img_id, []),
        }
        dfs = [a["depth_file"] for a in rec["annotations"] if a.get("depth_file")]
        if dfs:
            rec["depth_file"] = dfs[0]
        dataset.append(rec)
    return dataset


# ------------------------ helpers ------------------------
def _to_single_channel(arr: np.ndarray) -> np.ndarray:
    """Force depth to 2D: squeeze 1-channel or take channel 0 if 3-channel/colorized."""
    if arr is None:
        return None
    a = np.asarray(arr)
    # If colorized (HxWx3 or more), take first channel
    if a.ndim == 3 and a.shape[2] >= 3:
        a = a[..., 0]
    # Squeeze leftover singleton dims (HxWx1 -> HxW, HxWx1x1 -> HxW, etc.)
    if a.ndim > 2:
        a = np.squeeze(a)
    return a

def _read_depth_mm(path: str) -> np.ndarray:
    """Read depth PNG; return float32 millimetres; zeros→NaN. Robust to 3-ch colorized files."""
    if not path:
        return None
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = _to_single_channel(d)
    dm = d.astype(np.float32)
    dm[dm == 0] = np.nan
    return dm

def _read_depth_m(path: str) -> np.ndarray:
    """Meters (for geometric cues). Robust to 3-ch colorized files."""
    if not path:
        return None
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = _to_single_channel(d)
    dm = d.astype(np.float32)
    if dm.max() > 50.0:
        dm = dm / 1000.0
    return dm

def _resize_depth_like(depth_arr: np.ndarray, H: int, W: int) -> np.ndarray:
    if depth_arr is None:
        return None
    return cv2.resize(depth_arr, (W, H), interpolation=cv2.INTER_NEAREST)

def _depth_stat_in_box(depth_m: np.ndarray, x: float, y: float, w: float, h: float,
                       H: int, W: int) -> float:
    x0 = max(int(x), 0); y0 = max(int(y), 0)
    x1 = min(int(x + w), W); y1 = min(int(y + h), H)
    if x1 <= x0 or y1 <= y0 or depth_m is None:
        return 0.0
    crop = depth_m[y0:y1, x0:x1]
    v = crop[np.isfinite(crop) & (crop > 0)]
    if v.size == 0:
        return 0.0
    return float(np.median(v))

def _pinhole_mm(px_w: float, z_m: float, fx_px: float) -> float:
    if z_m <= 0.0 or fx_px <= 0.0: return 0.0
    return float(px_w * (z_m / fx_px) * 1000.0)

def _numpy_to_torch_state_dict(sd):
    """
    Detectron2 model_zoo checkpoints (.pkl) often store params as NumPy arrays.
    Convert all leaves to torch.Tensor so load_state_dict() accepts them.
    Accepts either a flat dict or a dict with key 'model'.
    Returns a flat dict of torch.Tensors.
    """
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v
        elif isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)
        else:
            try:
                out[k] = torch.as_tensor(v)
            except Exception:
                continue
    return out


# -------------------- RGB-D mapper (snapshot style) --------------------
def mapper_rgbd(dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
    d = dataset_dict.copy()
    rgb_bgr = utils.read_image(d["file_name"], format="BGR").astype(np.float32)
    H, W = rgb_bgr.shape[:2]

    # Photometric-only on RGB
    if ENABLE_PHOTOMETRIC:
        aug = T.AugmentationList([
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
            T.RandomSaturation(0.9, 1.1),
        ])
        ain = T.AugInput(rgb_bgr)
        _ = aug(ain)
        rgb_bgr = ain.image

    # Depth path (relative to DEPTH_ROOT if not absolute)
    dp = d.get("depth_file")
    depth_path = None
    if dp:
        depth_path = dp if os.path.isabs(dp) else os.path.join(DEPTH_ROOT, dp)

    # ---- Depth channel in mm (robusto a 3-ch, 1x1, ecc.) ----
    depth_mm = _read_depth_mm(depth_path)
    depth_mm = _resize_depth_like(depth_mm, H, W)
    if depth_mm is None:
        depth_mm = np.full((H, W), np.nan, dtype=np.float32)
    else:
        depth_mm = np.asarray(depth_mm)
        # se 3-ch (colorized), usa il primo canale; squeeze il resto
        if depth_mm.ndim == 3 and depth_mm.shape[2] >= 3:
            depth_mm = depth_mm[..., 0]
        if depth_mm.ndim > 2:
            depth_mm = np.squeeze(depth_mm)
        if depth_mm.ndim != 2:
            if depth_mm.size == H * W:
                depth_mm = depth_mm.reshape(H, W)
            else:
                depth_mm = cv2.resize(depth_mm.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        depth_mm = depth_mm.astype(np.float32)

    # riempi NaN con mediana (o 0 se tutto NaN)
    if np.isnan(depth_mm).any():
        med = np.nanmedian(depth_mm) if np.isfinite(depth_mm).any() else 0.0
        depth_mm = np.where(np.isnan(depth_mm), med, depth_mm)

    # ---- Build [4,H,W] senza concatenate (evita mismatch dimensioni) ----
    rgb_rgb = cv2.cvtColor(rgb_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)  # [H,W,3]
    rgbd = np.empty((H, W, 4), dtype=np.float32)
    rgbd[..., :3] = rgb_rgb
    rgbd[..., 3]  = depth_mm
    image = torch.as_tensor(rgbd.transpose(2, 0, 1))  # [4,H,W]

    # Normalize
    mean = torch.tensor(np.array(PIXEL_MEAN, dtype=np.float32).reshape(4,1,1))
    std  = torch.tensor(np.array(PIXEL_STD,  dtype=np.float32).reshape(4,1,1))
    image = (image - mean) / std

    # ---- Instances + geometric cues (depth in meters con stesso forcing) ----
    annos = []
    xyxy  = []
    cues  = []  # [log_w, log_h, log_z, geom_mm_norm]
    depth_m = _read_depth_m(depth_path)
    depth_m = _resize_depth_like(depth_m, H, W)
    if depth_m is not None:
        depth_m = np.asarray(depth_m)
        if depth_m.ndim == 3 and depth_m.shape[2] >= 3:
            depth_m = depth_m[..., 0]
        if depth_m.ndim > 2:
            depth_m = np.squeeze(depth_m)
        if depth_m.ndim != 2:
            if depth_m.size == H * W:
                depth_m = depth_m.reshape(H, W)
            else:
                depth_m = cv2.resize(depth_m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        depth_m = depth_m.astype(np.float32)

    for a in d.get("annotations", []):
        x, y, w, h = a["bbox"]
        xyxy.append([x, y, x+w, y+h])

        gt_d_norm = float(a["diameter_mm"]) / SCALE_MM
        z_m = _depth_stat_in_box(depth_m, x, y, w, h, H, W) if depth_m is not None else 0.0
        log_z = float(np.log(max(z_m, 1e-6)))
        geom_mm = _pinhole_mm(w, z_m, FX_PX)
        geom_norm = float(geom_mm / SCALE_MM)

        cues.append([np.log(max(w,1e-6)), np.log(max(h,1e-6)), log_z, geom_norm])
        annos.append({
            "bbox": [x, y, w, h],
            "bbox_mode": BoxMode.XYWH_ABS,
            "category_id": a.get("category_id", 0),
            "diameter_norm": gt_d_norm
        })

    instances = utils.annotations_to_instances(annos, (H, W))
    if xyxy:
        instances.gt_boxes = Boxes(torch.tensor(xyxy, dtype=torch.float32))
        instances.set("diameter_norm", torch.tensor([a["diameter_norm"] for a in annos], dtype=torch.float32))
        instances.set("geom_cues",     torch.tensor(cues, dtype=torch.float32))

    d["image"] = image
    d["instances"] = instances
    return d


# ---------------------- 4-ch ResNet-FPN backbone ----------------------
class BasicStem4(BasicStem):
    def __init__(self, in_channels=4, out_channels=64, norm="FrozenBN"):
        super().__init__(in_channels=in_channels, out_channels=out_channels, norm=norm)

@BACKBONE_REGISTRY.register()
def build_resnet_fpn_backbone_rgbd_v1(cfg, input_shape):
    """
    Build a ResNet-FPN backbone then patch bottom_up.stem.conv1 to 4 channels, robustly.
    """
    backbone = build_resnet_fpn_backbone(cfg, input_shape)  # FPN
    old_conv: nn.Conv2d = backbone.bottom_up.stem.conv1

    out_ch  = old_conv.out_channels
    k, s, p, d = old_conv.kernel_size, old_conv.stride, old_conv.padding, old_conv.dilation
    bias = old_conv.bias is not None

    new_conv = Conv2d(
        in_channels=4,
        out_channels=out_ch,
        kernel_size=k,
        stride=s,
        padding=p,
        dilation=d,
        bias=bias,
        norm=None,
        activation=None,
    )

    with torch.no_grad():
        w_old = old_conv.weight  # [out_ch, Cin_old, k, k]
        Cin_old = w_old.shape[1]
        if Cin_old == 3:
            new_conv.weight[:, :3, :, :] = w_old
            new_conv.weight[:, 3:4, :, :] = w_old.mean(dim=1, keepdim=True)
        elif Cin_old == 4:
            new_conv.weight.copy_(w_old)
        else:
            c = min(Cin_old, 4)
            new_conv.weight[:, :c, :, :] = w_old[:, :c, :, :]
            fill = w_old.mean(dim=1, keepdim=True)
            if c < 4:
                new_conv.weight[:, c:4, :, :] = fill
        if bias and old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    backbone.bottom_up.stem.conv1 = new_conv

    print(f"[DEBUG] bottom_up.stem.conv1 weight shape -> {backbone.bottom_up.stem.conv1.weight.shape}")
    print(f"[DEBUG] FPN output features -> {list(backbone.output_shape().keys())}")
    return backbone


# ---------------------- ROI HEAD (INLINE, @configurable) ----------------------
@ROI_HEADS_REGISTRY.register()
class DiameterROIHeads(ROIHeads):
    """
    Minimal ROI heads that:
     - use GT boxes during training (no RPN interaction),
     - pool FPN features over GT boxes,
     - fuse pooled features + geometric cues and predict normalized diameter,
     - compute Smooth L1 (Huber) on normalized units; optional L2 on residual vs geom prior.
    """

    @configurable
    def __init__(self, *, cfg, input_shape, **kwargs):
        # Base ROIHeads expects kwargs from ROIHeads.from_config(cfg)
        super().__init__(**kwargs)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.pooler = ROIPooler(
            output_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            scales=tuple(1.0 / input_shape[f].stride for f in self.in_features),
            sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO,
            pooler_type=cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE,
        )
        in_channels = [input_shape[f].channels for f in self.in_features][0]
        rep = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        flat_dim = in_channels * rep * rep

        self.mlp = nn.Sequential(
            nn.Linear(flat_dim + 4, HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT_P) if DROPOUT_P > 0 else nn.Identity(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM_2),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN_DIM_2, 1),
        )

        self.huber_delta = HUBER_DELTA
        self.residual_l2 = RESIDUAL_L2

    @classmethod
    def from_config(cls, cfg, input_shape=None):
        base_kwargs = ROIHeads.from_config(cfg)          # version that takes only cfg
        base_kwargs.update({"cfg": cfg, "input_shape": input_shape})
        return base_kwargs

    def _pool(self, features, boxes):
        feats = [features[f] for f in self.in_features]
        return self.pooler(feats, boxes)

    def forward(self, images, features, proposals, targets=None):
        train = self.training
        insts = targets if train and targets is not None else proposals

        boxes = [i.gt_boxes if i.has("gt_boxes") else i.proposal_boxes for i in insts]
        pooled = self._pool(features, boxes)
        N = pooled.shape[0]
        if N == 0:
            return proposals, {} if train else (proposals, [])

        x = pooled.flatten(start_dim=1)
        if not insts[0].has("geom_cues"):
            raise ValueError("Instances missing 'geom_cues' field.")
        cues = torch.cat([i.get("geom_cues") for i in insts], dim=0)
        x = torch.cat([x, cues.to(x.device)], dim=1)

        pred = self.mlp(x).squeeze(1)

        if train:
            if not insts[0].has("diameter_norm"):
                raise ValueError("Instances missing 'diameter_norm' field.")
            target = torch.cat([i.get("diameter_norm") for i in insts], dim=0).to(pred.device)

            loss = F.smooth_l1_loss(pred, target, beta=self.huber_delta, reduction="mean")

            if self.residual_l2 > 0:
                geom = torch.cat([i.get("geom_cues") for i in insts], dim=0)[:, 3].to(pred.device)
                residual = pred - geom
                loss = loss + self.residual_l2 * torch.mean(residual * residual)

            return proposals, {"loss_diameter": loss}

        out = []
        idx = 0
        for inst in proposals:
            m = len(inst)
            if m == 0:
                out.append(inst); continue
            new_inst = Instances(inst.image_size, **inst.get_fields())
            d_norm = pred[idx:idx+m]
            d_mm = d_norm * SCALE_MM
            new_inst.set("pred_diameter_mm", d_mm.detach().cpu())
            out.append(new_inst)
            idx += m

        return out, {}


# ------------------------- trainer -------------------------
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper_rgbd)

    def build_writers(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        return [
            CommonMetricPrinter(max_iter=self.cfg.SOLVER.MAX_ITER),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]


# --------------------- validation hook ---------------------
class PeriodicValHook(HookBase):
    def __init__(self, eval_period, json_path, img_root):
        self._period = eval_period
        self._json   = json_path
        self._img_root = img_root
        self._name  = "apples_eval"

        if self._name not in DatasetCatalog.list():
            DatasetCatalog.register(self._name, lambda: load_coco_with_diameter(self._json, self._img_root))
            MetadataCatalog.get(self._name).set(thing_classes=_THING_CLASSES or ["apple"])

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if (self._period > 0) and (next_iter % self._period == 0 or next_iter == self.trainer.max_iter):
            self._run_eval()

    @torch.no_grad()
    def _run_eval(self):
        model = self.trainer.model
        model.eval()
        cfg = self.trainer.cfg
        val_loader = build_detection_test_loader(cfg, self._name, mapper=mapper_rgbd)

        abs_errs, sq_errs, n = 0.0, 0.0, 0
        for batch in val_loader:
            inputs = [{"image": b["image"], "instances": b["instances"]} for b in batch]
            images = model.preprocess_image(inputs)           # ImageList
            features = model.backbone(images.tensor)          # dict of FPN features

            targets = [x["instances"].to(images.tensor.device) for x in inputs]
            pred_insts, _ = model.roi_heads(images, features, proposals=targets, targets=targets)
            for tgt, pred in zip(targets, pred_insts):
                gt = tgt.get("diameter_norm") * SCALE_MM  # mm
                pr = pred.get("pred_diameter_mm")
                if gt is None or pr is None:
                    continue
                gt = gt.detach().cpu().numpy().astype(np.float32)
                pr = pr.detach().cpu().numpy().astype(np.float32)
                abs_errs += float(np.abs(pr - gt).sum())
                sq_errs  += float(((pr - gt) ** 2).sum())
                n        += gt.shape[0]

        if n > 0:
            mae = abs_errs / n
            rmse = math.sqrt(sq_errs / n)
            print(f"[Val] MAE(mm)={mae:.3f}  RMSE(mm)={rmse:.3f}  (N={n})")
            storage = self.trainer.storage
            storage.put_scalar("val_mae_mm", mae)
            storage.put_scalar("val_rmse_mm", rmse)

        model.train()


# ------------------------ config ---------------------------
def build_cfg():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # 4-ch ResNet-FPN backbone
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone_rgbd_v1"

    # Replace ROI heads with our inline head
    cfg.MODEL.ROI_HEADS.NAME = "DiameterROIHeads"
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]  # FPN levels
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = POOLER_RESOLUTION
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES will be set in main() from _THING_CLASSES

    # 4-channel normalization
    cfg.MODEL.PIXEL_MEAN = PIXEL_MEAN
    cfg.MODEL.PIXEL_STD  = PIXEL_STD

    # Datasets
    cfg.DATASETS.TRAIN = ("apples_train",)
    cfg.DATASETS.TEST  = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    # Solver
    cfg.SOLVER.IMS_PER_BATCH = BATCH
    cfg.SOLVER.BASE_LR = BASE_LR
    cfg.SOLVER.MAX_ITER = MAX_ITER
    cfg.SOLVER.WARMUP_ITERS = max(10, int(WARMUP_FRAC * MAX_ITER))
    cfg.SOLVER.STEPS = []  # flat LR

    # Image sizes
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST  = 800
    cfg.INPUT.MAX_SIZE_TEST  = 1333

    # Pretrained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


# -------------------------- main --------------------------
def _expand_conv1_in_state_dict_to_4ch(state_dict):
    """
    Espande i pesi conv1 da 3→4 canali prima del load, se necessario.
    Prova sia il path FPN (bottom_up) che quello semplice.
    """
    keys = [
        "backbone.bottom_up.stem.conv1.weight",  # FPN
        "backbone.stem.conv1.weight",            # non-FPN fallback
    ]
    for key in keys:
        w = state_dict.get(key, None)
        if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[1] == 3:
            w4 = torch.zeros((w.shape[0], 4, w.shape[2], w.shape[3]), dtype=w.dtype)
            w4[:, :3, :, :] = w
            w4[:, 3:4, :, :] = w.mean(dim=1, keepdim=True)
            state_dict[key] = w4
            print(f"[DEBUG] Patched checkpoint conv1 3->4 channels at '{key}'")
    return state_dict

def main():
    # ---- build category map from TRAIN_JSON ----
    _build_category_mapping(TRAIN_JSON)
    assert _ID_TO_CONTIG is not None and _THING_CLASSES is not None

    # Register datasets (uses mapping)
    DatasetCatalog.clear()
    DatasetCatalog.register("apples_train", lambda: load_coco_with_diameter(TRAIN_JSON, IMG_ROOT))
    MetadataCatalog.get("apples_train").set(thing_classes=_THING_CLASSES)

    if EVAL_JSON and os.path.exists(EVAL_JSON):
        if "apples_eval" not in DatasetCatalog.list():
            DatasetCatalog.register("apples_eval", lambda: load_coco_with_diameter(EVAL_JSON, IMG_ROOT))
            MetadataCatalog.get("apples_eval").set(thing_classes=_THING_CLASSES)

    cfg = build_cfg()
    # Align NUM_CLASSES to categories
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(_THING_CLASSES)
    default_setup(cfg, {})

    trainer = MyTrainer(cfg)

    # ---- load pretrained weights with conv1 3->4 patch if needed ----
    trainer.model.train()  # ensure modules are built
    checkpointer = DetectionCheckpointer(trainer.model)

    raw_sd = checkpointer._load_file(cfg.MODEL.WEIGHTS)   # may contain NumPy arrays
    state_dict = _numpy_to_torch_state_dict(raw_sd)       # <- convert all to torch.Tensors
    state_dict = _expand_conv1_in_state_dict_to_4ch(state_dict)  # keep as torch tensors

    missing, unexpected = trainer.model.load_state_dict(state_dict, strict=False)
    print("[DEBUG] load_state_dict missing:", missing)
    print("[DEBUG] load_state_dict unexpected:", unexpected)

    # Debug sulla conv1 del modello finale e sulle feature FPN
    try:
        print(f"[DEBUG] model bottom_up.stem.conv1 weight shape -> {trainer.model.backbone.bottom_up.stem.conv1.weight.shape}")
        print(f"[DEBUG] model FPN features -> {list(trainer.model.backbone.output_shape().keys())}")
    except Exception as e:
        print("[DEBUG] cannot print FPN debug:", e)

    # --- Periodic validation with GT boxes (inserted before writers) ---
    if EVAL_JSON and os.path.exists(EVAL_JSON):
        val_hook = PeriodicValHook(EVAL_PERIOD, EVAL_JSON, IMG_ROOT)
        # Detectron2 mette i writer (CommonMetricPrinter e JSONWriter) alla fine,
        # quindi li spingiamo giù e inseriamo il val_hook prima di loro
        WRITERS_COUNT = 2
        trainer._hooks.insert(-WRITERS_COUNT, val_hook)

    trainer.train()

if __name__ == "__main__":
    main()
