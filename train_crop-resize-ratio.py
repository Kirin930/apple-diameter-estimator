# ═══════════════════════════════════════════════════════════════════════════
# SEGMENTATION + DIAMETER REGRESSION + DEPTH TRAINING SCRIPT
# ───────────────────────────────────────────────────────────────────────────
# RGB+DEPTH CROPPED AND RESIZED 224x224 + Resize Ratio -  DIAMETER REGRESSION
# ═══════════════════════════════════════════════════════════════════════════


import os
import copy
import torch
import torch.nn.functional as F
from torch.nn import SmoothL1Loss
import numpy as np

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data.datasets import load_coco_json
from detectron2.structures import Boxes
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads import StandardROIHeads

# 1. Register datasets (COCO JSON with additional 'diameter' field)
train_json = "/content/drive/MyDrive/RGB-D_diameter_thesis/data/single_largest_apple_crops/coco_single_largest_apples_resize_train.json"
val_json = "/content/drive/MyDrive/RGB-D_diameter_thesis/data/single_largest_apple_crops/coco_single_largest_apples_resize_val.json"
image_train_dir = "/content/drive/MyDrive/RGB-D_diameter_thesis/data/single_largest_apple_crops/rgb/train"
image_val_dir = "/content/drive/MyDrive/RGB-D_diameter_thesis/data/single_largest_apple_crops/rgb/val"
output_dir = "/content/drive/MyDrive/RGB-D_diameter_thesis/output_single_largest_apple_crops_resizeratio"

DatasetCatalog.register("apple_train", lambda: load_coco_json(train_json, image_train_dir, "apple_train", extra_annotation_keys=["diameter"]))
MetadataCatalog.get("apple_train").set(thing_classes=["apple"])

DatasetCatalog.register("apple_val", lambda: load_coco_json(val_json, image_val_dir, "apple_val", extra_annotation_keys=["diameter"]))
MetadataCatalog.get("apple_val").set(thing_classes=["apple"])

# 2. Define Custom ROIHeads with diameter regression
@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads):
    """
    A custom ROI head that:
      - During TRAINING: computes both classification/box/mask losses AND
        a diameter‐regression L1 loss using `gt_diameter` attached to proposals.
      - During INFERENCE: returns a list of Instances (one per image) with
        an extra field `pred_diameter` on each Instances object, so you can
        evaluate diameter MAE/RMSE later.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # Create a linear layer for diameter (one scalar per instance)
        in_features = self.box_predictor.cls_score.in_features
        self.diameter_pred = torch.nn.Linear(in_features + 1, 1)
        torch.nn.init.normal_(self.diameter_pred.weight, mean=0.0, std=0.001)
        torch.nn.init.zeros_(self.diameter_pred.bias)

    def _forward_box(self, features, proposals):
        """
        Args:
          features (dict[str, Tensor]): feature maps from the backbone (FPN, etc.)
          proposals (list[Instances]): each Instances contains proposal_boxes & possibly gt_boxes & gt_classes & gt_diameter during training

        Returns:
          - During TRAINING: a dict of losses, including "loss_cls", "loss_box_reg", "loss_mask" (if mask head present), and "loss_diameter".
          - During INFERENCE: a list[Instances], one per input image, each Instances containing fields:
                - pred_boxes, scores, pred_classes, pred_masks (if mask head present),
                - pred_diameter (Tensor[N_preds,]).
        """
        features_list = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        if self.training:
            # Standard classification and box regression losses
            losses = self.box_predictor.losses(predictions, proposals)
            # Optionally refine boxes
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(predictions, proposals)
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            # Diameter regression loss (L1 loss)
            # Recupera il fattore di scala per ciascun proposal
            scales = [getattr(p, "resize_scale", torch.ones(len(p), device=box_features.device)).unsqueeze(1) for p in proposals]
            scale_tensor = torch.cat(scales, dim=0)

            # Concatena scale al box features
            extended_features = torch.cat([box_features, scale_tensor], dim=1)
            diam_pred = self.diameter_pred(extended_features).view(-1)
            # Collect ground-truth diameters from proposals
            gt_diams = [p.gt_diameter for p in proposals]
            if len(gt_diams) > 0:
                gt_diams = torch.cat(gt_diams, dim=0)
                losses["loss_diameter"] = F.l1_loss(diam_pred, gt_diams)
            return losses
        else:
            # ===== INFERENCE Branch =====
            # (a) Get predicted Instances (list of Instances, one per image) from box_predictor
            pred_instances_list, _ = self.box_predictor.inference(predictions, proposals)

            # (b) Compute predicted diameters for all RoIs (same order as proposals flattened)
            with torch.no_grad():
                diam_vals = self.diameter_pred(box_features).view(-1)

            # (c) Split diam_vals into chunks—one chunk per image—based on how many
            #     proposals (i.e. predicted instances) were kept for that image.
            # Note: box_predictor.inference returns pred_instances_list in the same order
            # as the proposals list passed in, but only for those proposals that survived NMS.
            #       So we collect the counts from pred_instances_list directly.
            counts = [len(inst) for inst in pred_instances_list]
            idx = 0
            for inst, count in zip(pred_instances_list, counts):
                if count > 0:
                    # Assign slice [idx:idx+count] to inst.pred_diameter
                    inst.pred_diameter = diam_vals[idx : idx + count]
                else:
                    inst.pred_diameter = torch.tensor([], dtype=torch.float32)
                idx += count

            return pred_instances_list

# 3. Custom mapper to include 'gt_diameter'
from detectron2.data import DatasetMapper
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

class MyMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(is_train=is_train, cfg=cfg)

    def __call__(self, dataset_dict):
        # Extract original annotations including diameter
        dataset_dict = copy.deepcopy(dataset_dict)
        orig_annos = dataset_dict.get("annotations", [])
        diameters = [anno.get("diameter") for anno in orig_annos]
        # Use default mapper for image, boxes, masks
        data = super().__call__(dataset_dict)
        # Attach diameter to Instances if present
        if hasattr(data["instances"], "gt_classes") and len(diameters) > 0:
            image = data["image"]
            # Load and process depth map
            import numpy as np

            import os

            # Infer .npy path from image filename if depth_file is missing
            depth_path = dataset_dict.get("depth_file")
            if depth_path is None:
                filename = os.path.basename(dataset_dict["file_name"]).replace(".png", ".npy")
                #if "train" in dataset_dict["file_name"]:
                depth_path = os.path.join(
                    "/content/drive/MyDrive/RGB-D_diameter_thesis/data/single_largest_apple_crops/depth", filename
                )
                #elif "val" in dataset_dict["file_name"]:
                #    depth_path = os.path.join(
                #        "/content/drive/MyDrive/RGB-D_diameter_thesis/data/cropped_margin_depth/val", filename
                #    )
            if depth_path is None:
                raise FileNotFoundError(f"Missing 'depth_file' in dataset_dict for image: {dataset_dict.get('file_name')}")

            # Load .npy file
            depth_array = np.load(depth_path)  # shape: (H, W), dtype usually float32 or uint16
            depth_tensor = torch.tensor(depth_array).unsqueeze(0).float()  # → [1, H, W]

            # Optional: normalize from mm to meters (or any other scaling)
            depth_tensor = depth_tensor / 1000.0

            # Resize depth tensor to match RGB
            depth_tensor = F.interpolate(depth_tensor.unsqueeze(0), size=image.shape[1:], mode="bilinear", align_corners=False)[0]

            # Concatenate with RGB image
            image = torch.cat([image, depth_tensor], dim=0)
            data["image"] = image
            # Optional: Normalize depth (e.g., mm → [0, 1])
            depth_tensor = depth_tensor / 10000.0

            new_h, new_w = image.shape[1], image.shape[2]
            orig_h = dataset_dict["height"]
            orig_w = dataset_dict["width"]
            scale = new_w / orig_w

            scaled_diams = [d * scale for d in diameters]
            data["instances"].gt_diameter = torch.tensor(scaled_diams, dtype=torch.float32)
            data["instances"].resize_scale = torch.tensor([scale] * len(scaled_diams), dtype=torch.float32)
        return data

# 4. Configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.MASK_ON = False
cfg.DATASETS.TRAIN = ("apple_train",)
cfg.DATASETS.TEST = ("apple_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NAME = "CustomROIHeads"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # apple
# Initialize with pretrained Mask R-CNN weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.OUTPUT_DIR = output_dir
cfg.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53, 2.0]
cfg.MODEL.PIXEL_STD  = [58.395,  57.12,  57.375, 1.0]
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# 5. Trainer using custom mapper
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=MyMapper(cfg, True))

# Modify the first conv layer to accept 4 channels (RGB + Depth)
import torch.nn as nn

model = CustomTrainer.build_model(cfg)
conv1 = model.backbone.bottom_up.stem.conv1

# Only patch if current conv1 expects 3 channels
if conv1.in_channels == 3:
    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = conv1.weight  # copy RGB weights
        new_conv.weight[:, 3] = conv1.weight[:, 0]  # init depth from red
    model.backbone.bottom_up.stem.conv1 = new_conv


trainer = CustomTrainer(cfg)
trainer.model = model
trainer.resume_or_load(resume=False)
trainer.train()