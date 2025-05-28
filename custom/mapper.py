import copy
import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

class RGBDMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        # define your augmentations here, using T.RandomFlip, etc.
        self.augmentations = [T.RandomFlip(prob=0.5, horizontal=True, vertical=False)]
        self.img_format = cfg.INPUT.FORMAT  # expect "RGBD"

    def __call__(self, dataset_dict):
        # 1. Deep copy input dict
        d = copy.deepcopy(dataset_dict)

        # 2. Load RGB and depth
        img = utils.read_image(d["file_name"], format="BGR")
        depth = np.load(d["depth_file"])  # shape HxW

        # 3. Apply same transforms to both
        aug_input = T.AugInput(image=img, sem_seg=depth)
        transforms = T.AugmentationList(self.augmentations)(aug_input)
        img = aug_input.image
        depth = aug_input.sem_seg

        # 4. Normalize + convert to torch.Tensor
        img = utils.convert_image_to_rgb(img)  # BGR→RGB
        img = utils.normalize(img, self.img_format)  # (3,H,W)
        depth = (depth - depth.mean()) / (depth.std() + 1e-6)
        depth = torch.as_tensor(depth[None], dtype=torch.float32)  # (1,H,W)

        # 5. Pack into a single 4-channel tensor
        image = torch.cat([img, depth], dim=0)

        # 6. Process annotations
        annos = d.get("annotations", [])
        instances = utils.annotations_to_instances(annos, image.shape[1:], mask_format="bitmask")

        return {"image": image, "instances": instances, "height": d["height"], "width": d["width"]}