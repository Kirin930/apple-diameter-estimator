import os
import argparse
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer

def setup_cfg(args):
    cfg = get_cfg()
    # load base config
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
    # merge your overrides
    cfg.merge_from_file("configs/mask_rcnn_r50_rgbd.yaml")
    # register datasets
    from custom.dataset import register_datasets
    register_datasets(args.train_json, args.val_json)
    # set training & testing splits
    cfg.DATASETS.TRAIN = ("apple_train",)
    cfg.DATASETS.TEST  = ("apple_val",)
    # use custom mapper
    cfg.INPUT.MAPPER_NAME = "custom.mapper.RGBDMapper"
    # output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True, help="path to train JSON")
    parser.add_argument("--val-json",   required=True, help="path to val JSON")
    args = parser.parse_args()

    cfg = setup_cfg(args)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()