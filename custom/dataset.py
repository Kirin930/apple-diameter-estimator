import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog

def load_rgbd_dicts(json_path):
    """
    Expects a JSON file where each entry has:
    {
      "file_name": str,
      "depth_file": str,
      "width": int,
      "height": int,
      "annotations": [
         {"bbox": [x, y, w, h], "category_id": 0, "diameter_gt": float},
         …
      ]
    }
    """
    with open(json_path) as f:
        data = json.load(f)
    dataset_dicts = []
    for idx, entry in enumerate(data):
        record = {
            "file_name": entry["file_name"],
            "depth_file": entry["depth_file"],
            "image_id": idx,
            "height": entry["height"],
            "width": entry["width"],
            "annotations": entry["annotations"],
        }
        dataset_dicts.append(record)
    return dataset_dicts

def register_datasets(train_json, val_json):
    DatasetCatalog.register("apple_train", lambda: load_rgbd_dicts(train_json))
    DatasetCatalog.register("apple_val",   lambda: load_rgbd_dicts(val_json))
    MetadataCatalog.get("apple_train").set(thing_classes=["apple"])
    MetadataCatalog.get("apple_val").set(thing_classes=["apple"])