# Copyright (c) Facebook, Inc. and its affiliates.
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# ADE20K 150 class names
ADE20K_150_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel",
    "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
    "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"
]

# Known classes (0-74) and Unknown classes (75-149)
KNOWN_CLASSES = ADE20K_150_CLASSES[:75]  # 0-74
UNKNOWN_CLASSES = ADE20K_150_CLASSES[75:]  # 75-149


def _get_ade20k_150_meta():
    """
    Get metadata for ADE20K 150 classes with Open-World setup
    """
    stuff_classes = ADE20K_150_CLASSES + ["unknown"]
    stuff_ids = [k for k in range(len(stuff_classes))]

    # Create mapping from category_id to contiguous_id
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "known_classes": KNOWN_CLASSES,
        "unknown_classes": UNKNOWN_CLASSES,
    }
    return ret


def register_ade20k_150_semantic_segmentation(root):
    root = os.path.join(root, "ADEChallengeData2016")
    meta = _get_ade20k_150_meta()

    for name, dirname in [("train", "training"), ("val", "validation"), ("test", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)

        # # Standard semantic segmentation dataset
        # name_standard = f"ade20k_150_{name}_sem_seg"
        # DatasetCatalog.register(
        #     name_standard,
        #     lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg"),
        # )
        # MetadataCatalog.get(name_standard).set(
        #     stuff_classes=meta["stuff_classes"],
        #     stuff_dataset_id_to_contiguous_id=meta["stuff_dataset_id_to_contiguous_id"],
        #     evaluator_type="sem_seg",
        #     ignore_label=255,
        #     **meta,
        # )

        # Open-World semantic segmentation dataset
        name_ow = f"ade20k_150_ow_{name}_sem_seg"
        DatasetCatalog.register(
            name_ow,
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg"),
        )
        MetadataCatalog.get(name_ow).set(
            # stuff_classes=meta["stuff_classes"],
            # stuff_dataset_id_to_contiguous_id=meta["stuff_dataset_id_to_contiguous_id"],
            evaluator_type="ow_sem_seg",  # Use Open-World evaluator
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade20k_150_semantic_segmentation(_root)