import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy

def _get_ade20k_150_meta():
    original_sequence_class_ade150 = [
        "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ",
        "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
        "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting",
        "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat",
        "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
        "base", "box", "column", "signboard", "chest of drawers", "counter", "sand",
        "sink", "skyscraper", "refrigerator", "grandstand", "path", "stairs", "runway",
        "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
        "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
        "bench", "countertop", "stove", "palm", "kitchen island", "computer",
        "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
        "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
        "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
        "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage",
        "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything",
        "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag",
        "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name",
        "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen",
        "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray",
        "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board",
        "shower", "radiator", "glass", "clock", "flag"
    ]
    modified_sequence_class_ade150 = [
        "tree", "road", "bed", "grass", "cabinet", "person", "table", "mountain",
        "curtain", "chair", "car", "shelf", "house", "sea", "rug", "fence", "rock",
        "railing", "counter", "sand", "sink", "skyscraper", "refrigerator", "stairs",
        "pillow", "river", "bridge", "toilet", "flower", "book", "hill", "bench",
        "boat", "bus", "towel", "light", "truck", "airplane", "bottle", "tent",
        "oven", "microwave", "bicycle", "blanket", "vase", "traffic light", "clock",
        "wall", "building", "sky", "floor", "ceiling", "windowpane", "sidewalk",
        "earth", "door", "plant", "water", "sofa", "mirror", "field", "armchair",
        "desk", "blind", "coffee table", "computer", "swivel chair", "apparel",
        "land", "bag", "ball", "food", "screen", "glass", "stool", "painting",
        "seat", "wardrobe", "lamp", "bathtub", "cushion", "base", "box", "column",
        "signboard", "chest of drawers", "fireplace", "grandstand", "path", "runway",
        "case", "pool table", "screen door", "stairway", "bookcase", "countertop",
        "stove", "palm", "kitchen island", "bar", "arcade machine", "hovel", "tower",
        "chandelier", "awning", "streetlight", "booth", "television receiver",
        "dirt track", "pole", "bannister", "escalator", "ottoman", "buffet", "poster",
        "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
        "plaything", "swimming pool", "barrel", "basket", "waterfall", "minibike",
        "cradle", "step", "tank", "trade name", "pot", "animal", "lake", "dishwasher",
        "sculpture", "hood", "sconce", "tray", "ashcan", "fan", "pier", "crt screen",
        "plate", "monitor", "bulletin board", "shower", "radiator", "flag"
    ]
    original_sequence_class_indices = [
        4, 6, 7, 9, 10, 12, 15, 16, 18, 19, 20, 24, 25, 26, 28, 32, 34, 38, 45, 46,
        47, 48, 49, 52, 56, 59, 60, 64, 65, 66, 67, 68, 75, 79, 80, 81, 82, 89, 97, 113,
        117, 123, 126, 130, 134, 135, 147, 0, 1, 2, 3, 5, 8, 11, 13, 14, 17, 21, 23,
        27, 29, 30, 33, 62, 63, 73, 74, 91, 93, 114, 118, 119, 129, 146, 109, 22, 31,
        35, 36, 37, 39, 40, 41, 42, 43, 44, -1, 50, 51, 53, 54, 55, 57, 58, 61, 69,
        70, 71, 72, 76, 77, 78, 83, 84, 85, 86, 87, 88, 90, 92, 94, 95, 96, 98, 99,
        100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 115, 116, 120,
        121, 122, 124, 125, 127, 128, 131, 132, 133, 136, 137, 138, 139, 140, 141,
        142, 143, 144, 145, 148
    ]

    ret = {
        "original_sequence_class_ade150" : original_sequence_class_ade150,
        "modified_sequence_class_ade150": modified_sequence_class_ade150,
        "original_sequence_class_indices" : original_sequence_class_indices,
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
        DatasetCatalog.register(name_ow, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name_ow).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="ow_sem_seg", ignore_label=255, **meta,)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade20k_150_semantic_segmentation(_root)