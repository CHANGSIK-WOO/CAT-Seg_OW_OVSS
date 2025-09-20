import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
import copy

def _get_ade20k_150_meta():
    ade20k_150_classes = ["tree", "road", "bed", "grass", "cabinet", "person", "table", "mountain",
    "curtain", "chair", "car", "shelf", "house", "sea", "rug", "fence", "rock",
    "railing", "counter", "sand", "sink", "skyscraper", "refrigerator", "stairs",
    "pillow", "river", "bridge", "toilet", "flower", "book", "hill", "bench",
    "boat", "bus", "towel", "light", "truck", "airplane", "bottle", "tent",
    "oven", "microwave", "bicycle", "blanket", "vase", "traffic light", "clock",
    "wall", "building", "sky", "floor", "ceiling", "windowpane", "sidewalk",
    "earth", "door", "plant", "water", "sofa", "mirror", "field", "armchair",
    "desk", "blind", "coffee table", "computer", "swivel chair", "apparel",
    "land", "bag", "ball", "food", "screen", "glass", "stool",
    "painting", "seat", "wardrobe", "lamp", "bathtub", "cushion", "base", "box",
    "column", "signboard", "chest of drawers", "fireplace", "grandstand", "path",
    "runway", "case", "pool table", "screen door", "stairway", "bookcase",
    "countertop", "stove", "palm", "kitchen island", "bar", "arcade machine",
    "hovel", "tower", "chandelier", "awning", "streetlight", "booth",
    "television receiver", "dirt track", "pole", "bannister", "escalator",
    "ottoman", "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "barrel",
    "basket", "waterfall", "minibike", "cradle", "step", "tank", "trade name",
    "pot", "animal", "lake", "dishwasher", "sculpture", "hood", "sconce", "tray",
    "ashcan", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board",
    "shower", "radiator", "flag"]

    ret = {
        "stuff_classes" : ade20k_150_classes,
    }
    return ret

def register_ade20k_150(root):
    root = os.path.join(root, "ADEChallengeData2016")
    meta = _get_ade20k_150_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("test", "images/validation", "annotations_detectron2/validation"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"ade20k_150_{name}_sem_seg"
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(image_root=image_dir, seg_seg_root=gt_dir, evaluator_type="sem_seg", ignore_label=255, **meta,)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_ade20k_150(_root)
