# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_cat_seg_config

# dataset loading
from .data.dataset_mappers.detr_panoptic_dataset_mapper import DETRPanopticDatasetMapper
from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.ow_mask_former_semantic_dataset_mapper import (
    OWMaskFormerSemanticDatasetMapper,
)

# models
from .cat_seg_model import CATSeg
from .modeling.heads.cat_seg_head import CATSegHead
from .test_time_augmentation import SemanticSegmentorWithTTA

# ow-ovss
from .ow_cat_seg_model import OWCATSeg
from .modeling.heads.ow_cat_seg_head import OWCATSegHead
