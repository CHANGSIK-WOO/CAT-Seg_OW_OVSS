# Open-World Open-Vocabulary Semantic Segmentation (OW-OVSS)

This is an implementation of Open-World Open-Vocabulary Semantic Segmentation based on CAT-Seg, adapted from Open-World Object Detection (OW-OVD) concepts.

## Key Features

- **Open-World Setting**: Distinguishes between known (seen) and unknown (unseen) classes
- **Attribute-based Unknown Detection**: Uses attribute embeddings to detect unknown classes
- **Comprehensive Evaluation**: Provides separate metrics for known and unknown classes
- **Harmonic Mean**: Reports harmonic mean between known and unknown class performance

## Dataset Setup

### ADE20K-150 Open-World Split

- **Known Classes**: 0-74 (75 classes)
- **Unknown Classes**: 75-149 (75 classes)
- **Total**: 150 classes

The split is automatically handled by the dataset registration in `cat_seg/data/datasets/register_ade20k_150.py`.

## Usage

### Training

```bash
# Train OW-OVSS model
sh ow_run.sh configs/ow_ade20k_vitb_384.yaml 4 outputs/ow_ovss_ade20k
```

### Evaluation Only

```bash
# Evaluate trained model
sh ow_eval.sh configs/ow_ade20k_vitb_384.yaml 4 outputs/ow_ovss_ade20k
```

### Single Commands

```bash
# Training
python ow_train_net.py --config configs/ow_ade20k_vitb_384.yaml --num-gpus 4 OUTPUT_DIR outputs/ow_ovss_ade20k

# Evaluation
python ow_train_net.py --config configs/ow_ade20k_vitb_384.yaml --num-gpus 4 --eval-only \
    OUTPUT_DIR outputs/ow_ovss_ade20k/eval \
    MODEL.WEIGHTS outputs/ow_ovss_ade20k/model_final.pth \
    DATASETS.TEST "(\"ade20k_150_ow_val_sem_seg\",)"
```

## Key Components

### 1. OW-OVSS Evaluator (`ow_train_net.py`)

The `OWSemSegEvaluator` class provides:
- Separate evaluation for known (0-74) and unknown (75-149) classes
- Standard semantic segmentation metrics (mIoU, mACC, pACC)
- Open-World specific metrics:
  - `Known_mIoU`: mIoU for known classes only
  - `Unknown_mIoU`: mIoU for unknown classes only  
  - `Harmonic_Mean`: Harmonic mean between known and unknown mIoU

### 2. Dataset Registration (`cat_seg/data/datasets/register_ade20k_150.py`)

Registers two versions of ADE20K-150:
- `ade20k_150_*_sem_seg`: Standard semantic segmentation dataset
- `ade20k_150_ow_*_sem_seg`: Open-World version with `evaluator_type="ow_sem_seg"`

### 3. OW-OVSS Model (`cat_seg/ow_cat_seg_model.py`)

The `OWCATSeg` model extends CAT-Seg with:
- Attribute embedding integration
- Unknown class prediction using uncertainty and top-k attributes
- Distribution logging for attribute analysis

### 4. OW-OVSS Head (`cat_seg/modeling/heads/ow_cat_seg_head.py`)

The `OWCATSegHead` provides:
- Attribute embedding management
- Unknown class prediction logic
- Distribution tracking for attribute selection

## Configuration

Key configuration parameters in `configs/ow_ade20k_vitb_384.yaml`:

```yaml
MODEL:
  SEM_SEG_HEAD:
    NUM_CLASSES: 150           # Total number of classes
    UNKNOWN_CLS: 75            # First unknown class index
    ATT_EMBEDDINGS: "..."      # Path to attribute embeddings
    TOP_K: 10                  # Top-k attributes for unknown prediction
    THR: 0.75                  # Threshold for attribute prediction
    ALPHA: 0.3                 # Alpha for attribute selection
```

## Expected Results

The evaluation will output metrics including:

```
=== Open-World Semantic Segmentation Results ===
Overall mIoU: XX.XX
Known Classes mIoU: XX.XX
Unknown Classes mIoU: XX.XX  
Harmonic Mean: XX.XX
```

## File Structure

```
CAT-Seg/
├── ow_train_net.py                                    # Main training script
├── ow_run.sh                                          # Training + evaluation script  
├── ow_eval.sh                                         # Evaluation script
├── configs/ow_ade20k_vitb_384.yaml                   # Open-World config
├── cat_seg/
│   ├── data/
│   │   ├── datasets/register_ade20k_150.py           # Dataset registration
│   │   └── dataset_mappers/
│   │       └── mask_former_semantic_dataset_mapper.py # Dataset mapper
│   ├── modeling/heads/ow_cat_seg_head.py             # OW-OVSS head
│   └── ow_cat_seg_model.py                           # OW-OVSS model
└── datasets/ade150.json                              # Class names
```

## Notes

1. Make sure to prepare attribute embeddings and distributions files as specified in the config
2. The model requires CLIP pretrained weights (ViT-B/16 by default)
3. Adjust batch size and learning rates according to your hardware setup
4. The harmonic mean provides a balanced view of both known and unknown class performance

## Troubleshooting

1. **Missing dataset**: Ensure ADE20K dataset is properly downloaded and preprocessed
2. **CUDA out of memory**: Reduce `IMS_PER_BATCH` in config
3. **Missing attribute embeddings**: Check paths in config file and ensure files exist
4. **Import errors**: Verify all `__init__.py` files are present and imports are correct