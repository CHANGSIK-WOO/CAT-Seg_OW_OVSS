# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import copy
import itertools
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, configurable

from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import CityscapesInstanceEvaluator, CityscapesSemSegEvaluator, \
    COCOEvaluator, COCOPanopticEvaluator, DatasetEvaluators, SemSegEvaluator, verify_results, \
    DatasetEvaluator

from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

import numpy as np
from PIL import Image
import glob

import pycocotools.mask as mask_util
import json

from detectron2.engine.hooks import HookBase


class OWPipelineHook(HookBase):
    """
    ✅ 수정된 OW Pipeline Hook - Iteration 기반으로 명확한 동작
    """

    def __init__(self, log_start_iter=0, select_iter=600):
        """
        Args:
            log_start_iter: 로깅 시작 iteration
            select_iter: attribute selection 수행 iteration
        """
        self.log_start_iter = log_start_iter
        self.select_iter = select_iter
        self.logging_enabled = False
        self.selection_done = False

        print(f"✅ OWPipelineHook initialized: log_start={log_start_iter}, select={select_iter}")

    def before_step(self):
        """각 iteration 시작 전 체크"""
        current_iter = self.trainer.iter

        # 로깅 시작
        if current_iter == self.log_start_iter and not self.logging_enabled:
            model = self._get_model()
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'enable_log'):
                model.sem_seg_head.enable_log()
                self.logging_enabled = True
                print(f"✅ [Iter {current_iter}] Enabled attribute logging")
            else:
                print(f"❌ [Iter {current_iter}] Cannot enable logging - missing methods")

    def after_step(self):
        """각 iteration 완료 후 체크"""
        current_iter = self.trainer.iter

        # Attribute selection 수행
        if current_iter == self.select_iter and not self.selection_done:
            model = self._get_model()
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'select_att'):
                try:
                    print(f"📊 [Iter {current_iter}] Starting attribute selection...")
                    model.sem_seg_head.select_att()
                    self.selection_done = True
                    print(f"✅ [Iter {current_iter}] Attribute selection completed")

                    # 선택 후 로깅 비활성화
                    if hasattr(model.sem_seg_head, 'disable_log'):
                        model.sem_seg_head.disable_log()
                        print(f"🔒 [Iter {current_iter}] Disabled attribute logging")

                except Exception as e:
                    print(f"❌ [Iter {current_iter}] Attribute selection failed: {e}")
            else:
                print(f"❌ [Iter {current_iter}] Cannot perform selection - missing methods")

    def _get_model(self):
        """DDP 모델 처리"""
        if hasattr(self.trainer.model, 'module'):
            return self.trainer.model.module
        else:
            return self.trainer.model

class OWSemSegEvaluator(DatasetEvaluator):
    """
    Open-World Semantic Segmentation Evaluator.
    Evaluates both known (seen) and unknown (unseen) classes separately.
    """

    @configurable
    def __init__(
            self, dataset_name, distributed, output_dir=None, *,
            prev_intro_cls=None, cur_intro_cls=None, unknown_class_index=None, ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (True): if True, will collect results from all ranks for evaluation.
            output_dir (str): an output directory to dump results.
            num_classes, ignore_label: deprecated argument
            known_classes_end (int): last index of known classes (0-indexed)
        """
        self._logger = logging.getLogger(__name__)

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)

        self.prev_intro_cls = prev_intro_cls
        self.cur_intro_cls = cur_intro_cls
        self.unknown_class_index = unknown_class_index
        self._num_seen_classes = self.prev_intro_cls + self.cur_intro_cls
        self._class_names = meta.stuff_classes + (["unknown"] if self.unknown_class_index is not None else [])
        self._num_classes = len(self._class_names)
        self._known_classes = self._class_names[:self._num_seen_classes]
        self._val_extra_classes = self._class_names[self._num_seen_classes:]
        self._ignore_label = ignore_label

        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")

        self._logger.info(f"Known classes: {len(self._known_classes)} (0-{len(self._known_classes)-1})")
        self._logger.info(f"Unknown classes: {len(self._class_names) - len(self._known_classes)} ({len(self._known_classes)}-{len(self._class_names)-1})")

    @classmethod
    def from_config(cls, cfg, dataset_name, distributed, output_dir=None):

        ret = {
            "dataset_name": dataset_name,
            "distributed": distributed,
            "output_dir": output_dir,
            "prev_intro_cls": cfg.MODEL.SEM_SEG_HEAD.PREV_INTRO_CLS,
            "cur_intro_cls": cfg.MODEL.SEM_SEG_HEAD.CUR_INTRO_CLS,
            "unknown_class_index": cfg.MODEL.SEM_SEG_HEAD.UNKNOWN_ID,
            "ignore_label": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
        }
        return ret

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes+1, self._num_classes+1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
            outputs: the outputs of a model.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=int)

            #ignore gt == self._ignore_label point
            valid_mask = (gt != self._ignore_label)
            gt_valid = gt[valid_mask]
            pred_valid = pred[valid_mask]

            gt[gt == self._ignore_label] = self._num_classes

            print("[DEBUG] Confusion matrix shape:", self._conf_matrix.shape)
            print("[DEBUG] GT unique values:", sorted(np.unique(gt_valid)))
            print("[DEBUG] Pred unique values:", sorted(np.unique(pred_valid)))
            print("[DEBUG] GT range:", int(gt_valid.min()), "to", int(gt_valid.max()))
            print("[DEBUG] Pred range:", int(gt_valid.min()), "to", int(pred_valid.max()))

            print(f"pred.shape : {pred.shape}, gt.shape : {gt.shape}")
            print(f"pred.shape : {pred.reshape(-1)}, gt.shape : {gt.reshape(-1)}")
            assert pred.reshape(-1).shape==gt.reshape(-1).shape, "shape of pred and gt have to be same."
            self._conf_matrix += np.bincount(
                (self._num_classes+1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            # self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics with separation for known/unknown classes.
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        # ignore_label 제외한 유효한 confusion matrix 사용
        valid_conf_matrix = self._conf_matrix[:-1, :-1]

        print(f"[DEBUG] Valid confusion matrix shape: {valid_conf_matrix.shape}")
        print(f"[DEBUG] Known classes: 0-{self._num_seen_classes - 1}")
        print(f"[DEBUG] Unknown classes: {self._num_seen_classes}-{self._num_classes - 2}")
        print(f"[DEBUG] Unknown prediction index: {self._num_classes - 1}")

        # 기본 메트릭 계산
        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = valid_conf_matrix.diagonal().astype(np.float)
        pos_gt = np.sum(valid_conf_matrix, axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt) if np.sum(pos_gt) > 0 else pos_gt
        pos_pred = np.sum(valid_conf_matrix, axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]

        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid) if np.sum(acc_valid) > 0 else 0.0
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid) if np.sum(iou_valid) > 0 else 0.0
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid]) if np.sum(iou_valid) > 0 else 0.0
        pacc = np.sum(tp) / np.sum(pos_gt) if np.sum(pos_gt) > 0 else 0.0

        print("[DEBUG] Confusion Matrix Sample (Top 5 Predict vs GT):")
        top_pred = np.argsort(pos_pred)[-5:][::-1]
        top_gt = np.argsort(pos_gt)[-5:][::-1]
        print("GT Class:", top_gt)
        print("Pred Class:", top_pred)

        total_pixels = np.sum(pos_pred)
        unknown_pixels = pos_pred[150] if len(pos_pred) > 150 else 0
        print(
            f"[DEBUG] Unknown(150) Predict Ratio: {int(unknown_pixels) / int(total_pixels) * 100:.1f}% ({int(unknown_pixels)} / {int(total_pixels)})")

        # CAT-Seg + OW-OVD Unknown mIoU
        known_classes_end = self._num_seen_classes  # 75
        unknown_pred_index = self._num_classes - 1  # 150
        unknown_gt_start = known_classes_end  # 75
        unknown_gt_end = self._num_classes - 1  # 150

        # Known mIoU
        known_iou_values = []
        for i in range(known_classes_end):
            if i < len(iou) and iou_valid[i]:
                known_iou_values.append(iou[i])
        known_miou = np.mean(known_iou_values) if known_iou_values else 0.0

        # Unknown mIoU
        unknown_tp = 0  # True Positive
        unknown_fp = 0  # False Positive
        unknown_fn = 0  # False Negative

        # True Positive
        for gt_idx in range(unknown_gt_start, unknown_gt_end):
            unknown_tp += valid_conf_matrix[gt_idx, gt_idx]  # diagonal

        for gt_idx in range(unknown_gt_start, unknown_gt_end):
            unknown_tp += valid_conf_matrix[unknown_pred_index, gt_idx]

        # False Positive
        for gt_idx in range(known_classes_end):  # GT Known
            for pred_idx in range(unknown_gt_start, unknown_gt_end):  # Pred Individual Unknown
                unknown_fp += valid_conf_matrix[pred_idx, gt_idx]

        for gt_idx in range(known_classes_end):  # GT Known
            unknown_fp += valid_conf_matrix[unknown_pred_index, gt_idx]

        # False Negative
        for gt_idx in range(unknown_gt_start, unknown_gt_end):  # GT Unknown
            for pred_idx in range(known_classes_end):  # Pred Known
                unknown_fn += valid_conf_matrix[pred_idx, gt_idx]

            for pred_idx in range(unknown_gt_start, unknown_gt_end):
                if pred_idx != gt_idx:
                    unknown_fn += valid_conf_matrix[pred_idx, gt_idx]

        print(f"[DEBUG] Unknown TP: {unknown_tp}")
        print(f"[DEBUG] Unknown FP: {unknown_fp}")
        print(f"[DEBUG] Unknown FN: {unknown_fn}")

        # Unknown IoU
        if (unknown_tp + unknown_fp + unknown_fn) > 0:
            unknown_miou = unknown_tp / (unknown_tp + unknown_fp + unknown_fn)
        else:
            unknown_miou = 0.0

        # Harmonic Mean
        if known_miou > 0 and unknown_miou > 0:
            harmonic_mean = 2 * known_miou * unknown_miou / (known_miou + unknown_miou)
        else:
            harmonic_mean = 0.0

        print(f"[DEBUG] Known mIoU: {known_miou:.4f}")
        print(f"[DEBUG] Unknown mIoU: {unknown_miou:.4f}")
        print(f"[DEBUG] Harmonic Mean: {harmonic_mean:.4f}")

        # Result
        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc

        # CAT-Seg Metric
        res["Known_mIoU"] = 100 * known_miou
        res["Unknown_mIoU"] = 100 * unknown_miou
        res["Harmonic_Mean"] = 100 * harmonic_mean

        # Previous Metric
        res["seen_IoU"] = 100 * known_miou
        res["unseen_IoU"] = 100 * unknown_miou
        res["harmonic mean"] = 100 * harmonic_mean

        # Class Metrics
        for i, name in enumerate(self._class_names[:self._num_classes]):
            if i < len(iou):
                res[f"IoU-{name}"] = 100 * iou[i]
            if i < len(acc):
                res[f"ACC-{name}"] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                if label in self._contiguous_id_to_dataset_id:
                    dataset_id = self._contiguous_id_to_dataset_id[label]
                else:
                    continue  # Skip unknown labels
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list

class VOCbEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics for VOC background dataset.
    """

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
            outputs: the outputs of a model.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            pred[pred >= 20] = 20  # Clip predictions to 20 classes
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=np.int)


            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))
# MaskFormer
from cat_seg import (
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    OWMaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_cat_seg_config,
)


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # ✅ 수정: 파일 존재 확인 및 안전한 Hook 등록
        att_embeddings_path = cfg.MODEL.SEM_SEG_HEAD.ATT_EMBEDDINGS

        if att_embeddings_path and os.path.exists(att_embeddings_path):
            print(f"✅ Found attribute embeddings: {att_embeddings_path}")

            # iteration 기반 스케줄링
            log_start_iter = cfg.get('ATTRIBUTE_LOG_START_ITER', 0)
            select_iter = cfg.get('ATTRIBUTE_SELECT_ITER', 600)

            # OWPipelineHook 등록
            pipeline_hook = OWPipelineHook(
                log_start_iter=log_start_iter,
                select_iter=select_iter
            )
            self.register_hooks([pipeline_hook])

            print(f"✅ OWPipelineHook registered: start={log_start_iter}, select={select_iter}")

        else:
            print(f"❌ Attribute embeddings not found: {att_embeddings_path}")
            print("⚠️  OW pipeline will be disabled. Please generate embeddings first:")
            print("    python generate_clip_embeddings.py --classnames datasets/ade150.json --out_dir data/coco/")

        self.distributions_path = cfg.MODEL.SEM_SEG_HEAD.DISTRIBUTIONS

    def train(self):
        """✅ 수정: 훈련 완료 후 distribution 저장"""
        try:
            result = super().train()
            self._save_distributions_immediately()
            return result
        except Exception as e:
            print(f"❌ Training error: {e}")
            try:
                self._save_distributions_immediately()
            except Exception as save_err:
                print(f"❌ Failed to save distributions: {save_err}")
            raise e

    def _save_distributions_immediately(self):
        """훈련 완료 즉시 distribution 저장"""
        if not self.distributions_path or not comm.is_main_process():
            return

        try:
            print("💾 Attempting to save distributions...")

            model = self.model.module if hasattr(self.model, 'module') else self.model

            if (hasattr(model, 'sem_seg_head') and
                    hasattr(model.sem_seg_head, 'positive_distributions') and
                    hasattr(model.sem_seg_head, 'negative_distributions')):

                pos_dist = model.sem_seg_head.positive_distributions
                neg_dist = model.sem_seg_head.negative_distributions

                if pos_dist is not None and neg_dist is not None:
                    distributions_data = {
                        'positive_distributions': pos_dist,
                        'negative_distributions': neg_dist,
                        'thresholds': getattr(model.sem_seg_head, 'thrs', [0.75]),
                        'num_attributes': len(pos_dist[0]) if pos_dist and len(pos_dist) > 0 else 0,
                        'saved_at': 'training_completion'
                    }

                    os.makedirs(os.path.dirname(self.distributions_path), exist_ok=True)
                    torch.save(distributions_data, self.distributions_path)
                    print(f"✅ Distributions saved to: {self.distributions_path}")

                    # 검증
                    if os.path.exists(self.distributions_path):
                        saved_data = torch.load(self.distributions_path, map_location='cpu')
                        print(f"📊 Verified: {saved_data.get('num_attributes', 0)} attributes saved")
                    else:
                        print("❌ File verification failed")
                else:
                    print("⚠️ No distribution data collected during training")
            else:
                print("⚠️ Distribution attributes not found in model")

        except Exception as e:
            print(f"❌ Error saving distributions: {e}")
            import traceback
            traceback.print_exc()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        # Open-World Semantic Segmentation evaluator
        if evaluator_type == "ow_sem_seg":
            evaluator_list.append(
                OWSemSegEvaluator(
                    cfg,
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        if evaluator_type == "sem_seg_background":
            evaluator_list.append(
                VOCbEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # OW-Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "ow_mask_former_semantic":
            mapper = OWMaskFormerSemanticDatasetMapper(cfg, True)
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        # DETR-style dataset mapper for COCO panoptic segmentation
        elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic":
            mapper = DETRPanopticDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        # import ipdb;
        # ipdb.set_trace()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "clip_model" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.CLIP_MULTIPLIER
                # for deformable detr

                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def main(args):
    cfg = setup(args)
    torch.set_float32_matmul_precision("high")

    import os
    abs_path = os.path.abspath(str(cfg.MODEL.SEM_SEG_HEAD.DISTRIBUTIONS))
    print(f"[SAVE-CHECK] target abs path = {abs_path}")
    print(f"[SAVE-CHECK] is_main_process = {comm.is_main_process()}, rank = {comm.get_rank()}")

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    res = trainer.train()  # 훈련 실행
    #
    # # ✅ 수정: 훈련 실행 (distribution은 train() 메서드에서 자동 저장됨)
    # try:
    #     res = trainer.train()
    #     print("✅ Training completed successfully")
    # except Exception as e:
    #     print(f"❌ Training failed: {e}")
    #     # 에러 발생 시에도 파일 존재 확인
    #     if os.path.exists(abs_path):
    #         print(f"✅ Distributions file exists despite error: {abs_path}")
    #     else:
    #         print(f"❌ Distributions file not found: {abs_path}")
    #     raise e
    #
    # # ✅ 추가 확인: 파일 생성 검증
    # if comm.is_main_process():
    #     if os.path.exists(abs_path):
    #         print(f"✅ Final verification: Distributions file successfully created at {abs_path}")
    #         try:
    #             data = torch.load(abs_path, map_location='cpu')
    #             print(f"📊 File contains {data.get('num_attributes', 'unknown')} attributes")
    #         except Exception as e:
    #             print(f"⚠️ File exists but cannot be loaded: {e}")
    #     else:
    #         print(f"❌ Final verification failed: Distributions file not found at {abs_path}")

    return res


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
