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
    Iteration based OWPipelineHook
    """

    def __init__(self, log_start_iter=0, select_iter=600):
        """
        Args:
            log_start_iter: Logging start iteration
            select_iter: attribute selection start iteration
        """
        self.log_start_iter = log_start_iter
        self.select_iter = select_iter
        self.logging_enabled = False
        self.selection_done = False

        print(f"OWPipelineHook initialized: log_start={log_start_iter}, select={select_iter}")

    def before_step(self):
        """check before iteration"""
        current_iter = self.trainer.iter

        # logging start
        if current_iter == self.log_start_iter and not self.logging_enabled:
            model = self._get_model()
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'enable_log'):
                model.sem_seg_head.enable_log() # initialize positive / negative_distributions
                self.logging_enabled = True
                print(f"[Iter {current_iter}] Enabled attribute logging")
            else:
                print(f"[Iter {current_iter}] Cannot enable logging - missing methods")

    def after_step(self):
        """check after iteration"""
        current_iter = self.trainer.iter

        # Start Attribute selection
        if current_iter == self.select_iter and not self.selection_done:
            model = self._get_model()
            if hasattr(model, 'sem_seg_head') and hasattr(model.sem_seg_head, 'select_att'):
                try:
                    print(f"[Iter {current_iter}] Starting attribute selection...")
                    model.sem_seg_head.select_att()
                    self.selection_done = True
                    print(f"[Iter {current_iter}] Attribute selection completed")

                    # # 선택 후 로깅 비활성화
                    # if hasattr(model.sem_seg_head, 'disable_log'):
                    #     model.sem_seg_head.disable_log()
                    #     print(f"🔒 [Iter {current_iter}] Disabled attribute logging")

                except Exception as e:
                    print(f"[Iter {current_iter}] Attribute selection failed: {e}")
            else:
                print(f"[Iter {current_iter}] Cannot perform selection - missing methods")

    def _get_model(self):
        """DDP 모델 처리"""
        if hasattr(self.trainer.model, 'module'):
            print(f"self.trainer.model.module : {self.trainer.model.module}")
            return self.trainer.model.module
        else:
            print(f"self.trainer.model : {self.trainer.model}")
            return self.trainer.model


class OWSemSegEvaluator(DatasetEvaluator):
    """
    Open-World Semantic Segmentation Evaluator.
    Works with original ADE150 order, uses known/unknown indices for evaluation.
    """

    @configurable
    def __init__(
            self, dataset_name, distributed, output_dir=None, *,
            prev_intro_cls=None, cur_intro_cls=None, unknown_class_index=None, ignore_label=None,
            evaluation_mode=None, ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (True): if True, will collect results from all ranks for evaluation.
            output_dir (str): an output directory to dump results.
            prev_intro_cls (int): previous introduced classes
            cur_intro_cls (int): current introduced classes
            unknown_class_index (int): index for unknown class (used in OWSS mode)
            ignore_label (int): label to ignore
            evaluation_mode (str): "OVSS" or "OWSS"
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
        self.evaluation_mode = evaluation_mode

        # 원복된 메타데이터에서 정보 가져오기
        self._num_known_classes = self.prev_intro_cls + self.cur_intro_cls  # 75
        self._class_names = meta.stuff_classes  # 원본 ADE150 순서
        self._known_class_indices = set(meta.known_class_indices)  # Known 클래스 인덱스들
        self._unknown_class_indices = set(meta.unknown_class_indices)  # Unknown 클래스 인덱스들
        self._num_gt_classes = len(self._class_names)  # 150

        # Mode-specific settings
        if evaluation_mode == "OVSS":
            # OVSS: 150 classes, 원본 인덱스 그대로 사용
            self._num_pred_classes = len(meta.stuff_classes)
            self._logger.info(f"OVSS Mode:")
            self._logger.info(f"  Known class indices: {sorted(list(self._known_class_indices))[:10]}...")
            self._logger.info(f"  Unknown class indices: {sorted(list(self._unknown_class_indices))[:10]}...")

        elif evaluation_mode == "OWSS":
            # OWSS: 76 classes (0~75), Known: 원본 인덱스 그대로, Unknown: 150
            self._num_pred_classes = self._num_known_classes + 1  # 75 known + 1 unknown
            self._unknown_pred_index = self._num_known_classes  # 75
            self._unknown_gt_index = 150  # GT에서 unknown은 150번
            self._logger.info(f"OWSS Mode:")
            self._logger.info(f"  Known class indices: {sorted(list(self._known_class_indices))[:10]}...")
            self._logger.info(f"  Unknown GT index: {self._unknown_gt_index}")
        else:
            raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")

        self._ignore_label = ignore_label
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")

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
            "evaluation_mode": cfg.MODEL.SEM_SEG_HEAD.EVALUATION_MODE,
        }
        return ret

    def reset(self):
        if self.evaluation_mode == "OVSS":
            # OVSS: 150x151 confusion matrix (원본 인덱스 기준)
            self._conf_matrix = np.zeros((150, 151), dtype=np.int64)  # +1 for ignore
        elif self.evaluation_mode == "OWSS":
            # OWSS: 76x77 confusion matrix
            # Pred: 0~75, GT: 0~74(known remapped) + 75(unknown) + 76(ignore)
            self._conf_matrix = np.zeros((76, 77), dtype=np.int64)

        self._predictions = []

    def process(self, inputs, outputs):
        """Process predictions and ground truth"""
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=int)

            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=int)

            # Mode-specific processing
            if self.evaluation_mode == "OVSS":
                pred_processed, gt_processed = self.process_ovss(pred, gt)
            elif self.evaluation_mode == "OWSS":
                pred_processed, gt_processed = self.process_owss(pred, gt)

            # Update confusion matrix
            assert pred_processed.shape == gt_processed.shape
            self._conf_matrix += np.bincount(
                self._conf_matrix.shape[1] * pred_processed.reshape(-1) + gt_processed.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

    def process_ovss(self, pred, gt):
        """
        OVSS Mode processing - 원본 순서 그대로 사용
        """
        print(f"[OVSS] Pred range: {pred.min()}~{pred.max()}, GT range: {gt.min()}~{gt.max()}")

        # Handle ignore label (255 → 150)
        gt_processed = gt.copy()
        gt_processed[gt_processed == self._ignore_label] = 150

        return pred, gt_processed

    def process_owss(self, pred, gt):
        """
        OWSS Mode processing - Known 클래스를 0~74로, Unknown을 75로 리매핑
        """
        print(f"[OWSS] Pred range: {pred.min()}~{pred.max()}, GT range: {gt.min()}~{gt.max()}")
        gt_remapped = gt.copy()

        # Known 클래스들을 0~74로 리매핑 (sorted order로 매핑)
        known_indices_sorted = sorted(list(self._known_class_indices))
        for new_idx, original_idx in enumerate(known_indices_sorted):
            gt_remapped[gt == original_idx] = new_idx

        # Unknown 클래스들을 75번으로 리매핑
        for unknown_idx in self._unknown_class_indices:
            gt_remapped[gt == unknown_idx] = 75

        # Handle ignore label (255 → 76)
        gt_remapped[gt == self._ignore_label] = 76

        # 통계 출력
        known_pixels = np.sum((gt_remapped >= 0) & (gt_remapped < 75))
        unknown_pixels = np.sum(gt_remapped == 75)
        ignore_pixels = np.sum(gt_remapped == 76)

        print(f"[OWSS] Remapped GT range: {gt_remapped.min()}~{gt_remapped.max()}")
        print(f"[OWSS] Known pixels: {known_pixels}, Unknown pixels: {unknown_pixels}, Ignore pixels: {ignore_pixels}")

        return pred, gt_remapped

    def evaluate(self):
        """Evaluate with mode-specific metrics"""
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

        # Mode-specific evaluation
        if self.evaluation_mode == "OVSS":
            return self.evaluate_ovss()
        elif self.evaluation_mode == "OWSS":
            return self.evaluate_owss()

    def evaluate_ovss(self):
        """
        OVSS Mode evaluation - Known/Unknown 인덱스 기반 분리 평가
        """
        print("[OVSS] Evaluating Open-Vocabulary mode with known/unknown indices...")

        # Exclude ignore label column
        valid_conf_matrix = self._conf_matrix[:, :-1]  # 150x150

        # Calculate per-class metrics
        acc = np.full(150, np.nan, dtype=float)
        iou = np.full(150, np.nan, dtype=float)

        tp = valid_conf_matrix.diagonal().astype(float)
        pos_gt = np.sum(valid_conf_matrix, axis=0).astype(float)
        pos_pred = np.sum(valid_conf_matrix, axis=1).astype(float)

        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]

        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]

        # Overall metrics
        class_weights = pos_gt / np.sum(pos_gt) if np.sum(pos_gt) > 0 else pos_gt
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid) if np.sum(acc_valid) > 0 else 0.0
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid) if np.sum(iou_valid) > 0 else 0.0
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid]) if np.sum(iou_valid) > 0 else 0.0
        pacc = np.sum(tp) / np.sum(pos_gt) if np.sum(pos_gt) > 0 else 0.0

        # Known vs Unknown metrics using class indices
        known_iou_values = [iou[i] for i in self._known_class_indices if i < len(iou) and iou_valid[i]]
        unknown_iou_values = [iou[i] for i in self._unknown_class_indices if i < len(iou) and iou_valid[i]]

        known_miou = np.mean(known_iou_values) if known_iou_values else 0.0
        unknown_miou = np.mean(unknown_iou_values) if unknown_iou_values else 0.0

        if known_miou > 0 and unknown_miou > 0:
            harmonic_mean = 2 * known_miou * unknown_miou / (known_miou + unknown_miou)
        else:
            harmonic_mean = 0.0

        print(f"[OVSS] Known classes: {known_miou:.4f}")
        print(f"[OVSS] Unknown classes: {unknown_miou:.4f}")
        print(f"[OVSS] Harmonic Mean: {harmonic_mean:.4f}")
        print(f"[OVSS] Known classes count: {len(known_iou_values)}/{len(self._known_class_indices)}")
        print(f"[OVSS] Unknown classes count: {len(unknown_iou_values)}/{len(self._unknown_class_indices)}")

        return self.format_results(miou, fiou, macc, pacc, known_miou, unknown_miou, harmonic_mean, "OVSS")

    def evaluate_owss(self):
        """
        OWSS Mode evaluation - 76x77 confusion matrix
        """
        print("[OWSS] Evaluating Open-World mode...")

        # Exclude ignore label column (76번 컬럼 제외)
        valid_conf_matrix = self._conf_matrix[:, :-1]  # 76x76

        print(f"[OWSS] Confusion matrix shape: {valid_conf_matrix.shape}")
        print(f"[OWSS] Total pixels: {valid_conf_matrix.sum()}")

        # Known classes IoU (개별 계산)
        known_iou_values = []
        for i in range(75):  # known classes 0~74
            tp = valid_conf_matrix[i, i]
            fp = np.sum(valid_conf_matrix[i, :]) - tp  # row sum - diagonal
            fn = np.sum(valid_conf_matrix[:, i]) - tp  # column sum - diagonal

            if (tp + fp + fn) > 0:
                iou = tp / (tp + fp + fn)
                known_iou_values.append(iou)

        # Known mIoU
        known_miou = np.mean(known_iou_values) if known_iou_values else 0.0

        # Unknown class metrics (pred 75 vs GT 75)
        unknown_pred_idx = 75
        unknown_gt_idx = 75

        # True Positive: GT unknown (75) predicted as unknown (75)
        unknown_tp = valid_conf_matrix[unknown_pred_idx, unknown_gt_idx]

        # False Positive: GT known (0~74) predicted as unknown (75)
        unknown_fp = np.sum(valid_conf_matrix[unknown_pred_idx, :75])

        # False Negative: GT unknown (75) predicted as known (0~74)
        unknown_fn = np.sum(valid_conf_matrix[:75, unknown_gt_idx])

        print(f"[OWSS] Known metrics - Valid classes: {len(known_iou_values)}/75, mIoU: {known_miou:.4f}")
        print(f"[OWSS] Unknown metrics - TP: {unknown_tp}, FP: {unknown_fp}, FN: {unknown_fn}")

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

        print(f"[OWSS] Known mIoU: {known_miou:.4f}")
        print(f"[OWSS] Unknown mIoU: {unknown_miou:.4f}")
        print(f"[OWSS] Harmonic Mean: {harmonic_mean:.4f}")

        # Overall metrics
        all_tp = sum([valid_conf_matrix[i, i] for i in range(75)]) + unknown_tp
        all_pixels = valid_conf_matrix.sum()

        # 가중 평균 mIoU (known:unknown = 75:1)
        weighted_miou = (known_miou * 75 + unknown_miou * 1) / 76

        return self.format_results(
            miou=weighted_miou,
            fiou=weighted_miou,
            macc=known_miou,  # Approximation
            pacc=all_tp / all_pixels if all_pixels > 0 else 0.0,
            known_miou=known_miou,
            unknown_miou=unknown_miou,
            harmonic_mean=harmonic_mean,
            mode="OWSS"
        )

    def format_results(self, miou, fiou, macc, pacc, known_miou, unknown_miou, harmonic_mean, mode):
        """Format evaluation results"""
        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        res["Known_mIoU"] = 100 * known_miou
        res["Unknown_mIoU"] = 100 * unknown_miou
        res["Harmonic_Mean"] = 100 * harmonic_mean
        res["seen_IoU"] = 100 * known_miou
        res["unseen_IoU"] = 100 * unknown_miou
        res["harmonic mean"] = 100 * harmonic_mean

        if self._output_dir:
            file_path = os.path.join(self._output_dir, f"sem_seg_evaluation_{mode.lower()}.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)

        results = OrderedDict({"sem_seg": res})
        self._logger.info(f"[{mode}] Results: {results}")
        return results


    # def evaluate(self):
    #     """
    #     Evaluates standard semantic segmentation metrics with separation for known/unknown classes.
    #     """
    #     if self._distributed:
    #         synchronize()
    #         conf_matrix_list = all_gather(self._conf_matrix)
    #         self._predictions = all_gather(self._predictions)
    #         self._predictions = list(itertools.chain(*self._predictions))
    #         if not is_main_process():
    #             return
    #
    #         self._conf_matrix = np.zeros_like(self._conf_matrix)
    #         for conf_matrix in conf_matrix_list:
    #             self._conf_matrix += conf_matrix
    #
    #     if self._output_dir:
    #         PathManager.mkdirs(self._output_dir)
    #         file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
    #         with PathManager.open(file_path, "w") as f:
    #             f.write(json.dumps(self._predictions))
    #
    #     # ignore_label 제외한 유효한 confusion matrix 사용
    #     valid_conf_matrix = self._conf_matrix[:-1, :-1]
    #
    #     print(f"[DEBUG] Valid confusion matrix shape: {valid_conf_matrix.shape}")
    #     print(f"[DEBUG] Known classes: 0-{self._num_seen_classes - 1}")
    #     print(f"[DEBUG] Unknown classes: {self._num_seen_classes}-{self._num_classes - 2}")
    #     print(f"[DEBUG] Unknown prediction index: {self._num_classes - 1}")
    #
    #     # 기본 메트릭 계산
    #     acc = np.full(self._num_classes, np.nan, dtype=float)
    #     iou = np.full(self._num_classes, np.nan, dtype=float)
    #     tp = valid_conf_matrix.diagonal().astype(float)
    #     pos_gt = np.sum(valid_conf_matrix, axis=0).astype(float)
    #     class_weights = pos_gt / np.sum(pos_gt) if np.sum(pos_gt) > 0 else pos_gt
    #     pos_pred = np.sum(valid_conf_matrix, axis=1).astype(float)
    #     acc_valid = pos_gt > 0
    #     acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
    #     iou_valid = (pos_gt + pos_pred) > 0
    #     union = pos_gt + pos_pred - tp
    #     iou[iou_valid] = tp[iou_valid] / union[iou_valid]
    #
    #     macc = np.sum(acc[acc_valid]) / np.sum(acc_valid) if np.sum(acc_valid) > 0 else 0.0
    #     miou = np.sum(iou[iou_valid]) / np.sum(iou_valid) if np.sum(iou_valid) > 0 else 0.0
    #     fiou = np.sum(iou[iou_valid] * class_weights[iou_valid]) if np.sum(iou_valid) > 0 else 0.0
    #     pacc = np.sum(tp) / np.sum(pos_gt) if np.sum(pos_gt) > 0 else 0.0
    #
    #     print("[DEBUG] Confusion Matrix Sample (Top 5 Predict vs GT):")
    #     top_pred = np.argsort(pos_pred)[-5:][::-1]
    #     top_gt = np.argsort(pos_gt)[-5:][::-1]
    #     print("GT Class:", top_gt)
    #     print("Pred Class:", top_pred)
    #
    #     total_pixels = np.sum(pos_pred)
    #     unknown_pixels = pos_pred[150] if len(pos_pred) > 150 else 0
    #     print(
    #         f"[DEBUG] Unknown(150) Predict Ratio: {int(unknown_pixels) / int(total_pixels) * 100:.1f}% ({int(unknown_pixels)} / {int(total_pixels)})")
    #
    #     # CAT-Seg + OW-OVD Unknown mIoU
    #     known_classes_end = self._num_seen_classes  # 75
    #     unknown_pred_index = self._num_classes - 1  # 150
    #     unknown_gt_start = known_classes_end  # 75
    #     unknown_gt_end = self._num_classes - 1  # 150
    #
    #     # Known mIoU
    #     known_iou_values = []
    #     for i in range(known_classes_end):
    #         if i < len(iou) and iou_valid[i]:
    #             known_iou_values.append(iou[i])
    #     known_miou = np.mean(known_iou_values) if known_iou_values else 0.0
    #
    #     # Unknown mIoU
    #     unknown_tp = 0  # True Positive
    #     unknown_fp = 0  # False Positive
    #     unknown_fn = 0  # False Negative
    #
    #     # True Positive
    #     for gt_idx in range(unknown_gt_start, unknown_gt_end):
    #         unknown_tp += valid_conf_matrix[gt_idx, gt_idx]  # diagonal
    #
    #     for gt_idx in range(unknown_gt_start, unknown_gt_end):
    #         unknown_tp += valid_conf_matrix[unknown_pred_index, gt_idx]
    #
    #     # False Positive
    #     for gt_idx in range(known_classes_end):  # GT Known
    #         for pred_idx in range(unknown_gt_start, unknown_gt_end):  # Pred Individual Unknown
    #             unknown_fp += valid_conf_matrix[pred_idx, gt_idx]
    #
    #     for gt_idx in range(known_classes_end):  # GT Known
    #         unknown_fp += valid_conf_matrix[unknown_pred_index, gt_idx]
    #
    #     # False Negative
    #     for gt_idx in range(unknown_gt_start, unknown_gt_end):  # GT Unknown
    #         for pred_idx in range(known_classes_end):  # Pred Known
    #             unknown_fn += valid_conf_matrix[pred_idx, gt_idx]
    #
    #         for pred_idx in range(unknown_gt_start, unknown_gt_end):
    #             if pred_idx != gt_idx:
    #                 unknown_fn += valid_conf_matrix[pred_idx, gt_idx]
    #
    #     print(f"[DEBUG] Unknown TP: {unknown_tp}")
    #     print(f"[DEBUG] Unknown FP: {unknown_fp}")
    #     print(f"[DEBUG] Unknown FN: {unknown_fn}")
    #
    #     # Unknown IoU
    #     if (unknown_tp + unknown_fp + unknown_fn) > 0:
    #         unknown_miou = unknown_tp / (unknown_tp + unknown_fp + unknown_fn)
    #     else:
    #         unknown_miou = 0.0
    #
    #     # Harmonic Mean
    #     if known_miou > 0 and unknown_miou > 0:
    #         harmonic_mean = 2 * known_miou * unknown_miou / (known_miou + unknown_miou)
    #     else:
    #         harmonic_mean = 0.0
    #
    #     print(f"[DEBUG] Known mIoU: {known_miou:.4f}")
    #     print(f"[DEBUG] Unknown mIoU: {unknown_miou:.4f}")
    #     print(f"[DEBUG] Harmonic Mean: {harmonic_mean:.4f}")
    #
    #     # Result
    #     res = {}
    #     res["mIoU"] = 100 * miou
    #     res["fwIoU"] = 100 * fiou
    #     res["mACC"] = 100 * macc
    #     res["pACC"] = 100 * pacc
    #
    #     # CAT-Seg Metric
    #     res["Known_mIoU"] = 100 * known_miou
    #     res["Unknown_mIoU"] = 100 * unknown_miou
    #     res["Harmonic_Mean"] = 100 * harmonic_mean
    #
    #     # Previous Metric
    #     res["seen_IoU"] = 100 * known_miou
    #     res["unseen_IoU"] = 100 * unknown_miou
    #     res["harmonic mean"] = 100 * harmonic_mean
    #
    #     # Class Metrics
    #     for i, name in enumerate(self._class_names[:self._num_classes]):
    #         if i < len(iou):
    #             res[f"IoU-{name}"] = 100 * iou[i]
    #         if i < len(acc):
    #             res[f"ACC-{name}"] = 100 * acc[i]
    #
    #     if self._output_dir:
    #         file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
    #         with PathManager.open(file_path, "wb") as f:
    #             torch.save(res, f)
    #
    #     results = OrderedDict({"sem_seg": res})
    #     self._logger.info(results)
    #     return results

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
    Evaluate semantic segmentation metrics.
    """
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            pred[pred >= 20] = 20
            with PathManager.open(self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

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


# trainer = Trainer(cfg)
# └─> DefaultTrainer.__init__(cfg)
#     ├─> Trainer.build_model(cfg)           ✓ 호출됨
#     ├─> Trainer.build_optimizer(cfg, model) ✓ 호출됨
#     ├─> Trainer.build_train_loader(cfg)     ✓ 호출됨
#     └─> Trainer.build_lr_scheduler(cfg, opt) ✓ 호출됨
#
# trainer.train()
# └─> DefaultTrainer.train()
#     └─> 필요시 Trainer.build_evaluator(...)  ✓ 호출됨

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
    res = trainer.train()
    return res

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
