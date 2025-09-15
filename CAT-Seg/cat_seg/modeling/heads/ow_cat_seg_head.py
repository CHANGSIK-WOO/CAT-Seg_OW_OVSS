# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from einops import rearrange

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.ow_cat_seg_predictor import OWCATSegPredictor


@SEM_SEG_HEADS_REGISTRY.register()
class OWCATSegHead(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,

        #ow-ovss new
        device="cuda",
        att_embeddings: Optional[str] = None,
        prev_intro_cls: int = 0,
        cur_intro_cls: int = 0,
        unknown_cls: int = 0,
        thr: float = 0.8,
        alpha: float = 0.5,
        use_sigmoid: bool = True,
        prev_distribution: Optional[str] = None,
        distributions: Optional[str] = None,
        top_k: int = 10,
        fusion_att: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            ignore_value: category id to be ignored during training.
            feature_resolution: resolution of the feature map
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

        # ow-ovss new
        self.device = device
        self.thr = thr
        self.alpha = alpha
        self.use_sigmoid = use_sigmoid
        self.distributions = distributions
        self.thrs = [thr]
        self.prev_intro_cls = prev_intro_cls
        self.cur_intro_cls = cur_intro_cls
        self.unknown_cls = unknown_cls
        self.prev_distribution = prev_distribution
        self.top_k = top_k
        self.fusion_att = fusion_att

        self.load_att_embeddings(att_embeddings)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": OWCATSegPredictor(cfg),

            # ow-ovss new
            "att_embeddings": cfg.MODEL.SEM_SEG_HEAD.ATT_EMBEDDINGS,
            "prev_intro_cls": cfg.MODEL.SEM_SEG_HEAD.PREV_INTRO_CLS,
            "cur_intro_cls": cfg.MODEL.SEM_SEG_HEAD.CUR_INTRO_CLS,
            "unknown_cls": cfg.MODEL.SEM_SEG_HEAD.UNKNOWN_CLS,
            "thr": cfg.MODEL.SEM_SEG_HEAD.THR,
            "alpha": cfg.MODEL.SEM_SEG_HEAD.ALPHA,
            "use_sigmoid": cfg.MODEL.SEM_SEG_HEAD.USE_SIGMOID,
            "prev_distribution": cfg.MODEL.SEM_SEG_HEAD.PREV_DISTRIBUTION,
            "distributions": cfg.MODEL.SEM_SEG_HEAD.DISTRIBUTIONS,
            "top_k": cfg.MODEL.SEM_SEG_HEAD.TOP_K,
            "fusion_att": cfg.MODEL.SEM_SEG_HEAD.FUSION_ATT,
        }

    @property
    def clip_dtype(self):
        """CLIP 모델의 dtype을 자동으로 감지"""
        return next(self.predictor.clip_model.parameters()).dtype

    def disable_log(self):
        self.positive_distributions = None
        self.negative_distributions = None
        print('disable log')

    def enable_log(self):
        self.reset_log()
        print('enable log')

    def load_att_embeddings(self, att_embeddings):
        if att_embeddings is None:
            self.att_embeddings = None
            self.disable_log()
            return
        atts = torch.load(att_embeddings)
        self.texts = atts['att_text']
        self.all_atts = atts['att_embedding'].to(dtype=self.clip_dtype)
        if self.prev_distribution is not None:
            # todo this
            prev_atts_num = len(torch.load(self.prev_distribution, map_location='cuda')['positive_distributions'][self.thrs.index(self.thr)])
        else:
            prev_atts_num = 0
        self.att_embeddings = torch.nn.Parameter(atts['att_embedding'].to(dtype=self.clip_dtype)[prev_atts_num:])
        self.enable_log()
        # self.att_embeddings = torch.nn.Parameter(torch.zeros(1000, 512).float())

    def reset_log(self, interval=0.0001):
        """Reset the log."""
        # [0, 1] interval = 0.0001
        if self.att_embeddings is None: return
        self.positive_distributions = [{att_i: torch.zeros(int((1) / interval)).to(self.device) for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]
        self.negative_distributions = [{att_i: torch.zeros(int((1) / interval)).to(self.device) for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]

    def calculate_uncertainty(self, known_logits):
        known_logits = torch.clamp(known_logits, 1e-6, 1 - 1e-6)
        entropy = (-known_logits * torch.log(known_logits) - (1 - known_logits) * torch.log(1 - known_logits)).mean(
            dim=-1, keepdim=True)
        return entropy

    def select_top_k_attributes(self, adjusted_scores: Tensor, k: int = 3) -> Tensor:
        top_k_scores, _ = adjusted_scores.topk(k, dim=-1)
        top_k_average = top_k_scores.mean(dim=-1, keepdim=True)
        return top_k_average

    def compute_weighted_top_k_attributes(self, adjusted_scores: Tensor, k: int = 10) -> Tensor:
        top_k_scores, top_k_indices = adjusted_scores.topk(k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        weighted_average = torch.sum(top_k_scores * top_k_weights, dim=-1, keepdim=True)
        return weighted_average

    def get_sim(self, a, b):
        """Return distribution a and b similarity using JS divergence."""

        def jensen_shannon_divergence(p, q):
            m = 0.5 * (p + q)
            m = m.clamp(min=1e-6)
            js_div = 0.5 * (torch.sum(p * torch.log((p / m).clamp(min=1e-6))) +
                            torch.sum(q * torch.log((q / m).clamp(min=1e-6))))
            return js_div

        return jensen_shannon_divergence(a, b)

    def get_all_dis_sim(self, positive_dis, negative_dis):
        dis_sim = []
        for i in range(len(positive_dis)):
            positive = positive_dis[i]
            negative = negative_dis[i]
            positive = positive / positive.sum()
            negative = negative / negative.sum()
            dis_sim.append(self.get_sim(positive, negative))
        return torch.stack(dis_sim).to('cuda')

    # def select_att(self, per_class=25):
    #     """Select attributes based on distribution similarity and diversity."""
    #     if self.distributions is None or self.att_embeddings is None:
    #         return
    #
    #     distributions = torch.load(self.distributions, map_location='cuda')
    #     self.positive_distributions, self.negative_distributions = distributions['positive_distributions'], \
    #     distributions['negative_distributions']
    #
    #     thr_id = self.thrs.index(self.thr)
    #     distribution_sim = self.get_all_dis_sim(self.positive_distributions[thr_id],
    #                                             self.negative_distributions[thr_id])
    #
    #     all_atts = self.all_atts.to(self.att_embeddings.device)
    #     att_embeddings_norm = F.normalize(all_atts, p=2, dim=1)
    #     if self.use_sigmoid:
    #         cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).sigmoid()
    #     else:
    #         cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).abs()
    #
    #     selected_indices = []
    #     for _ in range(per_class * self.num_classes):
    #         if len(selected_indices) == 0:
    #             _, idx = distribution_sim.min(dim=0)
    #         else:
    #             unselected_indices = list(set(range(len(self.texts))) - set(selected_indices))
    #             cosine_sim_with_selected = cosine_sim_matrix[unselected_indices][:, selected_indices].mean(dim=1)
    #             distribution_sim_unselected = distribution_sim[unselected_indices]
    #             score = self.alpha * distribution_sim_unselected + (1 - self.alpha) * cosine_sim_with_selected
    #             idx = unselected_indices[score.argmin()]
    #         selected_indices.append(idx)
    #
    #     selected_indices = torch.tensor(selected_indices).to(self.att_embeddings.device)
    #     self.att_embeddings = torch.nn.Parameter(all_atts[selected_indices]).to(self.att_embeddings.device)
    #     self.texts = [self.texts[i] for i in selected_indices]
    #
    #     print(f"Attribute selection completed: selected {len(selected_indices)} attributes from current training data")

    def select_att(self, per_class=25):
        """Select attributes based on distribution similarity and diversity."""
        if self.att_embeddings is None:
            print("No att_embeddings available, skipping attribute selection")
            return

        # 현재 수집된 distributions가 있는지 확인
        if (self.positive_distributions is None or
                self.negative_distributions is None):
            print("No distributions collected in current training, skipping attribute selection")
            return

        import os

        # 파일이 있는지 확인하여 분기 처리
        if self.distributions is not None and os.path.exists(self.distributions):
            print(f"Loading existing distributions from {self.distributions}")
            try:
                # 기존 파일에서 데이터 로드
                saved_distributions = torch.load(self.distributions, map_location='cuda')
                saved_positive = saved_distributions['positive_distributions']
                saved_negative = saved_distributions['negative_distributions']

                # 기존 데이터와 현재 수집된 데이터를 결합/업데이트
                thr_id = self.thrs.index(self.thr)

                # 기존 데이터를 사용하여 계산
                distribution_sim = self.get_all_dis_sim(saved_positive[thr_id],
                                                        saved_negative[thr_id])
                print("Using existing distributions for attribute selection")

            except Exception as e:
                print(f"Error loading existing distributions: {e}")
                print("Falling back to current training data")
                # 에러 발생시 현재 데이터 사용
                thr_id = self.thrs.index(self.thr)
                distribution_sim = self.get_all_dis_sim(self.positive_distributions[thr_id],
                                                        self.negative_distributions[thr_id])
        else:
            print("No existing distributions found, using current training data")
            # 파일이 없으면 현재 수집된 데이터만 사용
            thr_id = self.thrs.index(self.thr)
            distribution_sim = self.get_all_dis_sim(self.positive_distributions[thr_id],
                                                    self.negative_distributions[thr_id])

        # 전체 attribute 개수 확인
        total_attributes = len(self.texts)
        target_selection_count = per_class * self.num_classes

        # 선택하려는 개수가 전체보다 많으면 제한
        actual_selection_count = min(target_selection_count, total_attributes)
        print(
            f"Total attributes: {total_attributes}, Target: {target_selection_count}, Actual: {actual_selection_count}")

        # attribute selection 로직
        all_atts = self.all_atts.to(self.att_embeddings.device)
        att_embeddings_norm = F.normalize(all_atts, p=2, dim=1)
        if self.use_sigmoid:
            cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).sigmoid()
        else:
            cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).abs()

        selected_indices = []
        for _ in range(actual_selection_count):  # ← 수정: 실제 선택 가능한 개수로 제한
            if len(selected_indices) == 0:
                _, idx = distribution_sim.min(dim=0)
            else:
                unselected_indices = list(set(range(total_attributes)) - set(selected_indices))

                # 빈 리스트 체크
                if len(unselected_indices) == 0:
                    print(f"All attributes selected. Total selected: {len(selected_indices)}")
                    break

                cosine_sim_with_selected = cosine_sim_matrix[unselected_indices][:, selected_indices].mean(dim=1)
                distribution_sim_unselected = distribution_sim[unselected_indices]
                score = self.alpha * distribution_sim_unselected + (1 - self.alpha) * cosine_sim_with_selected

                # score가 빈 텐서인지 체크
                if score.numel() == 0:
                    print("Score tensor is empty, stopping selection")
                    break

                idx = unselected_indices[score.argmin()]
            selected_indices.append(idx)

        selected_indices = torch.tensor(selected_indices).to(self.att_embeddings.device)
        self.att_embeddings = torch.nn.Parameter(all_atts[selected_indices]).to(self.att_embeddings.device)
        self.texts = [self.texts[i] for i in selected_indices]

        print(f"Attribute selection completed: selected {len(selected_indices)} attributes")


    def log_distribution(self, att_scores, gt_labels, gt_masks):
        """Log distribution of attributes for known/unknown classes."""
        if not self.training or self.positive_distributions is None or self.att_embeddings is None:
            return

        B, C, H, W = att_scores.shape
        att_scores = att_scores.sigmoid()

        for b in range(B):
            for idx, thr in enumerate(self.thrs):
                # Process only valid regions
                valid_mask = (gt_labels[b] != self.ignore_value)
                if not valid_mask.any():
                    continue

                gt_labels_valid = gt_labels[b][valid_mask]
                att_scores_valid = att_scores[b][:, valid_mask]  # C x N

                # Determine positive/negative based on known classes
                positive = (gt_labels_valid < self.unknown_cls) & (gt_labels_valid >= self.prev_intro_cls)

                if positive.any():
                    positive_scores = att_scores_valid[:, positive].T  # N_pos x C
                    for att_i in range(C):
                        self.positive_distributions[idx][att_i] += torch.histc(
                            positive_scores[:, att_i], bins=int(1 / 0.0001), min=0, max=1)

                if (~positive).any():
                    negative_scores = att_scores_valid[:, ~positive].T  # N_neg x C
                    for att_i in range(C):
                        self.negative_distributions[idx][att_i] += torch.histc(
                            negative_scores[:, att_i], bins=int(1 / 0.0001), min=0, max=1)


    def forward(self, features, guidance_features, prompt=None, gt_cls=None):
        """
        Arguments:
            features: (B, C, HW)
            guidance_features: (B, C, H, W)
        """
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat, guidance_features, prompt, gt_cls, self.att_embeddings, self.fusion_att)