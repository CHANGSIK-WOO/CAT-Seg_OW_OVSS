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
        num_classes_train: int,
        num_classes_test: int,
        ignore_label: int = None,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,

        #ow-ovss new
        device="cuda",
        att_embeddings: Optional[str] = None,
        prev_intro_cls: int = None,
        cur_intro_cls: int = None,
        unknown_class_index: int = None,
        thr: float = None,
        alpha: float = None,
        use_sigmoid: bool = True,
        prev_distribution: Optional[str] = None,
        distributions: Optional[str] = None,
        top_k: int = None,
        fusion_att: bool = False,
        enable_ow_mode:bool = True, #ow mode control
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
        self.ignore_label = ignore_label
        self.predictor = transformer_predictor

        self.num_classes_train = num_classes_train
        self.num_classes_test = num_classes_test

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
        self.unknown_class_index = unknown_class_index
        self.prev_distribution = prev_distribution
        self.top_k = top_k
        self.fusion_att = fusion_att
        self.enable_ow_mode = enable_ow_mode  # NEW

        self.positive_distributions = None
        self.negative_distributions = None

        self.load_att_embeddings(att_embeddings)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return{
            "ignore_label": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes_train": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES_TRAIN,
            "num_classes_test": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES_TEST,

            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": OWCATSegPredictor(cfg),

            # ow-ovss new
            "att_embeddings": cfg.MODEL.SEM_SEG_HEAD.ATT_EMBEDDINGS,
            "prev_intro_cls": cfg.MODEL.SEM_SEG_HEAD.PREV_INTRO_CLS,
            "cur_intro_cls": cfg.MODEL.SEM_SEG_HEAD.CUR_INTRO_CLS,
            "unknown_class_index": cfg.MODEL.SEM_SEG_HEAD.UNKNOWN_ID,
            "thr": cfg.MODEL.SEM_SEG_HEAD.THR,
            "alpha": cfg.MODEL.SEM_SEG_HEAD.ALPHA,
            "use_sigmoid": cfg.MODEL.SEM_SEG_HEAD.USE_SIGMOID,
            "prev_distribution": cfg.MODEL.SEM_SEG_HEAD.PREV_DISTRIBUTION,
            "distributions": cfg.MODEL.SEM_SEG_HEAD.DISTRIBUTIONS,
            "top_k": cfg.MODEL.SEM_SEG_HEAD.TOP_K,
            "fusion_att": cfg.MODEL.SEM_SEG_HEAD.FUSION_ATT,
            "enable_ow_mode": cfg.MODEL.SEM_SEG_HEAD.ENABLE_OW_MODE,
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

    def reset_log(self, interval=0.0001):
        """Reset the log."""
        # [0, 1] interval = 0.0001
        if self.att_embeddings is None: return
        self.positive_distributions = [{att_i: torch.zeros(int((1) / interval)).to(self.device) for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]
        self.negative_distributions = [{att_i: torch.zeros(int((1) / interval)).to(self.device) for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]

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
        # self.att_embeddings = torch.nn.Parameter(torch.zeros(1000, 512).float())

    def calculate_uncertainty(self, known_logits):
        known_logits = torch.clamp(known_logits, 1e-6, 1 - 1e-6)
        entropy = (-known_logits * torch.log(known_logits) - (1 - known_logits) * torch.log(1 - known_logits)).mean(dim=-1, keepdim=True)
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

    # def get_sim(self, a, b):
    #     """
    #     Calculate Jensen-Shannon Divergence between two distributions
    #     Lower JSD means distributions are more similar (less discriminative)
    #     Higher JSD means distributions are more different (more discriminative)
    #     """
    #
    #     def jensen_shannon_divergence(p, q):
    #         # Ensure distributions are valid
    #         p = p.clamp(min=1e-8)
    #         q = q.clamp(min=1e-8)
    #
    #         # Calculate M = (P + Q) / 2
    #         m = 0.5 * (p + q)
    #         m = m.clamp(min=1e-8)
    #
    #         # Calculate KL divergences
    #         kl_pm = torch.sum(p * torch.log(p / m))
    #         kl_qm = torch.sum(q * torch.log(q / m))
    #
    #         # JSD = 0.5 * (KL(P||M) + KL(Q||M))
    #         js_div = 0.5 * (kl_pm + kl_qm)
    #         return js_div
    #
    #     return jensen_shannon_divergence(a, b)

    def get_all_dis_sim(self, positive_dis, negative_dis):
        dis_sim = []
        for i in range(len(positive_dis)):
            positive = positive_dis[i]
            negative = negative_dis[i]
            positive = positive / positive.sum()
            negative = negative / negative.sum()
            dis_sim.append(self.get_sim(positive, negative))
        return torch.stack(dis_sim).to('cuda')

    def select_att(self, per_class=25):
        """
        Step 5: Iterative Attribute Selection (VSAS)

        Implementation of Visual Similarity Attribute Selection algorithm
        """
        print(f"=== Step 5: Iterative Attribute Selection ===")
        print(f"att_embeddings available: {self.att_embeddings is not None}")
        print(f"positive_distributions available: {self.positive_distributions is not None}")
        print(f"negative_distributions available: {self.negative_distributions is not None}")
        print(f"distributions file path: {self.distributions}")

        if self.att_embeddings is None:
            print("❌ No att_embeddings available, skipping attribute selection")
            return

        # Load existing distributions if available
        saved_positive, saved_negative = None, None
        if self.distributions is not None and os.path.exists(self.distributions):
            print(f"📂 Loading existing distributions from {self.distributions}")
            try:
                distributions = torch.load(self.distributions, map_location='cuda')
                saved_positive = distributions['positive_distributions']
                saved_negative = distributions['negative_distributions']
                print("✅ Successfully loaded existing distributions")
            except Exception as e:
                print(f"❌ Error loading existing distributions: {e}")

        # Use current training data or saved data
        if (self.positive_distributions is not None and self.negative_distributions is not None):
            print("📊 Using current training distributions")
            thr_id = self.thrs.index(self.thr)
            current_positive = self.positive_distributions[thr_id]
            current_negative = self.negative_distributions[thr_id]

            # Combine with saved data if available
            if saved_positive is not None and saved_negative is not None:
                print("🔄 Combining current and saved distributions")
                combined_positive = {}
                combined_negative = {}
                for att_i in range(len(current_positive)):
                    combined_positive[att_i] = current_positive[att_i] + saved_positive[thr_id][att_i]
                    combined_negative[att_i] = current_negative[att_i] + saved_negative[thr_id][att_i]
                use_positive, use_negative = combined_positive, combined_negative
            else:
                use_positive, use_negative = current_positive, current_negative

        elif saved_positive is not None and saved_negative is not None:
            print("📂 Using only saved distributions")
            thr_id = self.thrs.index(self.thr)
            use_positive, use_negative = saved_positive[thr_id], saved_negative[thr_id]
        else:
            print("❌ No distributions available for selection")
            return

        # Step 5: Calculate JSD for each attribute
        distribution_similarities = self.calculate_all_jsd(use_positive, use_negative)

        # Get total number of attributes and target selection count
        total_attributes = self.att_embeddings.shape[0]
        target_selection_count = min(per_class * self.num_classes_train, total_attributes)

        print(f"🎯 Target selection: {target_selection_count}/{total_attributes} attributes")

        # Step 5: Iterative Attribute Selection
        all_atts = self.all_atts.to(self.att_embeddings.device)
        att_embeddings_norm = F.normalize(all_atts, p=2, dim=1)

        # Precompute cosine similarity matrix for redundancy penalty
        if self.use_sigmoid:
            cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).sigmoid()
        else:
            cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).abs()

        # Step 5.34-39: Iterative selection with JSD and redundancy penalty
        selected_indices = []

        for iteration in range(target_selection_count):
            if iteration == 0:
                # Step 5.36: First selection - minimize JSD (최소 JSD = 최대 차이)
                _, idx = distribution_similarities.min(dim=0)
                idx = idx.item()
            else:
                # Step 5.36-37: Subsequent selections with redundancy penalty
                unselected_indices = list(set(range(total_attributes)) - set(selected_indices))

                if len(unselected_indices) == 0:
                    print(f"⚠️ All attributes selected at iteration {iteration}")
                    break

                # Calculate redundancy penalty for unselected attributes
                cosine_sim_with_selected = cosine_sim_matrix[unselected_indices][:, selected_indices].mean(dim=1)
                distribution_sim_unselected = distribution_similarities[unselected_indices]

                # Step 5.36-37: β * JSD + (1-β) * RedundancyPenalty
                selection_score = self.alpha * distribution_sim_unselected + (1 - self.alpha) * cosine_sim_with_selected

                # Select attribute with minimum combined score
                min_score_idx = selection_score.argmin()
                idx = unselected_indices[min_score_idx.item()]

            selected_indices.append(idx)

            if (iteration + 1) % 100 == 0:
                print(f"🔄 Selected {iteration + 1}/{target_selection_count} attributes")

        # Step 5.38: Update selected attributes
        selected_indices = torch.tensor(selected_indices).to(self.att_embeddings.device)
        self.att_embeddings = torch.nn.Parameter(all_atts[selected_indices]).to(self.att_embeddings.device)
        self.texts = [self.texts[i] for i in selected_indices]

        print(f"✅ Attribute selection completed: {len(selected_indices)} attributes selected")
        print(
            f"📈 Final JSD range: {distribution_similarities[selected_indices].min():.4f} - {distribution_similarities[selected_indices].max():.4f}")

    def calculate_all_jsd(self, positive_distributions, negative_distributions):
        """
        Step 4.26-32: Calculate Jensen-Shannon Divergence for all attributes

        Returns:
            torch.Tensor: JSD values for each attribute [num_attributes]
        """
        jsd_values = []

        for att_i in range(len(positive_distributions)):
            positive_dist = positive_distributions[att_i]
            negative_dist = negative_distributions[att_i]

            # Normalize distributions
            positive_normalized = positive_dist / (positive_dist.sum() + 1e-8)
            negative_normalized = negative_dist / (negative_dist.sum() + 1e-8)

            # Calculate JSD
            jsd = self.get_sim(positive_normalized, negative_normalized)
            jsd_values.append(jsd)

        return torch.stack(jsd_values).to('cuda')

    def log_distribution(self, att_scores, assigned_scores, valid_masks):
        """
        Step 3-4: Distribution Construction & Similarity Calculation
        Args:
            att_scores: [B, C_att, H, W] - attribute prediction scores (after sigmoid)
            assigned_scores: [B, C_known, H, W] - known class prediction scores  
            valid_masks: [B, H, W] - valid pixel mask (not ignore_label)
        """
        if not self.training or self.positive_distributions is None or self.att_embeddings is None:
            print("log_distribution: conditions not met, returning")
            return

        num_att = att_scores.shape[1]
        num_known = assigned_scores.shape[1]

        # Flatten for processing [B*H*W, C]
        att_scores_flat = att_scores.sigmoid().permute(0, 2, 3, 1).reshape(-1, num_att).float()
        assigned_scores_flat = assigned_scores.permute(0, 2, 3, 1).reshape(-1, num_known)
        valid_masks_flat = valid_masks.reshape(-1)

        # Apply valid masks
        att_scores_valid = att_scores_flat[valid_masks_flat]  # [N_valid, C_att]
        assigned_scores_valid = assigned_scores_flat[valid_masks_flat]  # [N_valid, C_known]

        if att_scores_valid.size(0) == 0:
            return

        print(f"Processing {att_scores_valid.size(0)} valid pixels for distribution logging")
        # Step 3: Distribution Construction
        # For each threshold, separate positive and negative based on known class confidence
        for thr_idx, thr in enumerate(self.thrs):
            # Step 3.16-22: MatchedScores threshold-based separation
            # Get maximum confidence across known classes for each pixel
            max_known_scores = assigned_scores_valid.max(dim=1)[0]  # [N_valid]

            # Separate positive (confident) and negative (uncertain) pixels
            positive_mask = max_known_scores >= thr  # High confidence in known classes
            negative_mask = ~positive_mask  # Low confidence in known classes

            positive_att_scores = att_scores_valid[positive_mask]  # [N_pos, C_att]
            negative_att_scores = att_scores_valid[negative_mask]  # [N_neg, C_att]

            print(f"Threshold {thr}: {positive_mask.sum()} positive, {negative_mask.sum()} negative pixels")

            # Step 4: Similarity Calculation & Distribution Update
            # For each attribute, calculate similarity distributions
            for att_i in range(num_att):
                if positive_att_scores.size(0) > 0:
                    # Positive distribution: similarity scores when model is confident about known classes
                    self.positive_distributions[thr_idx][att_i] += torch.histc(positive_att_scores[:, att_i], bins=int(1 / 0.0001), min=0, max=1)

                if negative_att_scores.size(0) > 0:
                    # Negative distribution: similarity scores when model is uncertain about known classes
                    self.negative_distributions[thr_idx][att_i] += torch.histc(negative_att_scores[:, att_i], bins=int(1 / 0.0001), min=0, max=1)

        print("Distribution logging completed")

    def forward(self, features, guidance_features, prompt=None, gt_cls=None):
        """
        Arguments:
            features: (B, C, HW)
            guidance_features: (B, C, H, W)
        """
        print("ow_cat_seg_head forward")
        img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat,
                              guidance_features,
                              prompt,
                              gt_cls,
                              self.att_embeddings,
                              self.fusion_att,
                              self.enable_ow_mode,  # Pass the flag
                              self.training,  # NEW: Pass training status
                              self.num_classes_train,  # NEW: Training classes
                              self.num_classes_test  # NEW: Evaluation classes
                              )