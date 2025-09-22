# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# Modified by Heeseong Shin from: https://github.com/dingjiansw101/ZegFormer/blob/main/mask_former/mask_former_model.py
import fvcore.nn.weight_init as weight_init
import torch

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .model import Aggregator
from cat_seg.third_party import clip
from cat_seg.third_party import imagenet_templates

import numpy as np
import open_clip
class OWCATSegPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,

        # ow-ovss new
        prev_intro_cls: int = None,
        cur_intro_cls: int = None,
        unknown_class_index: int = None,
        ignore_label: int = None,
        top_k: int = None,
        num_classes_train: int = None,
        num_classes_test: int = None,  # NEW: Total number of classes (150 ADE20K + 1 unknown)
        evaluation_mode: str = None,
    ):
        """
        Args:

        """
        super().__init__()

        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name,
                pretrained=pretrain,
                device=device,
                force_image_size=336,)

            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)

        self.prompt_ensemble_type = prompt_ensemble_type

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError

        self.prompt_templates = prompt_templates

        self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()

        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess

        # OW-OVD specific
        self.prev_intro_cls = prev_intro_cls
        self.cur_intro_cls = cur_intro_cls
        self.unknown_class_index = unknown_class_index
        self.ignore_label = ignore_label
        self.top_k = top_k
        self.num_classes_train = num_classes_train
        self.num_classes_test = num_classes_test

        self.evaluation_mode = evaluation_mode
        self.num_known_classes = prev_intro_cls + cur_intro_cls

        print(f"[INIT] Evaluation Mode : {self.evaluation_mode}")
        if self.evaluation_mode == "OVSS":
            print(f"OVSS : Known (0~{self.num_known_classes-1} | Unknown ({self.num_known_classes}~{self.unknown_class_index-1}")
        elif self.evaluation_mode == "OWSS":
            print(f"OWSS : Known (0~{self.num_known_classes-1} | Unknown ({self.unknown_class_index}, remapped from {self.num_known_classes}~{self.unknown_class_index-1})")

        transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads,
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
            )
        self.transformer = transformer

        self.tokens = None
        self.cache = None
        # Separate caches for train/test
        self.train_cache = {}
        self.test_cache = {}

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification:
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

        # OW-OVD parameters
        ret["prev_intro_cls"] = cfg.MODEL.SEM_SEG_HEAD.PREV_INTRO_CLS
        ret["cur_intro_cls"] = cfg.MODEL.SEM_SEG_HEAD.CUR_INTRO_CLS
        ret["unknown_class_index"] = cfg.MODEL.SEM_SEG_HEAD.UNKNOWN_ID
        ret["ignore_label"] = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        ret["top_k"] = cfg.MODEL.SEM_SEG_HEAD.TOP_K
        ret["num_classes_train"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES_TRAIN
        ret["num_classes_test"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES_TEST
        ret["evaluation_mode"] = cfg.MODEL.SEM_SEG_HEAD.EVALUATION_MODE

        return ret

    def forward(self,
                x,
                guidance_features,
                prompt=None,
                gt_cls=None,
                att_embeddings=None,
                fusion_att=False,
                enable_ow_mode=False,
                training=False,
                num_classes_train=None,  # NEW: Training classes
                num_classes_test=None,  # NEW: Evaluation classes
    ):

        print(f"[DEBUG] Forward - mode: {self.evaluation_mode}, training: {self.training}")
        vis = [guidance_features[k] for k in guidance_features.keys()][::-1]

        # =======================================================================================================
        # Mode-specific logic
        if self.evaluation_mode == "OVSS":
            return self.forward_ovss(x, vis, prompt, gt_cls)
        elif self.evaluation_mode == "OWSS":
            return self.forward_owss(x, vis, prompt, gt_cls, att_embeddings, fusion_att, enable_ow_mode, training)
        else:
            raise ValueError(f"Unknown evaluation mode: {self.evaluation_mode}")
        # =======================================================================================================

        # text = self.class_texts if self.training else self.test_class_texts[:self.unknown_class_index]
        # text = [text[c] for c in gt_cls] if gt_cls is not None else text
        # text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
        # text = text.expand(x.shape[0], -1, -1, -1)
        #
        # # Get known class predictions
        # ovss_logits = self.transformer(x, text, vis)
        # if not self.training and enable_ow_mode and att_embeddings is not None: # OV + OW Mode
        #     print("forward_evaluation_ow")
        #     result = self.forward_evaluation_ow(x, vis, ovss_logits, att_embeddings, fusion_att)
        #     return result
        #
        # elif not self.training and not enable_ow_mode: # OV Mode
        #     print("forward_evaluation_baseline")
        #     return self.forward_evaluation_baseline(ovss_logits)
        # else:
        #     return ovss_logits

    def forward_ovss(self, x, vis, prompt=None, gt_cls=None):
        """
        🔧 OVSS Mode: Open-Vocabulary Semantic Segmentation
        - Use all 150 classes (0~149) with text embeddings
        - Output: [B, 150, H, W]
        """
        print("[OVSS] Open-Vocabulary mode")

        # Use all test classes (0~149)
        if self.training:
            text = self.class_texts  # COCO classes for training
        else:
            text = self.test_class_texts  # All 150 ADE20K classes

        print(f"  Using {len(text)} classes for text embedding")

        text = [text[c] for c in gt_cls] if gt_cls is not None else text
        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
        text = text.expand(x.shape[0], -1, -1, -1)

        # Direct prediction for all classes
        logits = self.transformer(x, text, vis)  # [B, 150, H, W]

        print(f"  OVSS output shape: {logits.shape}")
        return logits

    def forward_owss(self, x, vis, prompt=None, gt_cls=None, att_embeddings=None,
                     fusion_att=False, enable_ow_mode=False, training=False):
        """
        🔧 OWSS Mode: Open-World Semantic Segmentation
        - Use only known classes (0~74) with text embeddings
        - Use attributes to predict unknown class (150)
        - Remap GT classes 75~149 → 150 during evaluation
        - Output: [B, 76, H, W] (75 known + 1 unknown)
        """
        print("🔧 [OWSS] Open-World mode")

        # Use only known classes (0~74) for text embedding
        if self.training:
            text = self.class_texts  # COCO classes for training
        else:
            text = self.test_class_texts[:self.num_known_classes]  # Only first 75 classes

        print(f"  Using {len(text)} known classes for text embedding")

        text = [text[c] for c in gt_cls] if gt_cls is not None else text
        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
        text = text.expand(x.shape[0], -1, -1, -1)

        # Get known class predictions (0~74)
        known_logits = self.transformer(x, text, vis)  # [B, 75, H, W]

        if self.training:
            print(f"  Training: returning known logits {known_logits.shape}")
            return known_logits
        elif enable_ow_mode and att_embeddings is not None:
            print("  OW inference: adding attribute-based unknown detection")

            # Get attribute predictions
            if fusion_att:
                att_text = att_embeddings.unsqueeze(0).unsqueeze(2).expand(x.shape[0], -1, -1, -1)
            else:
                att_text = att_embeddings.unsqueeze(0).unsqueeze(2).expand(x.shape[0], -1, -1, -1)

            att_logits = self.transformer(x, att_text, vis)
            unknown_logit = self.predict_unknown_owss(known_logits, att_logits)

            # Combine: [B, 75+1, H, W] = [B, 76, H, W]
            final_output = torch.cat([known_logits, unknown_logit], dim=1)
            print(f"  OWSS output shape: {final_output.shape}")
            return final_output

    def predict_unknown_owss(self, known_logits, att_logits):
        """
        🔧 OWSS: Predict unknown class using attributes
        """
        B, C_known, H, W = known_logits.shape

        known_sigmoid = known_logits.sigmoid()
        unknown_sigmoid = att_logits.sigmoid()

        # Calculate uncertainty from known classes
        known_probs_clamped = torch.clamp(known_sigmoid, 1e-6, 1 - 1e-6)
        entropy = (-known_probs_clamped * torch.log(known_probs_clamped) -
                   (1 - known_probs_clamped) * torch.log(1 - known_probs_clamped))
        uncertainty = entropy.mean(dim=1, keepdim=True)

        # Top-k attribute combination
        top_k_scores, _ = unknown_sigmoid.topk(self.top_k, dim=1)
        top_k_weights = F.softmax(top_k_scores, dim=1)
        weighted_average = (top_k_scores * top_k_weights).sum(dim=1, keepdim=True)

        # Suppress unknown when known classes are confident
        known_max = known_sigmoid.max(dim=1, keepdim=True)[0]
        unknown_score = (weighted_average + uncertainty) / 2 * (1 - known_max)

        # Convert to logit
        unknown_prob = unknown_score.clamp(1e-6, 1 - 1e-6)
        unknown_logit = torch.log(unknown_prob) - torch.log(1 - unknown_prob)

        return unknown_logit

    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        """Generate text embeddings with proper caching"""
        is_training = self.training
        cache_key = f"{'train' if is_training else 'test'}_{len(classnames)}_{len(templates)}_{self.evaluation_mode}"

        cache = self.train_cache if is_training else self.test_cache
        cached_result = cache.get(cache_key)

        if cached_result is not None and not self.training:
            print(f"🔧 [DEBUG] Using cache for {len(classnames)} classes ({self.evaluation_mode} mode)")
            return cached_result

        print(f"🔧 [DEBUG] Generating embeddings for {len(classnames)} classes ({self.evaluation_mode} mode)")

        tokens = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = [template.format(classname_splits[0]) for template in templates]
            else:
                texts = [template.format(classname) for template in templates]
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else:
                texts = clip.tokenize(texts).cuda()
            tokens.append(texts)

        tokens = torch.stack(tokens, dim=0).squeeze(1)
        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.unsqueeze(1)

        if not self.training:
            cache[cache_key] = class_embeddings
            print(f"🔧 [DEBUG] Cached embeddings as {cache_key}")

        return class_embeddings

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else:
                texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    # def forward_evaluation_ow(self,
    #                           x,
    #                           vis,
    #                           ovss_logits,
    #                           att_embeddings,
    #                           fusion_att):
    #
    #     print("forward_evaluation_ow")
    #     B, C_ovss, H, W = ovss_logits.shape
    #
    #     # Process attribute embeddings
    #     if fusion_att:
    #         num_att = att_embeddings.shape[0]
    #         #att_text = att_embeddings.unsqueeze(0).unsqueeze(2).repeat(B, 1, 1, 1)
    #         att_text = att_embeddings.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)
    #     else:
    #         #att_text = att_embeddings.unsqueeze(0).unsqueeze(2).repeat(B, 1, 1, 1)
    #         att_text = att_embeddings.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)
    #         # print(f"att_text.shape after expanding : {att_text.shape}")
    #
    #     # Get attribute predictions
    #     att_logits = self.transformer(x, att_text, vis)
    #     final_output = self.predict_unknown(ovss_logits, att_logits)
    #
    #     return final_output
    #
    # def forward_evaluation_baseline(self, ovss_logits):
    #     """
    #     Baseline mode evaluation
    #     """
    #     # Fill remaining classes with very low scores
    #     unknown_padding = torch.full((B, 1, H, W), -100.0, device=ovss_logits.device, dtype=ovss_logits.dtype)
    #     baseline_output = torch.cat([ovss_logits, unknown_padding], dim=1)  # [B, 151, H, W]
    #
    #     return baseline_output
    #
    # @torch.no_grad()
    # def class_embeddings(self, classnames, templates, clip_model):
    #     zeroshot_weights = []
    #     for classname in classnames:
    #         if ', ' in classname:
    #             classname_splits = classname.split(', ')
    #             texts = []
    #             for template in templates:
    #                 for cls_split in classname_splits:
    #                     texts.append(template.format(cls_split))
    #         else:
    #             texts = [template.format(classname) for template in templates]  # format with class
    #         if self.tokenizer is not None:
    #             texts = self.tokenizer(texts).cuda()
    #         else:
    #             texts = clip.tokenize(texts).cuda()
    #         class_embeddings = clip_model.encode_text(texts)
    #         class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    #         if len(templates) != class_embeddings.shape[0]:
    #             class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
    #             class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    #         class_embedding = class_embeddings
    #         zeroshot_weights.append(class_embedding)
    #     zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    #     return zeroshot_weights
    #
    # def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
    #     # 🔧 해결책 1: 캐시 키를 클래스 수로 구분
    #     cache_key = f"cache_{len(classnames)}_{len(templates)}"
    #     cached_result = getattr(self, cache_key, None)
    #
    #     if cached_result is not None and not self.training:
    #         print(f"[DEBUG] Using cache for {len(classnames)} classes")
    #         return cached_result
    #
    #     print(f"[DEBUG] Generating new embeddings for {len(classnames)} classes")
    #
    #     tokens = []
    #     for classname in classnames:
    #         if ', ' in classname:
    #             classname_splits = classname.split(', ')
    #             texts = [template.format(classname_splits[0]) for template in templates]
    #         else:
    #             texts = [template.format(classname) for template in templates]
    #         if self.tokenizer is not None:
    #             texts = self.tokenizer(texts).cuda()
    #         else:
    #             texts = clip.tokenize(texts).cuda()
    #         tokens.append(texts)
    #
    #     tokens = torch.stack(tokens, dim=0).squeeze(1)
    #
    #     if prompt is None:
    #         self.tokens = tokens
    #
    #     class_embeddings = clip_model.encode_text(tokens, prompt)
    #     class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    #
    #     class_embeddings = class_embeddings.unsqueeze(1)
    #
    #     if not self.training:
    #         setattr(self, cache_key, class_embeddings)
    #         print(f"[DEBUG] Cached embeddings as {cache_key} with shape {class_embeddings.shape}")
    #
    #     return class_embeddings

    def calculate_uncertainty(self, ovss_logits):
        """Calculate uncertainty for known classes."""
        ovss_logits = torch.clamp(ovss_logits, 1e-6, 1 - 1e-6)
        entropy = (-ovss_logits * torch.log(ovss_logits) - (1 - ovss_logits) * torch.log(1 - ovss_logits)).mean(
            dim=-1, keepdim=True)
        return entropy

    def compute_weighted_top_k_attributes(self, adjusted_scores, k=10):
        """Compute weighted average of top-k attributes."""
        top_k_scores, top_k_indices = adjusted_scores.topk(k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        weighted_average = torch.sum(top_k_scores * top_k_weights, dim=-1, keepdim=True)
        return weighted_average


    # def predict_unknown(self, ovss_logits, att_logits):
    #     B, C_known, H, W = ovss_logits.shape
    #
    #     known_sigmoid = ovss_logits.sigmoid()
    #     unknown_sigmoid = att_logits.sigmoid()
    #     ret["evaluation_mode"] = cfg.MODEL.SEM_SEG_HEAD.EVALUATION_MODE  # 🔧 NEW
    #
    #     known_probs_clamped = torch.clamp(known_sigmoid, 1e-6, 1 - 1e-6)
    #     entropy = (-known_probs_clamped * torch.log(known_probs_clamped) -
    #                (1 - known_probs_clamped) * torch.log(1 - known_probs_clamped))
    #     uncertainty = entropy.mean(dim=1, keepdim=True)
    #
    #     top_k_scores, _ = unknown_sigmoid.topk(self.top_k, dim=1)
    #     top_k_weights = F.softmax(top_k_scores, dim=1)
    #     weighted_average = (top_k_scores * top_k_weights).sum(dim=1, keepdim=True)
    #
    #     known_max = known_sigmoid.max(dim=1, keepdim=True)[0]
    #     unknown_score = (weighted_average + uncertainty) / 2 * (1 - known_max)
    #
    #     unknown_prob = unknown_score.clamp(1e-6, 1 - 1e-6)
    #     unknown_logit = torch.log(unknown_prob) - torch.log(1 - unknown_prob)  # logit(p) = log(p/(1-p))
    #
    #     ret_logits = torch.cat([ovss_logits, unknown_logit], dim=1)
    #     return ret_logits