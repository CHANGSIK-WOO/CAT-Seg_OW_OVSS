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

        # ow-ovdd new
        unknown_cls: int = 75,
        top_k: int = 10,
        num_classes_train: int = 170,
        num_classes_test: int = 151,  # NEW: Total number of classes (150 ADE20K + 1 unknown)

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
        self.unknown_cls = unknown_cls
        self.top_k = top_k
        self.num_classes_train = num_classes_train
        self.num_classes_test = num_classes_test

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

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
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
        ret["unknown_cls"] = cfg.MODEL.SEM_SEG_HEAD.UNKNOWN_CLS
        ret["top_k"] = cfg.MODEL.SEM_SEG_HEAD.TOP_K
        ret["num_classes_train"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES_TRAIN
        ret["num_classes_test"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES_TEST

        return ret

    # def forward(self, x, vis_guidance, prompt=None, gt_cls=None, att_embeddings=None, fusion_att=False):
    #     vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
    #     text = self.class_texts if self.training else self.test_class_texts
    #     text = [text[c] for c in gt_cls] if gt_cls is not None else text
    #     text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
    #
    #     text = text.repeat(x.shape[0], 1, 1, 1)
    #
    #     # Get known class predictions
    #     known_logits = self.transformer(x, text, vis)
    #
    #     # Handle attribute embeddings for unknown prediction
    #     if att_embeddings is not None and not self.training:
    #         # Process attribute embeddings
    #         if fusion_att:
    #             # Fusion mode: attributes are already in text features
    #             num_att = att_embeddings.shape[0]
    #             att_text = text[:, -num_att:, :, :]
    #             text = text[:, :-num_att, :, :]
    #         else:
    #             # Separate mode: encode attributes separately
    #             att_text = self.get_text_embeds(
    #                 ["attribute"] * att_embeddings.shape[0],  # Dummy text for attributes
    #                 self.prompt_templates,
    #                 self.clip_model,
    #                 None
    #             )
    #             # Replace with actual attribute embeddings
    #             att_text = att_embeddings[None].repeat(x.shape[0], 1, 1, 1)
    #
    #         # Get attribute predictions
    #         att_logits = self.transformer(x, att_text, vis)
    #
    #         # Combine known and unknown predictions
    #         combined_logits = self.predict_unknown(known_logits, att_logits)
    #
    #         return combined_logits
    #
    #     return known_logits

    def forward(self, x, vis_guidance, prompt=None, gt_cls=None, att_embeddings=None,
                fusion_att=False, enable_ow_mode=True, is_training=True,
                num_classes_train=171, num_classes_test=151):
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        text = self.class_texts if self.training else self.test_class_texts[:self.unknown_cls]

        text = [text[c] for c in gt_cls] if gt_cls is not None else text
        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)

        text = text.repeat(x.shape[0], 1, 1, 1)

        # Get known class predictions
        logits = self.transformer(x, text, vis)

        # 🔧 OW mode 처리
        if not self.training and enable_ow_mode and att_embeddings is not None:
            # Test mode with OW: 151개 클래스 출력 (75 known + 75 padding + 1 unknown)
            return self.forward_evaluation_ow(x, vis, logits, att_embeddings, fusion_att)

        elif not self.training and not enable_ow_mode:
            # Test mode without OW: 151개 클래스 출력 (baseline)
            return self.forward_evaluation_baseline(logits)

        else:
            # Training mode: 171개 클래스 출력
            return logits

    def forward_evaluation_ow(self, x, vis, known_logits, att_embeddings, fusion_att):
        """
        🔧 OW mode evaluation: 151개 클래스 출력
        """
        B, C_known, H, W = known_logits.shape  # [B, 75, H, W]

        # Process attribute embeddings
        if fusion_att:
            # Fusion mode: attributes are integrated
            num_att = att_embeddings.shape[0]
            att_text = att_embeddings.unsqueeze(0).unsqueeze(2).repeat(B, 1, 1, 1)
        else:
            # Separate mode: encode attributes separately
            att_text = att_embeddings.unsqueeze(0).unsqueeze(2).repeat(B, 1, 1, 1)

        # Get attribute predictions
        att_logits = self.transformer(x, att_text, vis)  # [B, N_att, H, W]

        # Predict unknown class score
        unknown_score = self.predict_unknown(known_logits, att_logits)

        # Construct final output: [B, 151, H, W]
        # 0-74: known classes, 75-149: padding with low scores, 150: unknown
        padding = torch.full((B, 75, H, W), -100.0, device=known_logits.device, dtype=known_logits.dtype)
        final_output = torch.cat([known_logits, padding, unknown_score], dim=1)  # [B, 151, H, W]

        return final_output

    def forward_evaluation_baseline(self, known_logits):
        """
        🔧 Baseline mode evaluation: 151개 클래스 출력 (OW 없음)
        """
        B, C_known, H, W = known_logits.shape  # [B, 75, H, W]

        # Fill remaining classes with very low scores
        padding = torch.full((B, 75, H, W), -100.0, device=known_logits.device, dtype=known_logits.dtype)
        unknown_padding = torch.full((B, 1, H, W), -100.0, device=known_logits.device, dtype=known_logits.dtype)
        baseline_output = torch.cat([known_logits, padding, unknown_padding], dim=1)  # [B, 151, H, W]

        return baseline_output

    # def forward_evaluation(self, x, vis, prompt, gt_cls, att_embeddings, enable_ow_mode, num_classes_test):
    #     """
    #     Evaluation mode: ADE20K-style output with 151 classes
    #     """
    #     # Use only known classes (0-74) from ADE20K
    #     known_text = self.test_class_texts[:self.unknown_cls]  # ADE20K 0-74번만
    #     text = [known_text[c] for c in gt_cls] if gt_cls is not None else known_text
    #     text_embeddings = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
    #     text_embeddings = text_embeddings.repeat(x.shape[0], 1, 1, 1)
    #
    #     # Get known class predictions (0-74)
    #     known_logits = self.transformer(x, text_embeddings, vis)  # [B, 75, H, W]
    #     B, _, H, W = known_logits.shape
    #
    #     if enable_ow_mode and att_embeddings is not None:
    #         # OW Mode: Use attributes to predict unknown class
    #         att_text = att_embeddings.unsqueeze(0).unsqueeze(2).repeat(B, 1, 1, 1)
    #         att_logits = self.transformer(x, att_text, vis)  # [B, 1711, H, W]
    #         combined_logits = self.predict_unknown_for_evaluation(known_logits, att_logits)
    #         return combined_logits  # [B, 151, H, W]
    #     else:
    #         # Baseline Mode: Fill remaining classes with low scores
    #         unknown_padding = torch.full((B, 75, H, W), -100.0, device=known_logits.device, dtype=known_logits.dtype)
    #         final_unknown = torch.full((B, 1, H, W), -100.0, device=known_logits.device, dtype=known_logits.dtype)
    #         baseline_logits = torch.cat([known_logits, unknown_padding, final_unknown], dim=1)  # [B, 151, H, W]
    #         return baseline_logits


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
                texts = [template.format(classname) for template in templates]  # format with class
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
    
    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        if self.cache is not None and not self.training:
            return self.cache
        
        if self.tokens is None or prompt is not None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in templates]
                else:
                    texts = [template.format(classname) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(texts).cuda()
                else: 
                    texts = clip.tokenize(texts).cuda()
                tokens.append(texts)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            if prompt is None:
                self.tokens = tokens
        elif self.tokens is not None and prompt is None:
            tokens = self.tokens

        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
        
        class_embeddings = class_embeddings.unsqueeze(1)
        
        if not self.training:
            self.cache = class_embeddings
            
        return class_embeddings

    def calculate_uncertainty(self, known_logits):
        """Calculate uncertainty for known classes."""
        known_logits = torch.clamp(known_logits, 1e-6, 1 - 1e-6)
        entropy = (-known_logits * torch.log(known_logits) - (1 - known_logits) * torch.log(1 - known_logits)).mean(
            dim=-1, keepdim=True)
        return entropy

    def compute_weighted_top_k_attributes(self, adjusted_scores, k=10):
        """Compute weighted average of top-k attributes."""
        top_k_scores, top_k_indices = adjusted_scores.topk(k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        weighted_average = torch.sum(top_k_scores * top_k_weights, dim=-1, keepdim=True)
        return weighted_average

    def predict_unknown(self, known_logits, unknown_logits):
        """
        Predict unknown class scores for OW evaluation.

        Args:
            known_logits: [B, 75, H, W] - predictions for known classes (0-74)
            unknown_logits: [B, 1711, H, W] - attribute predictions

        Returns:
            combined_logits: [B, 151, H, W] - 75 known + 75 unknown padding + 1 unknown class
        """
        B, C_known, H, W = known_logits.shape

        # Apply sigmoid to get probabilities
        known_probs = known_logits.sigmoid()
        unknown_probs = unknown_logits.sigmoid()

        # Calculate uncertainty for known classes
        uncertainty = self.calculate_uncertainty(known_probs.permute(0, 2, 3, 1))  # B, H, W, 1

        # Compute weighted top-k attributes
        unknown_probs_perm = unknown_probs.permute(0, 2, 3, 1)  # B, H, W, C_att
        top_k_att_score = self.compute_weighted_top_k_attributes(unknown_probs_perm, k=self.top_k)  # B, H, W, 1

        # Fusion with uncertainty
        known_max = known_probs.max(dim=1, keepdim=True)[0]  # B, 1, H, W
        unknown_score = (top_k_att_score.permute(0, 3, 1, 2) + uncertainty.permute(0, 3, 1, 2)) / 2 * (1 - known_max)

        # # Concatenate known and unknown predictions
        # combined_logits = torch.cat([known_logits, unknown_score], dim=1)  # B, C_known+1, H, W

        return unknown_score