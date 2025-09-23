# python generate_clip_embeddings.py \
#   --classnames datasets/ade150.json \
#   --attributes data/ADEChallengeData2016/unique_attributes_ade150_claude.json \
#   --model openai/clip-vit-base-patch32 \
#   --out_dir data/ADEChallengeData2016/


#!/usr/bin/env python3
import argparse, json, os, sys
from typing import List, Dict, Tuple
import torch
import numpy as np

try:
    from transformers import CLIPModel, CLIPTokenizer
except Exception as e:
    print("Please install dependencies: pip install transformers torch", file=sys.stderr)
    raise

# -----------------------------
# Prompt templates
# -----------------------------
CLASS_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a cropped photo of the {}.",
    "a bright photo of a {}.",
    "a dim photo of the {}.",
    "a low-resolution photo of a {}.",
    "a clean photo of a {}.",
]

ATTR_TEMPLATES = {
    "Shape": [
        "Shape is {}.",
    ],
    "Color": [
        "Color is {}.",
    ],
    "Texture": [
        "Texture is {}.",
    ],
    "Size": [
        "Size is {}.",
    ],
    "Context": [
        "Context is {}.",
    ],
    "Features": [
        "Features is {}.",
    ],
    "Appearance": [
        "Appearance is {}.",
    ],
    "Behavior": [
        "Behavior is {}.",
    ],
    "Environment": [
        "Environment is {}.",
    ],
    "Material": [
        "Material is {}.",
    ],
}

# -----------------------------
# CLIP encode helpers
# -----------------------------
@torch.inference_mode()
def clip_encode_text(texts: List[str], model, tokenizer, device="cuda", batch_size=256) -> torch.Tensor:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        feats = model.get_text_features(**inputs)  # [B, D]
        feats = torch.nn.functional.normalize(feats, dim=-1)
        embs.append(feats.cpu())
    return torch.cat(embs, dim=0)

def build_class_prompts(classes: List[str]) -> List[str]:
    prompts = []
    for c in classes:
        for tmpl in CLASS_TEMPLATES:
            prompts.append(tmpl.format(c))
    return prompts

def build_attr_texts(unique_attrs: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    texts, cats = [], []
    for cat, vals in unique_attrs.items():
        # fallback if unseen category
        tmpls = ATTR_TEMPLATES.get(cat, ["{}"])
        for v in vals:
            # Use all templates for diversity
            for t in tmpls:
                texts.append(t.format(v))
                cats.append(cat)
    return texts, cats

def average_by_chunks(emb: torch.Tensor, chunk: int) -> torch.Tensor:
    """Average every `chunk` rows (e.g., templates per class/attribute)."""
    assert emb.shape[0] % chunk == 0, f"rows {emb.shape[0]} not divisible by chunk {chunk}"
    return emb.view(-1, chunk, emb.shape[1]).mean(dim=1)

def save_cls_embeddings(path: str, cls_emb: torch.Tensor):
    arr = cls_emb.to(torch.float16).numpy()
    np.save(path, arr)

def save_att_embeddings(path: str, att_emb: torch.Tensor, att_text: List[str], categories: List[str]):
    obj = {
        "att_embedding": att_emb.to(torch.float32),   # [N_attr, D]
        "att_text": att_text,                         # list[str] after averaging (1 per attribute concept)
        "categories": categories,                     # list[str] same length as att_text
        "num_attributes": len(att_text),
    }
    torch.save(obj, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classnames", type=str, required=True, help="Path to ADE150 classes JSON (list[str])")
    ap.add_argument("--attributes", type=str, required=True, help="Path to unique_attributes.json (dict[str, list[str]])")
    ap.add_argument("--model", type=str, default="openai/clip-vit-base-patch32", help="HF model id or local path")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default=".")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.classnames, "r") as f:
        classes = json.load(f)
    with open(args.attributes, "r") as f:
        unique_attrs = json.load(f)

    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(args.device)
    model.eval()

    # ---- Class embeddings ----
    class_prompts = build_class_prompts(classes)  # C * T
    class_emb_all = clip_encode_text(class_prompts, model, tokenizer, device=args.device)
    class_emb = average_by_chunks(class_emb_all, chunk=len(CLASS_TEMPLATES))  # [C, D]
    cls_out = os.path.join(args.out_dir, "cls_embeddings_original_sequence.npy")
    save_cls_embeddings(cls_out, class_emb)
    print(f"Saved class embeddings to {cls_out} with shape {tuple(class_emb.shape)}")

    # ---- Attribute embeddings ----
    attr_texts_all, cats_all = build_attr_texts(unique_attrs)  # N_attr * per-cat templates
    attr_emb_all = clip_encode_text(attr_texts_all, model, tokenizer, device=args.device)
    # average by number of templates per category
    # compute chunk size per attribute concept
    # we used all templates in ATTR_TEMPLATES[cat], so count per-cat
    cat_template_counts = {k: len(v) for k,v in ATTR_TEMPLATES.items()}
    # regroup by attribute concept (same phrase repeated with different templates)
    grouped_texts = []
    grouped_cats = []
    grouped_embs = []

    i = 0
    idx = 0
    # Flatten attributes in deterministic order
    for cat, vals in unique_attrs.items():
        tmpls = ATTR_TEMPLATES.get(cat, ["{}"])
        t = len(tmpls)
        for v in vals:
            # for each attribute value: take next t rows
            emb_chunk = attr_emb_all[idx:idx+t]
            idx += t
            grouped_embs.append(emb_chunk.mean(dim=0, keepdim=True))
            grouped_texts.append(v)
            grouped_cats.append(cat)

    att_emb = torch.cat(grouped_embs, dim=0)  # [N_attr, D]

    att_out = os.path.join(args.out_dir, "att_embeddings_coco_claude.pth")
    save_att_embeddings(att_out, att_emb, grouped_texts, grouped_cats)
    print(f"Saved attribute embeddings to {att_out} with shape {tuple(att_emb.shape)}")
    print("Done.")

if __name__ == "__main__":
    main()
