"""
clip_to_qwen_holistic.py
--------------------------------------------
A single, holistic script that lets you:

1) Turn an image (or first page of a PDF) into CLIP features
2) (Option A) Inject those features into Qwen3 (text-only) as SOFT PROMPT EMBEDDINGS
3) (Option B) Use a true vision model (Qwen-VL) and pass the image directly
4) (Optional) Retrieve zero-shot TAGS and nearest CAPTIONS via CLIP-only text retrieval

Why two options?
- Qwen3 is text-only. You can *condition* it with soft prompt vectors, but it is **not true vision**.
- Qwen-VL has a visual encoder. Use it if you need actual image understanding.

Prereqs (suggested):
  pip install torch pillow pdf2image transformers openai numpy
  # and poppler installed for pdf2image (if you plan to read PDFs)

Run a vLLM OpenAI-compatible server in *one* of two ways:

  [Soft-embeds path (Qwen3 text-only)]
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --enable-prompt-embeds \
    --enforce-eager --gpu-memory-utilization 0.8

  [Vision path (Qwen-VL)]
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --enforce-eager --gpu-memory-utilization 0.8

Then run this script, for example:

  # SOFT-PREFIX into Qwen3 (image -> CLIP -> projector -> soft tokens -> Qwen3)
  python clip_to_qwen_holistic.py \
    --input ./Konigsberg2.pdf \
    --mode soft \
    --soft-tokens 16 \
    --prompt "Describe the likely figure in 2–3 sentences focusing on structure."

  # TRUE VL path (Qwen-VL: pass image directly)
  python clip_to_qwen_holistic.py \
    --input ./Konigsberg2.pdf \
    --mode vl \
    --prompt "Describe the figure in 2–3 sentences focusing on structure."

  # Just test CLIP tag & caption retrieval (no LLM call)
  python clip_to_qwen_holistic.py --input ./Konigsberg2.pdf --mode retrieval --topk 5

Environment variables for the HTTP client:
  OPENAI_BASE_URL (default: http://localhost:8000/v1)
  OPENAI_API_KEY  (default: EMPTY)

Notes:
- The soft-prefix projector is random unless you load trained weights.
- For better results, prefix-tune the projector on (image, caption) pairs.
"""

import os
import base64
import argparse
import numpy as np
from dataclasses import dataclass

# Ensure vLLM worker spawn safety if this file later evolves to import vllm directly
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
import torch.nn as nn
from PIL import Image
from typing import List, Tuple

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMG = True
except Exception:
    HAS_PDF2IMG = False

from transformers import CLIPProcessor, CLIPModel

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # allow retrieval-only mode without openai client

# -------------------------------
# Defaults / small banks
# -------------------------------
CANDIDATE_TAGS = [
    "diagram", "graph", "map", "network", "bridges", "math figure",
    "printed text", "table", "chart", "equation", "flowchart",
    "scatter plot", "bar chart", "ROC curve", "heatmap", "topology",
]

TAG_TEMPLATES = [
    "a diagram of {}",
    "a figure showing {}",
    "a technical illustration of {}",
    "contains {}",
    "a chart about {}",
]

DEFAULT_CAPTIONS = [
    "A graph comparing two variables over time.",
    "A schematic diagram of bridges connecting landmasses.",
    "A flowchart showing steps in an algorithm.",
    "A table summarizing experimental results.",
    "A map illustrating routes between locations.",
]

# -------------------------------
# Utility: PDF -> Image
# -------------------------------
def pdf_first_page_to_image(pdf_path: str, dpi: int = 200, save_jpg_path: str | None = None) -> Image.Image:
    if not HAS_PDF2IMG:
        raise RuntimeError("pdf2image/poppler not available. Install pdf2image and poppler or pass a .jpg/.png.")
    pages = convert_from_path(pdf_path, dpi=dpi)
    if not pages:
        raise RuntimeError(f"No pages found in PDF: {pdf_path}")
    img = pages[0].convert("RGB")
    if save_jpg_path:
        img.save(save_jpg_path, "JPEG")
    return img

# -------------------------------
# CLIP: load & embeddings
# -------------------------------
@dataclass
class ClipBundle:
    model: CLIPModel
    proc: CLIPProcessor


def load_clip(model_id: str = "openai/clip-vit-base-patch32") -> ClipBundle:
    model = CLIPModel.from_pretrained(model_id)
    proc = CLIPProcessor.from_pretrained(model_id)
    return ClipBundle(model, proc)


@torch.no_grad()
def clip_image_embed(cb: ClipBundle, image: Image.Image) -> torch.Tensor:
    inp = cb.proc(images=image, return_tensors="pt")
    feats = cb.model.get_image_features(**inp)   # [1, D]
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def clip_text_embed(cb: ClipBundle, texts: List[str]) -> torch.Tensor:
    t = cb.proc(text=texts, padding=True, return_tensors="pt")
    f = cb.model.get_text_features(**t)
    f = f / f.norm(dim=-1, keepdim=True)
    return f


# -------------------------------
# CLIP: Tag ranking with templates
# -------------------------------
@torch.no_grad()
def rank_tags_with_templates(cb: ClipBundle, image_feats: torch.Tensor, tags: List[str], templates: List[str], topk: int = 5) -> List[str]:
    all_txt = [tpl.format(tag) for tag in tags for tpl in templates]
    txt = clip_text_embed(cb, all_txt)                                  # [T*|tags|, D]
    sims = (image_feats @ txt.T).squeeze(0)                              # [T*|tags|]
    T = len(templates)
    sims_by_tag = sims.view(len(tags), T).max(dim=1).values              # [|tags|]
    top_idx = torch.topk(sims_by_tag, k=min(topk, len(tags))).indices.tolist()
    return [tags[i] for i in top_idx]
    

# -------------------------------
# CLIP: Caption retrieval (nearest text)
# -------------------------------
@torch.no_grad()
def retrieve_captions(cb: ClipBundle, image_feats: torch.Tensor, candidate_sentences: List[str], topk: int = 5) -> List[str]:
    if not candidate_sentences:
        return []
    # embed in manageable chunks
    batch = 256
    all_scores = []
    for i in range(0, len(candidate_sentences), batch):
        chunk = candidate_sentences[i:i+batch]
        f = clip_text_embed(cb, chunk)                                   # [B, D]
        scores = (image_feats @ f.T).squeeze(0)                          # [B]
        all_scores.append(scores)
    sims = torch.cat(all_scores)
    top_idx = sims.topk(k=min(topk, sims.numel())).indices.tolist()
    return [candidate_sentences[i] for i in top_idx]


# -------------------------------
# Projector: CLIP -> Qwen3 hidden space (4096)
# -------------------------------
HIDDEN = 4096

class LinearProjector(nn.Module):
    def __init__(self, d_in: int, d_out: int = HIDDEN):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
    def forward(self, x):  # [1, d_in]
        return self.proj(x)  # [1, d_out]


class MLPProjector(nn.Module):
    def __init__(self, d_in: int, d_out: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 2048),
            nn.Tanh(),
            nn.Linear(2048, d_out),
        )
    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def make_soft_tokens(img_feat: torch.Tensor, projector: nn.Module, T: int) -> torch.Tensor:
    soft = projector(img_feat)                  # [1, 4096]
    soft = soft.repeat(T, 1).contiguous()       # [T, 4096]
    return soft


# -------------------------------
# OpenAI-compatible client helpers (vLLM server)
# -------------------------------
def get_openai_client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed. pip install openai")
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    return OpenAI(base_url=base_url, api_key=api_key)


def to_prompt_embeds_payload(tensor: torch.Tensor) -> dict:
    arr = tensor.detach().cpu().numpy().astype(np.float32)  # [T, 4096]
    payload = base64.b64encode(arr.tobytes()).decode("utf-8")
    return {"data": payload, "dtype": "float32", "shape": list(arr.shape)}


def call_qwen3_with_softprefix(soft_tokens: torch.Tensor, text_prompt: str, max_tokens: int = 128, temperature: float = 0.2) -> str:
    client = get_openai_client()
    resp = client.completions.create(
        model="Qwen/Qwen3-8B",
        prompt=text_prompt,  # optional; recommended
        prompt_embeds=to_prompt_embeds_payload(soft_tokens),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].text


def call_qwenvl_with_image(image_path: str, text_prompt: str, max_tokens: int = 128, temperature: float = 0.2) -> str:
    client = get_openai_client()
    abs_path = os.path.abspath(image_path)
    # vLLM OpenAI server accepts image inputs as file:// URLs in messages content
    msg = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": text_prompt or "Describe the image."},
            {"type": "input_image", "image_url": f"file://{abs_path}"},
        ],
    }]
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        messages=msg,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


# -------------------------------
# Orchestration
# -------------------------------
def load_image_auto(path: str, dpi: int = 200, save_preview: bool = True) -> Tuple[Image.Image, str | None]:
    ext = os.path.splitext(path)[1].lower()
    jpg_path = None
    if ext == ".pdf":
        jpg_path = os.path.splitext(path)[0] + "_page1.jpg"
        img = pdf_first_page_to_image(path, dpi=dpi, save_jpg_path=jpg_path)
        return img, jpg_path
    else:
        return Image.open(path).convert("RGB"), (path if ext in [".jpg", ".jpeg", ".png"] else None)


def main():
    p = argparse.ArgumentParser(description="CLIP → (soft) Qwen3 or Qwen-VL holistic demo")
    p.add_argument("--input", required=True, help="Path to image or PDF")
    p.add_argument("--mode", choices=["soft", "vl", "retrieval"], default="soft")
    p.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    p.add_argument("--prompt", type=str, default="",
                   help="Text prompt (prepended after soft tokens, or used for VL)")
    p.add_argument("--soft-tokens", type=int, default=16, help="# of soft tokens for Qwen3")
    p.add_argument("--projector", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--clip-id", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--caption-corpus", type=str, default="", help="Optional .txt file with one caption per line")
    p.add_argument("--topk", type=int, default=5, help="Top-K for tags/captions in retrieval mode")
    args = p.parse_args()

    # 1) Load image (or convert PDF page1)
    img, preview = load_image_auto(args.input, dpi=args.dpi)
    if preview:
        print(f"[info] Saved/using preview image: {preview}")

    # 2) Load CLIP & make image embedding
    cb = load_clip(args.clip_id)
    img_feat = clip_image_embed(cb, img)  # [1, D]
    D = int(img_feat.shape[-1])
    print(f"[info] CLIP image features shape: {tuple(img_feat.shape)}")

    # 3) Retrieval helpers (always available)
    # Tags
    top_tags = rank_tags_with_templates(cb, img_feat, CANDIDATE_TAGS, TAG_TEMPLATES, topk=args.topk)
    print(f"[retrieval] Top tags: {top_tags}")

    # Captions: prepare corpus
    captions = []
    if args.caption_corpus and os.path.isfile(args.caption_corpus):
        with open(args.caption_corpus, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if 5 <= len(s.split()) <= 30:  # keep concise sentences
                    captions.append(s)
    else:
        captions = DEFAULT_CAPTIONS

    top_caps = retrieve_captions(cb, img_feat, captions, topk=args.topk)
    print(f"[retrieval] Top captions: {top_caps}")

    if args.mode == "retrieval":
        print("[done] Retrieval-only mode completed.")
        return

    # 4) LLM paths
    if args.mode == "soft":
        # Build projector (random unless you load trained weights)
        projector = (LinearProjector(D) if args.projector == "linear" else MLPProjector(D))
        projector.eval()
        soft = make_soft_tokens(img_feat, projector, T=args.soft_tokens)  # [T, 4096]

        # Compose a helpful default prompt if none provided
        prompt = args.prompt or (
            "You are a concise technical describer. Given a latent visual prefix, "
            "describe the likely figure in 2–3 sentences, focusing on structure and relationships."
        )
        try:
            out = call_qwen3_with_softprefix(soft, prompt)
            print("\n[Qwen3-soft] Output:\n" + out)
        except Exception as e:
            print("[error] Qwen3 soft-prefix call failed:", e)
            print("Hint: Ensure your vLLM server is running with --enable-prompt-embeds and OPENAI_BASE_URL set.")

    elif args.mode == "vl":
        # Need a real image file path to send; if we started from PDF, use the preview we saved
        image_path = preview or args.input
        prompt = args.prompt or "Describe the image in 2–3 sentences focusing on structure and relationships."
        try:
            out = call_qwenvl_with_image(image_path, prompt)
            print("\n[Qwen-VL] Output:\n" + out)
        except Exception as e:
            print("[error] Qwen-VL call failed:", e)
            print("Hint: Ensure your vLLM server is running a VL model and OPENAI_BASE_URL is set.")


if __name__ == "__main__":
    main()
