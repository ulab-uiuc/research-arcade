# examples/clip_demo.py
# ------------------------------------------------------------
# Robust CLIP -> Qwen via vLLM, using a main-guard + spawn mode
# ------------------------------------------------------------
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # must be set before importing vllm

import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from vllm import LLM, SamplingParams


# -------------------------------
# 0) Config
# -------------------------------
MODEL_ID = "Qwen/Qwen3-8B"
PDF_PATH = "./Konigsberg2.pdf"
JPG_SAVE_PATH = "./Konigsberg2_page1.jpg"   # optional on-disk preview
PDF_DPI = 200
TOPK_TAGS = 5

# A small candidate label bank for CLIP zero-shot matching.
# You can expand/tune this to your domain (maps/graphs/bridges/diagrams, etc.).
CANDIDATE_TAGS = [
    "diagram", "graph", "map", "network", "bridges", "math figure",
    "handwritten notes", "printed text", "table", "chart", "equation",
    "street map", "topology", "graph theory", "flowchart"
]


# -------------------------------
# 1) PDF -> PIL image
# -------------------------------
def pdf_first_page_to_image(pdf_path: str, dpi: int = 200, save_jpg_path: str = None) -> Image.Image:
    pages = convert_from_path(pdf_path, dpi=dpi)   # requires poppler
    if not pages:
        raise RuntimeError(f"No pages found in PDF: {pdf_path}")
    img = pages[0].convert("RGB")
    if save_jpg_path:
        img.save(save_jpg_path, "JPEG")
    return img


# -------------------------------
# 2) Load CLIP and compute features
# -------------------------------
def load_clip():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor


@torch.no_grad()
def clip_image_features(clip_model: CLIPModel, clip_processor: CLIPProcessor, image: Image.Image) -> torch.Tensor:
    inputs = clip_processor(images=image, return_tensors="pt")
    feats = clip_model.get_image_features(**inputs)         # [1, D]
    feats = feats / feats.norm(dim=-1, keepdim=True)        # cosine norm
    return feats


@torch.no_grad()
def clip_rank_tags(
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    image_feats: torch.Tensor,
    candidates: list[str],
    topk: int = 5
) -> list[str]:
    text_inputs = clip_processor(text=candidates, padding=True, return_tensors="pt")
    text_feats = clip_model.get_text_features(**text_inputs)    # [N, D]
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    # cosine similarity: [1, D] @ [D, N] -> [1, N]
    sims = (image_feats @ text_feats.T).squeeze(0)
    topk_idx = torch.topk(sims, k=min(topk, sims.shape[0])).indices.tolist()
    return [candidates[i] for i in topk_idx]


# -------------------------------
# 3) (Experimental) Project to Qwen hidden size
#     NOTE: vLLM does NOT support directly overriding token embeddings
#           at runtime; we keep this to inspect the shape only.
# -------------------------------

def project_to_qwen_space(img_feats: torch.Tensor, target_dim: int = 4096) -> torch.Tensor:
    projector = torch.nn.Linear(img_feats.shape[-1], target_dim, bias=True)
    with torch.no_grad():
        img_token = projector(img_feats)   # [1, target_dim]
    return img_token


# -------------------------------
# 4) Build Qwen with vLLM (safe init)
# -------------------------------
def build_llm(model_id: str = MODEL_ID) -> LLM:
    # Keep memory modest; raise later if nvidia-smi shows headroom.
    llm = LLM(
        model_id,
        trust_remote_code=True,
        enforce_eager=True,          # avoids long torch.compile init
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,  # safer on shared GPU; bump later
        # max_model_len=8192,        # uncomment to shrink KV cache if needed
        # disable_custom_all_reduce=True,  # optional stability toggle
    )
    return llm


def llm_smoke(llm: LLM):
    out = llm.generate(["ping"], SamplingParams(max_tokens=8, temperature=0))
    print("LLM smoke:", out[0].outputs[0].text.strip())


# -------------------------------
# 5) Compose prompt and generate
# -------------------------------
def compose_prompt_from_tags(tags: list[str]) -> str:
    tag_text = ", ".join(tags)
    return (
        "You are a concise technical describer.\n"
        f"Image tags (from CLIP): {tag_text}.\n"
        "Describe the image clearly and succinctly in 2–3 sentences, "
        "focusing on structure and relationships."
    )


def qwen_generate(llm: LLM, prompt: str):
    params = SamplingParams(max_tokens=128, temperature=0.2)
    out = llm.generate([prompt], params)
    return out[0].outputs[0].text.strip()


# -------------------------------
# 6) Main
# -------------------------------
def main():
    # A) PDF -> image
    image = pdf_first_page_to_image(PDF_PATH, dpi=PDF_DPI, save_jpg_path=JPG_SAVE_PATH)
    print(f"Saved first page preview to: {JPG_SAVE_PATH}")

    # B) CLIP features + top-K tags
    clip_model, clip_processor = load_clip()
    img_feats = clip_image_features(clip_model, clip_processor, image)
    print("CLIP image features:", tuple(img_feats.shape))
    
    top_tags = clip_rank_tags(clip_model, clip_processor, img_feats, CANDIDATE_TAGS, topk=TOPK_TAGS)
    print("Top CLIP tags:", top_tags)

    # (Optional) show the experimental projection (shape check only)
    projected = project_to_qwen_space(img_feats, target_dim=4096)
    print("Projected to Qwen hidden size:", tuple(projected.shape))

    # C) Build vLLM/Qwen and run a smoke test + real prompt
    llm = build_llm(MODEL_ID)
    llm_smoke(llm)
    
    prompt = compose_prompt_from_tags(top_tags)
    print("\n--- Prompt to Qwen ---\n", prompt, "\n----------------------\n")
    
    text = qwen_generate(llm, prompt)
    print("Qwen output:\n", text)


if __name__ == "__main__":
    main()
