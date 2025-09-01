import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Any, Sequence

from dotenv import load_dotenv
from graph_constructor.utils import figure_latex_path_to_path
import shutil
import re
import unicodedata
from PIL import Image
import base64
import sys
import fitz
import io
from openai import OpenAI
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, AutoProcessor

from vllm import LLM, SamplingParams
import torch
import json
from pdf2image import convert_from_path
import numpy as np
from typing import Any, cast
from typing_extensions import TypeGuard
from collections.abc import Mapping


SOFT_TOKENS = int(os.getenv("SOFT_TOKENS", "16")) 

try:
    import fitz  # PyMuPDF for rendering PDFs
    _HAVE_PYMUPDF = True
except Exception:
    _HAVE_PYMUPDF = False

_GLOBAL_LLM = None


load_dotenv()

TOPK_TAGS = 1
"""
The local vllm version of task implementation varies from the usual one in two different ways.

Firstly, instead of directly interacting with the postgresql database, we assume that adequate data are given in the form of csv files.

Secondly, as the name suggests, the language model is deployed locally, using the transformer architecture, instead of making the API call.

Thirdly, we assume that it works on non visual language model, where the language model only takes textual information as input. Therefore, an adaptor is required.
"""


@dataclass
class Args:
    paragraph_ids: List[int]
    model_name: str
    k_neighbour: int = 2
    figure_available: bool = True
    table_available: bool = True
    download_path: str = "./download"


# ---------- Config ----------
PASSWORD = os.getenv("PGPASSWORD", "REPLACE_ME")
API_KEY = os.getenv("API_KEY")

def get_or_create_llm(
    model_name: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "float16",
    enable_prompt_embeds: bool = False,
):
    """Get or create a singleton LLM instance"""
    global _GLOBAL_LLM
    
    if _GLOBAL_LLM is None:
        print(f"Initializing LLM with model: {model_name}")
        
        # Determine if we should enable prompt embeds based on model capabilities
        try:
            # Try to import EmbedsPrompt to check if vLLM supports it
            from vllm.inputs import EmbedsPrompt
            can_use_embeds = enable_prompt_embeds
        except ImportError:
            can_use_embeds = False
            if enable_prompt_embeds:
                print("[WARN] vLLM version doesn't support EmbedsPrompt, falling back to text-only")
        
        llm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "enforce_eager": True,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "enforce_eager": True,
        }
        
        # Only add enable_prompt_embeds if the vLLM version supports it
        if can_use_embeds:
            llm_kwargs["enable_prompt_embeds"] = True
            
        _GLOBAL_LLM = LLM(**llm_kwargs)
    
    return _GLOBAL_LLM

def llm_generate(
    prompts: List[str],
    model_name: str,
    is_vlm: bool,
    image_embeddings: Optional[List[List[str]]] = None,
    image_labels: Optional[List[List[str]]] = None,
    image_projections: Optional[List[List[str]]] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    tensor_parallel_size: int = 1, # TODO adjust it dynamically
    gpu_memory_utilization: float = 0.99,
    dtype: str = "float16",
    clear_cuda_cache: bool = False,
    use_lora: bool = False,
    use_image_projection: bool = True,
    lora_path: str | None = None,
) -> List[str]:
    """
    Generate completions with vLLM (Python API only).
    """
    # --------- Validate shapes ----------
    n = len(prompts)
    if is_vlm:
        if image_labels is None:
            image_labels = [[] for _ in range(n)]
        if len(image_labels) != n:
            raise ValueError("len(image_labels) must match len(prompts) when is_vlm=True.")
    else:
        if image_labels is None:
            image_labels = [[] for _ in range(n)]
        if len(image_labels) != n:
            raise ValueError("len(image_labels) must match len(prompts) when not using projections.")

    # --------- Build augmented prompts ----------
    augmented_prompts: List[Any] = []
    for i in range(n):
        p = prompts[i]
        labels = image_labels[i] if image_labels else []

        # Text-only mode with optional labels
        if labels:
            label_text = ", ".join([str(x) for x in labels[:32]])
            p = f"[Image tags]: {label_text}\n\n{p}"
        augmented_prompts.append(p)

    # --------- Initialize model ----------
    llm = get_or_create_llm(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        enable_prompt_embeds=False  # projections removed
    )

    if use_lora and lora_path:
        try:
            llm.load_lora(lora_path, lora_name="my_adapter")
            llm.set_active_loras(["my_adapter"])
        except Exception as e:
            print(f"[WARN] Failed to load LoRA: {e}")

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # --------- Batch generate ----------
    outs = llm.generate(augmented_prompts, sampling)

    # --------- Collect texts ----------
    answers: List[str] = []
    for out in outs:
        if out and out.outputs:
            answers.append(out.outputs[0].text.strip())
        else:
            answers.append("")

    if clear_cuda_cache:
        torch.cuda.empty_cache()

    return answers

def _is_embeds_prompt(x) -> TypeGuard["EmbedsPrompt"]:
    # pick the minimal set of keys that identifies your EmbedsPrompt at runtime
    if not isinstance(x, Mapping):
        return False
    # Adjust the keys below to match your TypedDict fields:
    required_keys = {"text", "soft_tokens"}   # OR {"prompt", "embeds"} etc.
    return required_keys.issubset(x.keys())

def _decode_prompt_embeds_payload(p: Dict[str, Any]) -> np.ndarray:
    """Decode base64 prompt embeddings payload"""
    raw = base64.b64decode(p["data"])
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.reshape(p["shape"])

def _to_prompt_embeds_payload(soft_np: np.ndarray) -> Dict[str, Any]:
    """Encode soft tokens for engines that accept OpenAI-compatible prompt_embeds payload."""
    payload = base64.b64encode(soft_np.tobytes()).decode("utf-8")
    return {"data": payload, "dtype": "float32", "shape": list(soft_np.shape)}



def _decode_prompt_embeds_payload(p: Dict[str, Any]) -> np.ndarray:
    """Decode base64 prompt embeddings payload"""
    raw = base64.b64decode(p["data"])
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.reshape(p["shape"])

def _to_prompt_embeds_payload(soft_np: np.ndarray) -> Dict[str, Any]:
    """Encode soft tokens for engines that accept OpenAI-compatible prompt_embeds payload."""
    payload = base64.b64encode(soft_np.tobytes()).decode("utf-8")
    return {"data": payload, "dtype": "float32", "shape": list(soft_np.shape)}



def _repeat_to_soft_tokens(proj_1xH: torch.Tensor, T: int = SOFT_TOKENS) -> np.ndarray:
    """
    proj_1xH: torch.Tensor of shape [1, 4096] (Qwen3 hidden dim)
    returns np.ndarray [T, 4096] float32 (tiled)
    """
    if proj_1xH.ndim != 2 or proj_1xH.shape[0] != 1:
        raise ValueError(f"expected [1, H], got {tuple(proj_1xH.shape)}")
    soft = proj_1xH.repeat(T, 1).contiguous().detach().cpu().numpy().astype(np.float32)
    return soft




def _data_extraction_vlm(jsonl_file_path, use_figure=True, use_table=True, model_name="nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"):
    """Read a JSONL file and return a list of dicts ready for LLM generation."""
    outputs: List[Dict[str, Any]] = []
    download_path = "./download"

    with open(jsonl_file_path, "r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[WARN] Skipping line {line_no}: invalid JSON ({e})")
                continue

            # Extract fields with safe fallbacks
            paragraph_key_id = rec.get("paragraph_key_id")
            paper_arxiv_id = rec.get("paper_arxiv_id", "")
            paper_section = rec.get("paper_section", "")
            paragraph_id_loc = rec.get("paragraph_id_local")
            original_content = rec.get("original_content", "") or ""
            num_char = int(rec.get("num_char") or len(original_content))
            title = rec.get("title", "") or ""
            abstract = rec.get("abstract", "") or ""
            prev_paras = rec.get("prev_paras", []) or []
            next_paras = rec.get("next_paras", []) or []
            figures = rec.get("figures", []) or []
            tables = rec.get("tables", []) or []
            fig_labels = rec.get("fig_labels", []) or []
            table_labels = rec.get("table_labels", []) or []
            bib_keys = rec.get("bib_keys", []) or []
            cited_triples = rec.get("cited_triples", []) or []

            # Normalize abstracts
            abstracts_norm = []
            for t in cited_triples:
                if isinstance(t, dict):
                    abstracts_norm.append((
                        t.get("0") or t.get("bib_key") or "",
                        t.get("1") or t.get("title") or "",
                        t.get("2") or t.get("abstract") or "",
                    ))
                elif isinstance(t, (list, tuple)):
                    bk = t[0] if len(t) > 0 else ""
                    ttl = t[1] if len(t) > 1 else ""
                    abs_ = t[2] if len(t) > 2 else ""
                    abstracts_norm.append((bk or "", ttl or "", abs_ or ""))

            k = max(len(prev_paras), len(next_paras)) if (prev_paras or next_paras) else 2

            prompt = _build_prompt_contract(
                k=k,
                figures=figures,
                tables=tables,
                abstracts=abstracts_norm,
                prev_paras=prev_paras,
                next_paras=next_paras,
                paper_section=paper_section,
                num_char=num_char,
                title=title,
                abstract=abstract,
                fig_labels=fig_labels,
                table_labels=table_labels,
                bib_keys=bib_keys,
                use_figure=use_figure,
                use_table=use_table,
            )

            # Process figures
            figure_paths = []
            for f in figures:
                latex_path = (f.get("path") or "").strip()
                if not latex_path:
                    continue
                try:
                    abs_path = figure_latex_path_to_path(
                        path=download_path,
                        arxiv_id=paper_arxiv_id,
                        latex_path=latex_path,
                    )
                    figure_paths.append(abs_path)
                except Exception as e:
                    print(f"[WARN] Failed to resolve figure path '{latex_path}': {e}")

            image_embeddings = _figure_paths_to_embeddings(figure_paths=figure_paths, model_path=model_name)
            
            outputs.append({
                "paragraph_key_id": paragraph_key_id,
                "paper_arxiv_id": paper_arxiv_id,
                "prompt": prompt,
                "original_content": original_content,
                "image_embeddings": image_embeddings,
                "fig_labels": fig_labels,
                "table_labels": table_labels,
                "bib_keys": bib_keys,
                "k": k,
                "meta": {
                    "paper_section": paper_section,
                    "paragraph_id_local": paragraph_id_loc,
                    "title": title,
                    "abstract": abstract,
                    "num_char": num_char,
                    "prev_paras": prev_paras,
                    "next_paras": next_paras,
                    "figures": figures,
                    "tables": tables,
                    "cited_triples": abstracts_norm,
                    "figure_paths": figure_paths,
                }
            })

    return outputs


def _data_extraction_non_vlm(jsonl_file_path, use_figure=True, use_table=True, model_name="openai/clip-vit-base-patch32"):
    """Extract data for non-VLM models with visual adaptation"""
    outputs: List[Dict[str, Any]] = []
    download_path = "./download"

    with open(jsonl_file_path, "r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[WARN] Skipping line {line_no}: invalid JSON ({e})")
                continue

            # Extract all fields (same as VLM version)
            paragraph_key_id = rec.get("paragraph_key_id")
            paper_arxiv_id = rec.get("paper_arxiv_id", "")
            paper_section = rec.get("paper_section", "")
            paragraph_id_loc = rec.get("paragraph_id_local")
            original_content = rec.get("original_content", "") or ""
            num_char = int(rec.get("num_char") or len(original_content))
            title = rec.get("title", "") or ""
            abstract = rec.get("abstract", "") or ""
            prev_paras = rec.get("prev_paras", []) or []
            next_paras = rec.get("next_paras", []) or []
            figures = rec.get("figures", []) or []
            tables = rec.get("tables", []) or []
            fig_labels = rec.get("fig_labels", []) or []
            table_labels = rec.get("table_labels", []) or []
            bib_keys = rec.get("bib_keys", []) or []
            cited_triples = rec.get("cited_triples", []) or []

            # Normalize abstracts
            abstracts_norm = []
            for t in cited_triples:
                if isinstance(t, dict):
                    abstracts_norm.append((
                        t.get("0") or t.get("bib_key") or "",
                        t.get("1") or t.get("title") or "",
                        t.get("2") or t.get("abstract") or "",
                    ))
                elif isinstance(t, (list, tuple)):
                    bk = t[0] if len(t) > 0 else ""
                    ttl = t[1] if len(t) > 1 else ""
                    abs_ = t[2] if len(t) > 2 else ""
                    abstracts_norm.append((bk or "", ttl or "", abs_ or ""))

            k = max(len(prev_paras), len(next_paras)) if (prev_paras or next_paras) else 2

            prompt = _build_prompt_contract(
                k=k,
                figures=figures,
                tables=tables,
                abstracts=abstracts_norm,
                prev_paras=prev_paras,
                next_paras=next_paras,
                paper_section=paper_section,
                num_char=num_char,
                title=title,
                abstract=abstract,
                fig_labels=fig_labels,
                table_labels=table_labels,
                bib_keys=bib_keys,
                use_figure=use_figure,
                use_table=use_table,
            )

            # Process figure paths
            figure_paths = []
            for f in figures:
                latex_path = (f.get("path") or "").strip()
                if not latex_path:
                    continue
                try:
                    abs_path = figure_latex_path_to_path(
                        path=download_path,
                        arxiv_id=paper_arxiv_id,
                        latex_path=latex_path,
                    )
                    figure_paths.append(abs_path)
                except Exception as e:
                    print(f"[WARN] Failed to resolve figure path '{latex_path}': {e}")

            # Get visual adaptations (tags and projections)
            image_tags_projections = visual_adaptation(image_paths=figure_paths)
            
            image_tag_list: List[List[str]] = []
            image_projection_payloads: List[Dict[str, Any]] = []

            for tag, projection in image_tags_projections:
                image_tag_list.append(tag if tag else [])
                if projection is not None:
                    try:
                        soft = _repeat_to_soft_tokens(projection, T=SOFT_TOKENS)
                        image_projection_payloads.append(_to_prompt_embeds_payload(soft))
                    except Exception as e:
                        print(f"[WARN] Failed to build soft tokens: {e}")

            outputs.append({
                "paragraph_key_id": paragraph_key_id,
                "paper_arxiv_id": paper_arxiv_id,
                "prompt": prompt,
                "original_content": original_content,
                "image_tag_list": image_tag_list,
                "image_projections": image_projection_payloads,
                "fig_labels": fig_labels,
                "table_labels": table_labels,
                "bib_keys": bib_keys,
                "k": k,
                "meta": {
                    "paper_section": paper_section,
                    "paragraph_id_local": paragraph_id_loc,
                    "title": title,
                    "abstract": abstract,
                    "num_char": num_char,
                    "prev_paras": prev_paras,
                    "next_paras": next_paras,
                    "figures": figures,
                    "tables": tables,
                    "cited_triples": abstracts_norm,
                    "figure_paths": figure_paths,
                    "soft_tokens_T": SOFT_TOKENS,
                }
            })

    return outputs


# def load_model(model_name):
#     clip_model = CLIPModel.from_pretrained(model_name)
#     clip_processor = CLIPProcessor.from_pretrained(model_name)
#     return clip_model, clip_processor


# @torch.no_grad()
# def clip_image_features(clip_model: CLIPModel, clip_processor: CLIPProcessor, image: Image.Image) -> torch.Tensor:
#     inputs = clip_processor(images=image, return_tensors="pt")
#     feats = clip_model.get_image_features(**inputs)         # [1, D]
#     feats = feats / feats.norm(dim=-1, keepdim=True)        # cosine norm
#     return feats




# @torch.no_grad()
# def clip_rank_tags(
#     clip_model: CLIPModel,
#     clip_processor: CLIPProcessor,
#     image_feats: torch.Tensor, 
#     candidates: Sequence[str],
#     tag_templates: Sequence[str],
#     topk: int = 5,
# ) -> List[str]:
#     texts = [tpl.format(tag) for tag in candidates for tpl in tag_templates]
#     if not texts:
#         return []
#     inputs = clip_processor(text=texts, padding=True, return_tensors="pt")
#     text_feats = clip_model.get_text_features(**inputs)
#     text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
#     sims = (image_feats @ text_feats.T).squeeze(0)               # [len(cands)*len(tpls)]

#     T = len(tag_templates)
#     sims_by_tag = sims.view(len(candidates), T)
#     # pick best template for each tag
#     best_vals, best_tpl_idx = sims_by_tag.max(dim=1)             # [len(candidates)]
#     # rank tags by their best template
#     k = min(topk, len(candidates))
#     tag_order = torch.topk(best_vals, k=k).indices.tolist()
#     return [tag_templates[best_tpl_idx[i]].format(candidates[i]) for i in tag_order]


def project_to_embedding_space(img_feats: torch.Tensor, target_dim: int = 4096) -> torch.Tensor:
    projector = torch.nn.Linear(img_feats.shape[-1], target_dim, bias=True)
    with torch.no_grad():
        img_token = projector(img_feats)   # [1, target_dim]
    return img_token


def _as_rgb_image(path: Path) -> Optional[Image.Image]:
    """
    Open a raster image as PIL.Image in RGB. Returns None if it fails.
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return im
    except Exception as e:
        print(f"[WARN] Failed to open image '{path}': {e}")
        return None

def visual_adaptation(image_paths, model_name="llava-hf/llava-1.5-7b-hf", projection_dimension=4096):
    """
    Updated visual_adaptation function that uses LLaVA for image description instead of CLIP tags.
    Returns pairs of [description_list, projection] where description_list contains the LLaVA description.
    """
    pairs = []
    images = []

    # First, load all images
    for p in image_paths:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] Path not found: {path}")
            images.append(None)
            continue

        ext = path.suffix.lower()
        if ext == ".pdf":
            img = pdf_first_page_to_image(path)
        elif ext in {".jpg", ".jpeg", ".png"}:
            img = _as_rgb_image(path)
        else:
            print(f"[WARN] Unsupported extension '{ext}' for {path}; skipping.")
            img = None

        images.append(img)

    # Load LLaVA model once for all images
    llava_model, llava_processor = None, None
    if any(img is not None for img in images):
        try:
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            
            llava_processor = LlavaProcessor.from_pretrained(model_name)
            llava_model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()
            print(f"Loaded LLaVA model: {model_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load LLaVA model: {e}")
            # Fallback to CLIP if LLaVA fails
            # return _visual_adaptation_clip_fallback(image_paths, projection_dimension)
            return []
        
    # Process each image
    for image in images:
        if not image:
            pairs.append([None, None])
            continue

        try:
            # Generate description with LLaVA
            description = _generate_llava_description(llava_model, llava_processor, image)
            print(f"LLaVA description: {description[:100]}...")  # Show first 100 chars
            
            # Create projection if needed
            projected = None
            # if projection_dimension:
                # For projection, we still need CLIP features as LLaVA doesn't directly provide embeddings
                # Load CLIP for projection only
                # clip_model, clip_processor = load_model("openai/clip-vit-base-patch32")
                # img_feats = clip_image_features(clip_model, clip_processor, image)
                # projected = project_to_embedding_space(img_feats, target_dim=projection_dimension)
            
            # Return description as a list (to maintain compatibility with existing code structure)
            pairs.append([[description], projected])
            
        except Exception as e:
            print(f"[ERROR] Failed to process image with LLaVA: {e}")
            pairs.append([None, None])

    return pairs


def _generate_llava_description(model, processor, image: Image.Image) -> str:
    """Generate description using LLaVA model"""
    try:
        # Create prompt for academic/technical image description
        prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nDescribe this image in detail, focusing on its structure, content, and any technical or scientific elements you can identify. ASSISTANT:"
        
        # Process inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        input_token_len = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[0][input_token_len:]
        description = processor.decode(generated_tokens, skip_special_tokens=True)
        
        return description.strip()
        
    except Exception as e:
        print(f"[ERROR] LLaVA description generation failed: {e}")
        return "Image description unavailable"


# def _visual_adaptation_clip_fallback(image_paths, projection_dimension=4096):
    """
    # Fallback to original CLIP-based visual adaptation if LLaVA fails
    # """
    # pairs = []
    # images = []

    # for p in image_paths:
    #     path = Path(p)
    #     if not path.exists():
    #         print(f"[WARN] Path not found: {path}")
    #         images.append(None)
    #         continue

    #     ext = path.suffix.lower()
    #     if ext == ".pdf":
    #         img = pdf_first_page_to_image(path)
    #     elif ext in {".jpg", ".jpeg", ".png"}:
    #         img = _as_rgb_image(path)
    #     else:
    #         print(f"[WARN] Unsupported extension '{ext}' for {path}; skipping.")
    #         img = None

    #     images.append(img)

    # for image in images:
    #     if not image:
    #         pairs.append([None, None])
    #         continue

        # Load CLIP model
        # clip_model, clip_processor = load_model("openai/clip-vit-base-patch32")
        # img_feats = clip_image_features(clip_model, clip_processor, image)
        # print("CLIP image features:", tuple(img_feats.shape))
        
        # Use CLIP tags as fallback
        # top_tags = clip_rank_tags(
        #     clip_model=clip_model, 
        #     clip_processor=clip_processor, 
        #     image_feats=img_feats, 
        #     candidates=CANDIDATE_TAGS, 
        #     tag_templates=TAG_TEMPLATES, 
        #     topk=TOPK_TAGS
        # )
    #     print("Top CLIP tags (fallback):", top_tags)
        
    #     projected = None
    #     if projection_dimension:
    #         projected = project_to_embedding_space(img_feats, target_dim=projection_dimension)
    #     pairs.append([top_tags, projected])

    # return pairs

def _repeat_to_soft_tokens(proj_1xH: torch.Tensor, T: int = SOFT_TOKENS) -> np.ndarray:
    """
    proj_1xH: torch.Tensor of shape [1, 4096] (Qwen3 hidden dim)
    returns np.ndarray [T, 4096] float32 (tiled)
    """
    if proj_1xH.ndim != 2 or proj_1xH.shape[0] != 1:
        raise ValueError(f"expected [1, H], got {tuple(proj_1xH.shape)}")
    soft = proj_1xH.repeat(T, 1).contiguous().detach().cpu().numpy().astype(np.float32)  # [T, 4096]
    return soft


def _to_prompt_embeds_payload(soft_np: np.ndarray) -> Dict[str, Any]:
    """
    Encodes soft tokens for engines that accept OpenAI-compatible prompt_embeds payload.
    """
    payload = base64.b64encode(soft_np.tobytes()).decode("utf-8")
    return {"data": payload, "dtype": "float32", "shape": list(soft_np.shape)}  # [T, 4096]


def _decode_prompt_embeds_payload(p: Dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(p["data"])
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.reshape(p["shape"])


def pdf_first_page_to_image(pdf_path: str, dpi: int = 200, save_jpg_path: str = None) -> Image.Image:
    pages = convert_from_path(pdf_path, dpi=dpi)   # requires poppler
    if not pages:
        raise RuntimeError(f"No pages found in PDF: {pdf_path}")
    img = pages[0].convert("RGB")
    if save_jpg_path:
        img.save(save_jpg_path, "JPEG")
    return img


def _format_block(label: str, lines: List[str]) -> str:
    if not lines:
        return ""
    header = f"{label}:\n"
    return header + "\n".join(lines)


def _sanitize(s: Optional[str], limit: int) -> str:
    if not s:
        return ""
    return s.replace("\n", " ").strip()[:limit]


def _build_prompt_contract(
    k: int,
    figures: List[Dict],
    tables: List[Dict],
    abstracts: List[Tuple[str, str, str]],
    prev_paras: List[str],
    next_paras: List[str],
    paper_section: str,
    num_char: int,
    title: str,
    abstract: str,
    fig_labels: List[str],
    table_labels: List[str],
    bib_keys: List[str],
    use_figure: bool = True,
    use_table: bool = True,
    use_citations: bool = True,
) -> str:
    """Capability-contract prompt.
    - If a capability is False, we DON'T include its requirement line,
      and we add a negative constraint to forbid it.
    """

    # Prune sources by capability
    if not use_figure:
        figures, fig_labels = [], []
    if not use_table:
        tables, table_labels = [], []
    if not use_citations:
        abstracts, bib_keys = [], []

    # Build descriptive blocks
    fig_lines = [
        f"- label: {f.get('label','')}; caption: {_sanitize(f.get('caption'), 240)}"
        for f in figures if f.get("label")
    ]
    tab_lines = [
        f"- label: {t.get('label','')}; text: {_sanitize(t.get('text'), 240)}"
        for t in tables if t.get("label")
    ]
    abs_lines = []
    for bk, t, a in abstracts:
        abs_lines.append(f"- [{bk}] {_sanitize(t,180)}: {_sanitize(a,600)}")

    def block(title: str, lines: List[str]) -> str:
        if not lines:
            return "(none)"
        return f"{title}:\n" + "\n".join(lines)

    # Context block
    adj = []
    if prev_paras:
        adj.append("Previous:")
        adj += [f"{i}. {p}" for i, p in enumerate(prev_paras, 1)]
    if next_paras:
        adj.append("Next:")
        adj += [f"{i}. {p}" for i, p in enumerate(next_paras, 1)]
    adjacent = "\n".join(adj) if adj else "(none)"

    # Capability flags (drive requirements)
    can_fig = bool(fig_labels)
    can_tab = bool(table_labels)
    can_cite = bool(bib_keys)

    reqs = []
    bans = []

    if can_fig:
        reqs.append('Include each figure label exactly once via \\ref{<label>}.')
    else:
        bans.append('Do not include any \\ref{...} to figures.')

    if can_tab:
        reqs.append('Include each table label exactly once via \\ref{<label>}.')
    else:
        bans.append('Do not include any \\ref{...} to tables.')

    if can_cite:
        reqs.append('Cite all bib_keys using a single \\cite{...} or multiple as needed.')
    else:
        bans.append('Do not include any \\cite{...}.')

    # Ordering rule only when applicable
    ordering = []
    if can_fig:
        ordering.append('- Follow the order in fig_labels when referring to figures.')
    if can_tab:
        ordering.append('- Follow the order in table_labels when referring to tables.')

    # Compose prompt
    prompt = f"""You are reconstructing one missing LaTeX paragraph in a research paper.

Title: {title}
Abstract: {_sanitize(abstract, 1200)}
Section name: {paper_section}

{block("Figure block (optional)", fig_lines)}
{block("Table block (optional)", tab_lines)}
{block("Cited paper titles/abstracts (optional)", abs_lines)}

k-most adjacent paragraphs (context):
{adjacent}

Target length (characters): ~{num_char} (±15%)

# Canonicalized metadata
fig_labels = {fig_labels}
table_labels = {table_labels}
bib_keys = {bib_keys}

# Task
Write exactly one LaTeX-formatted paragraph that naturally fits between the adjacent paragraphs.

# HARD REQUIREMENTS
- Maintain objective, concise academic tone; ensure logical continuity with context.
- Output exactly one paragraph (no lists/headers/environments, no blank lines).
{"".join(f"- {r}\n" for r in reqs)}{"".join(f"- {b}\n" for b in bans)}

# ORDERING
{("\n".join(ordering) if ordering else "(no ordering constraints)")}

# SILENT SELF-CHECK (do not print this checklist)
- If any fig_labels/table_labels exist: verify each appears exactly once as "\\ref{{<label>}}".
- If any bib_keys exist: verify they appear in "\\cite{{...}}".
- If none exist: verify the paragraph contains NO "\\ref{{...}}" and NO "\\cite{{...}}".
- Verify single paragraph and length within tolerance.

# Output
Return only the LaTeX paragraph text (no extra lines, no explanations).
""".strip()
    return prompt


def _build_prompt(
    k: int,
    figures: List[Dict],
    tables: List[Dict],
    abstracts: List[Tuple[str, str, str]],
    prev_paras: List[str],
    next_paras: List[str],
    paper_section: str,
    num_char: int,
    title: str,
    abstract: str,
    fig_labels: List[str],
    table_labels: List[str],
    bib_keys: List[str],
    use_figure: bool = True,
    use_table: bool = True,
) -> str:

    # If we don't use figures, we mask them out
    # If we don't use tables, we mask them out

    if not use_figure:
        figures = []
        fig_labels = []
    
    if not use_table:
        tables = []
        table_labels = []


    paper_title = title
    # Figure block
    fig_lines = []
    for f in figures:
        key_detail = f["caption"][:240].replace("\n", " ").strip() if f["caption"] else ""
        fig_lines.append(f"- label: {f['label']}; caption: {key_detail}")
    figure_block = _format_block("Figure (optional)", fig_lines).strip()

    # Table block
    tab_lines = []
    for t in tables:
        key_detail = t["text"][:240].replace("\n", " ").strip() if t["text"] else ""
        tab_lines.append(f"- label: {t['label']}; text: {key_detail}")
    table_block = _format_block("Table (optional)", tab_lines).strip()

    # Abstracts block
    abs_lines = []
    for bib_key, title, abstract in abstracts:
        short_abs = abstract[:600].replace("\n", " ").strip()
        abs_lines.append(f"- [{bib_key}] {title}: {short_abs}")
    abstracts_block = _format_block("Abstract(s) of cited paper(s)", abs_lines).strip()

    # Adjacent paragraphs
    adj_lines = []
    if prev_paras:
        adj_lines.append("Previous:")
        for i, p in enumerate(prev_paras, 1):
            adj_lines.append(f"{i}. {p}")
    if next_paras:
        adj_lines.append("Next:")
        for i, p in enumerate(next_paras, 1):
            adj_lines.append(f"{i}. {p}")
    adjacent_paragraphs_block = "\n".join(adj_lines).strip()

    # prompt = "Describe the provided images senquentially."
    print(f"figure_block: {figure_block}")
    print(f"Collected fig_labels: {fig_labels}")
    prompt = f"""
    You are given the following inputs for reconstructing a missing paragraph in a research paper.

    Title: {paper_title}
    Abstract: {abstract}
    Section name: {paper_section}

    Figure block (optional): {figure_block if figure_block else "(none)"}
    Table block (optional): {table_block if table_block else "(none)"}
    Cited paper titles/abstracts (optional): {abstracts_block if abstracts_block else "(none)"}

    k-most adjacent paragraphs (context): {adjacent_paragraphs_block}
    Target length (characters): {num_char}

    # Canonicalized metadata for enforcement (already cleaned)
    fig_labels = {fig_labels}          # e.g., ["fig:framework","fig:sd_latents"]; [] if none
    table_labels = {table_labels}      # e.g., ["tab:results"]; [] if none
    bib_keys = {bib_keys}              # e.g., ["smith2024", "lee2023"]; [] if none

    # Task
    Write exactly one LaTeX-formatted paragraph that naturally fits between the adjacent paragraphs.

    # HARD REQUIREMENTS (must all be satisfied)
    1) If fig_labels is non-empty, include each label **exactly once** using \\ref{{<label>}} in the paragraph text (e.g., "see Fig.~\\ref{{fig:framework}}").
    2) If table_labels is non-empty, include each label **exactly once** using \\ref{{<label>}}.
    3) If bib_keys is non-empty, **cite all of them** (you may group keys in a single \\cite{{...}}).
    - If allow_derived_bib_keys=true and a cited item lacks a key, derive a stable placeholder from its title: lowercase, keep a-z0-9 and hyphens only.
    - Do not invent any other keys.
    4) Maintain objective, concise academic tone; ensure logical continuity with the provided context.
    5) Length approximately equals to {num_char} (plus or minus 15%). Produce exactly one paragraph (no lists/headers/environments).

    # ORDERING
    - When mentioning multiple figures/tables, follow the order in fig_labels/table_labels.

    # SILENT SELF-CHECK (do not print this checklist)
    - Verify every l in fig_labels and table_labels appears exactly once as the substring "\\ref{{" + l + "}}".
    - Verify every key in bib_keys (plus any allowed derived keys) appears in a \\cite{{...}}.
    - Verify the output is a single paragraph (no blank lines).

    # Output
    Return only the LaTeX paragraph text. No explanations, no markdown, no extra lines.
    """.strip()

    
    return prompt

# We need to replace this one with the embedding model suggested by the hugging face example:


def _load_images_from_pdf(pdf_path: str, dpi: int = 144) -> List[Image.Image]:
    """
    Render each page of a PDF to a PIL Image (RGB). Requires PyMuPDF.
    """
    if not _HAVE_PYMUPDF:
        raise RuntimeError(
            "PDF support requires PyMuPDF (`pip install pymupdf`). "
            f"Cannot process: {pdf_path}"
        )
    imgs: List[Image.Image] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)  # scale by DPI
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            imgs.append(img)
    finally:
        doc.close()
    return imgs


def _collect_pil_images(figure_paths: List[str]) -> List[Image.Image]:
    """
    Loads/creates PIL images from a mix of image files and PDFs.
    """
    allowed = {".jpg", ".jpeg", ".png", ".pdf"}
    pil_images: List[Image.Image] = []

    for fs_path in figure_paths or []:
        if not os.path.isfile(fs_path):
            print(f"[WARN] Not a file: {fs_path}")
            continue

        ext = Path(fs_path).suffix.lower()
        if ext not in allowed:
            print(f"[WARN] Skipping unsupported extension ({ext}): {fs_path}")
            continue

        try:
            if ext == ".pdf":
                pil_images.extend(_load_images_from_pdf(fs_path))
            else:
                img = Image.open(fs_path).convert("RGB")
                pil_images.append(img)
        except Exception as e:
            print(f"[ERROR] Failed to process {fs_path}: {e}")
            continue

    return pil_images


def _figure_paths_to_embeddings(
    figure_paths: List[str],
    model_path: str = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Convert image/PDF paths to **precomputed image embeddings** for vLLM VLMs.

    Returns:
        {
          "image_embeds": List[torch.FloatTensor],   # one tensor per image/page (on CPU)
          "image_grid_thw": torch.LongTensor         # shape (N, 3), on CPU
        }

    Use with vLLM:
        out = llm.generate(
            prompts=[...],
            sampling_params=SamplingParams(...),
            multi_modal_data={
                "image": _figure_paths_to_embeddings([...])
            },
        )
    """
    # 1) Gather PIL images
    images = _collect_pil_images(figure_paths)
    if not images:
        # Return an empty, well-typed structure
        return []
    
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, device="cuda")
    image_features = image_processor(images)

    return image_features


def _slug(s, prefix="fig", maxlen=48):
    # make a deterministic placeholder from caption text
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\\caption\s*{", "", s, flags=re.I)
    s = s.replace("}", " ")
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    s = re.sub(r"-{2,}", "-", s)
    s = s[:maxlen].strip("-")
    return f"{prefix}:{s or 'placeholder'}"

def collect_fig_labels(figure_labels_raw):
    """
    fig_blocks: list like \\label{fig:framework}, ...

    return what is inside of the parenthesis using regular expression
    """
    _LABEL_RE = re.compile(r"""\\label\s*\{\s*([^}]+?)\s*\}""", re.IGNORECASE | re.DOTALL)
    texts = [figure_labels_raw] if isinstance(figure_labels_raw, str) else list(figure_labels_raw)
    found = []
    for s in texts:
        matches = _LABEL_RE.findall(s or "")
        for m in matches:
            lbl = m.strip()
            found.append(lbl)
    return found[0]


def paragraph_generation_vlm(prompt_list, image_embedding_list):
    result = []
    


# ---------- Example LLM adapter (plug what you use) ----------
# Keep it generic so you can swap any provider. For OpenAI, Anthropic, vLLM, etc.,
# just replace the body of the function below.

def example_openai_adapter(model: str = "gpt-4o-mini", temperature: float = 0.2):
    """
    Returns a function that takes a prompt and returns text.
    Replace with your chosen provider/wrapper.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("pip install openai")

    client = OpenAI()

    def _generate(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful scientific writer who outputs valid LaTeX."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    return _generate


