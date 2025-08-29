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
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, AutoTokenizer, AutoProcessor
from vllm import LLM, SamplingParams
import torch
import json
from pdf2image import convert_from_path

try:
    import fitz  # PyMuPDF for rendering PDFs
    _HAVE_PYMUPDF = True
except Exception:
    _HAVE_PYMUPDF = False


 

load_dotenv()

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

TOPK_TAGS = 1
"""
The local vllm version of task implementation varies from the usual one in two different ways.

Firstly, instead of directly interacting with the postgresql database, we assume that adequate data are given in the form of csv files.

Secondly, as the name suggests, the language model is deployed locally, using the transformer architecture, instead of making the API call.

Thirdly, we assume that it works on non visual language model, where the language model only takes textual information as input. Therefore, an adaptor is required.
"""

# ---------- Config ----------
PASSWORD = os.getenv("PGPASSWORD", "REPLACE_ME")
API_KEY = os.getenv("API_KEY")

@dataclass
class Args:
    paragraph_ids: List[int]
    model_name: str
    k_neighbour: int = 2
    figure_available: bool = True
    table_available: bool = True
    download_path: str = "./download"


from typing import List, Optional
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from openai import OpenAI


from typing import List, Optional
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

def llm_generate(
    prompts: List[str],
    model_name: str,
    is_vlm: bool,
    image_embeddings: Optional[List[List[str]]] = None,  # kept for API parity; not used with Python vLLM API
    image_labels: Optional[List[List[str]]] = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "bfloat16",
    clear_cuda_cache: bool = False,
) -> List[str]:
    """
    Generate completions with vLLM (Python API only).
    - If is_vlm == True, we *inline* per-image labels as textual hints because the Python API
      doesn't take image tensors/blobs. (image_embeddings are ignored here.)
    - If is_vlm == False, image_labels (if provided) are also inlined as textual context.

    Returns: List[str] of model outputs aligned with `prompts`.
    """

    # --------- Validate shapes ----------
    n = len(prompts)
    if is_vlm:
        # Ensure lists are aligned; we'll ignore embeddings but keep shape checks
        if image_labels is None:
            image_labels = [[] for _ in range(n)]
        if len(image_labels) != n:
            raise ValueError("len(image_labels) must match len(prompts) when is_vlm=True.")
        if image_embeddings is not None and len(image_embeddings) != n:
            # Clarify that embeddings aren't used in this pathway
            raise ValueError(
                "len(image_embeddings) must match len(prompts) if provided; "
                "note: image embeddings are not consumed by the vLLM Python API."
            )
    else:
        if image_labels is None:
            image_labels = [[] for _ in range(n)]
        if len(image_labels) != n:
            raise ValueError("len(image_labels) must match len(prompts) when is_vlm=False.")

    # --------- Build augmented prompts ----------
    augmented_prompts: List[str] = []
    for i in range(n):
        p = prompts[i]
        labels = image_labels[i] if image_labels else []
        prefix_lines = []

        if labels:
            # Keep short to avoid blowing up prompt length
            label_text = ", ".join([str(x) for x in labels[:32]])
            prefix_lines.append(
                "You may use the following image tag summary as auxiliary context:\n"
                f"[Image tags]: {label_text}"
            )

        if is_vlm and image_embeddings and image_embeddings[i]:
            # Let future devs know what's happening
            prefix_lines.append(
                "(Note: Images were provided externally, but this Python vLLM API path "
                "does not accept image tensors; proceeding with textual tags only.)"
            )

        if prefix_lines:
            p = "\n".join(prefix_lines) + "\n\n" + p

        augmented_prompts.append(p)

    # --------- Initialize tokenizer & model (your pattern) ----------
    model_id = model_name
    _ = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)  # keep for parity/compat
    llm = LLM(
        model_id,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )

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
        if out.outputs:
            answers.append(out.outputs[0].text.strip())
        else:
            answers.append("")
    if clear_cuda_cache:
        torch.cuda.empty_cache()

    return answers



def _data_extraction_vlm(jsonl_file_path, model_name="nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"):
    """
    Read a JSONL file created by `_data_fetching` and return a list of dicts
    ready to use for LLM generation.

    Each returned dict contains:
      - paragraph_key_id (int)
      - paper_arxiv_id (str)
      - prompt (str)                          # built via _build_prompt(...)
      - original_content (str)                # ground truth paragraph
      - image_embeddings (List[str])          # base64 images for figures (PDF pages expanded)
      - fig_labels (List[str])
      - table_labels (List[str])
      - bib_keys (List[str])
      - k (int)                               # we infer from context length; defaults to 2
      - meta (Dict[str, Any])                 # everything else from the record for debugging
    """
    outputs: List[Dict[str, Any]] = []
    # Use the same default download path as in Args
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

            # Required core fields with safe fallbacks
            paragraph_key_id = rec.get("paragraph_key_id")
            paper_arxiv_id   = rec.get("paper_arxiv_id", "")
            paper_section    = rec.get("paper_section", "")
            paragraph_id_loc = rec.get("paragraph_id_local")
            original_content = rec.get("original_content", "") or ""
            num_char         = int(rec.get("num_char") or len(original_content))
            title            = rec.get("title", "") or ""
            abstract         = rec.get("abstract", "") or ""
            prev_paras       = rec.get("prev_paras", []) or []
            next_paras       = rec.get("next_paras", []) or []
            figures          = rec.get("figures", []) or []
            tables           = rec.get("tables", []) or []
            fig_labels       = rec.get("fig_labels", []) or []
            table_labels     = rec.get("table_labels", []) or []
            bib_keys         = rec.get("bib_keys", []) or []
            cited_triples    = rec.get("cited_triples", []) or []

            # Normalize abstracts triples to (bib_key, title, abstract) tuples
            abstracts_norm = []
            for t in cited_triples:
                # tolerate list/tuple/dict
                if isinstance(t, dict):
                    abstracts_norm.append((
                        t.get("0") or t.get("bib_key") or "",
                        t.get("1") or t.get("title")   or "",
                        t.get("2") or t.get("abstract") or "",
                    ))
                elif isinstance(t, (list, tuple)):
                    bk  = t[0] if len(t) > 0 else ""
                    ttl = t[1] if len(t) > 1 else ""
                    abs_ = t[2] if len(t) > 2 else ""
                    abstracts_norm.append((bk or "", ttl or "", abs_ or ""))
                else:
                    # unknown shape; skip
                    continue

            # k inferred from how many neighbours were saved (fallback 2)
            k_prev = len(prev_paras)
            k_next = len(next_paras)
            k = max(k_prev, k_next) if (k_prev or k_next) else 2

            # Build prompt using your existing helper
            prompt = _build_prompt(
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
            )

            # Turn figure latex paths into absolute paths and then into base64 embeddings
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
                    print(f"[WARN] Failed to resolve figure path '{latex_path}' "
                          f"for {paper_arxiv_id}: {e}")

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

# Instead of generate the embedding, we generate the top labels of images
def _data_extraction_non_vlm(jsonl_file_path, model_name="openai/clip-vit-base-patch32"):


    outputs: List[Dict[str, Any]] = []
    # Use the same default download path as in Args
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

            # Required core fields with safe fallbacks
            paragraph_key_id = rec.get("paragraph_key_id")
            paper_arxiv_id   = rec.get("paper_arxiv_id", "")
            paper_section    = rec.get("paper_section", "")
            paragraph_id_loc = rec.get("paragraph_id_local")
            original_content = rec.get("original_content", "") or ""
            num_char         = int(rec.get("num_char") or len(original_content))
            title            = rec.get("title", "") or ""
            abstract         = rec.get("abstract", "") or ""
            prev_paras       = rec.get("prev_paras", []) or []
            next_paras       = rec.get("next_paras", []) or []
            figures          = rec.get("figures", []) or []
            tables           = rec.get("tables", []) or []
            fig_labels       = rec.get("fig_labels", []) or []
            table_labels     = rec.get("table_labels", []) or []
            bib_keys         = rec.get("bib_keys", []) or []
            cited_triples    = rec.get("cited_triples", []) or []

            # Normalize abstracts triples to (bib_key, title, abstract) tuples
            abstracts_norm = []
            for t in cited_triples:
                # tolerate list/tuple/dict
                if isinstance(t, dict):
                    abstracts_norm.append((
                        t.get("0") or t.get("bib_key") or "",
                        t.get("1") or t.get("title")   or "",
                        t.get("2") or t.get("abstract") or "",
                    ))
                elif isinstance(t, (list, tuple)):
                    bk  = t[0] if len(t) > 0 else ""
                    ttl = t[1] if len(t) > 1 else ""
                    abs_ = t[2] if len(t) > 2 else ""
                    abstracts_norm.append((bk or "", ttl or "", abs_ or ""))
                else:
                    # unknown shape; skip
                    continue

            # k inferred from how many neighbours were saved (fallback 2)
            k_prev = len(prev_paras)
            k_next = len(next_paras)
            k = max(k_prev, k_next) if (k_prev or k_next) else 2

            # Build prompt using your existing helper
            prompt = _build_prompt(
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
            )

            # Turn figure latex paths into absolute paths and then into base64 embeddings
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
                    print(f"[WARN] Failed to resolve figure path '{latex_path}' "
                          f"for {paper_arxiv_id}: {e}")

            # image_embeddings = _figure_paths_to_embeddings(figure_paths=figure_paths, model_path=model_name)

            image_tags_projections = visual_adaptation(image_paths = figure_paths)

            # Extract the tags
            image_tag_list = []
            for tag, projection in image_tags_projections:
                image_tag_list.append(tag)

            print(f"Image tag list: {image_tag_list}")


# def visual_adaptation(image_paths, model_name = "openai/clip-vit-base-patch32", projection_dimension = 4096):

            outputs.append({
                "paragraph_key_id": paragraph_key_id,
                "paper_arxiv_id": paper_arxiv_id,
                "prompt": prompt,
                "original_content": original_content,
                "image_tag_list": image_tag_list,
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

def load_model(model_name):
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
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
    candidates: Sequence[str],
    tag_templates: Sequence[str],
    topk: int = 5,
) -> List[str]:
    texts = [tpl.format(tag) for tag in candidates for tpl in tag_templates]
    if not texts:
        return []
    inputs = clip_processor(text=texts, padding=True, return_tensors="pt")
    text_feats = clip_model.get_text_features(**inputs)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    sims = (image_feats @ text_feats.T).squeeze(0)               # [len(cands)*len(tpls)]

    T = len(tag_templates)
    sims_by_tag = sims.view(len(candidates), T)
    # pick best template for each tag
    best_vals, best_tpl_idx = sims_by_tag.max(dim=1)             # [len(candidates)]
    # rank tags by their best template
    k = min(topk, len(candidates))
    tag_order = torch.topk(best_vals, k=k).indices.tolist()
    return [tag_templates[best_tpl_idx[i]].format(candidates[i]) for i in tag_order]



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


def visual_adaptation(image_paths, model_name = "openai/clip-vit-base-patch32", projection_dimension = 4096):

    pairs = []
    images = []

    for p in image_paths:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] Path not found: {path}")
            # resolved.append(None)
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

        # resolved.append(path if img is not None else None)
        images.append(img)


    for image in images:
        # If the image does not exists, simply append [None, None]

        if not image:
            pairs.append([None, None])
            continue

        # Load the model 
        clip_model, clip_processor = load_model(model_name)
        img_feats = clip_image_features(clip_model, clip_processor, image)
        print("CLIP image features:", tuple(img_feats.shape))
        top_tags = clip_rank_tags(clip_model=clip_model, clip_processor=clip_processor, image_feats=img_feats, candidates=CANDIDATE_TAGS, tag_templates=TAG_TEMPLATES, topk=TOPK_TAGS)
        print("Top CLIP tags:", top_tags)
        
        projected = None
        if projection_dimension:
            projected = project_to_embedding_space(img_feats, target_dim=4096)
        pairs.append([top_tags, projected])

    return pairs



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
    bib_keys: List[str]
) -> str:

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


