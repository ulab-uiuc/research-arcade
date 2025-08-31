import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict
from dotenv import load_dotenv
from graph_constructor.utils import figure_latex_path_to_path
import shutil
import re
import unicodedata
from tasks.utils import paragraph_ref_to_global_ref
import pandas as pd

import json
from pathlib import Path

import psycopg2
import psycopg2.extras
from PIL import Image
import base64
import sys
import fitz
import io
from openai import OpenAI
from pathlib import Path

load_dotenv()

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

def llm_generate(prompt, image_embeddings, model_name):

    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = API_KEY
    )

    content = []
    if image_embeddings:
        for image_embedding in image_embeddings:
            content.append({ "type": "image_url", "image_url": { "url": f"data:image/png;base64,{image_embedding}" } },)
    content.append({ "type": "text", "text": prompt})
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": content
            }
            ],
        temperature=1.00,
        top_p=0.01,
        max_tokens=1024,
        stream=True
    )

    ans = ""
    
    for chunk in completion:
      if chunk.choices[0].delta.content is not None:
        ans = ans + chunk.choices[0].delta.content
        # print(chunk.choices[0].delta.content, end="")
    # print("")
    return ans

# ---------- Utilities ----------
def _derived_bib_key(title: str) -> str:
    """
    Derive a simple bib key from title when a 'real' key is unknown.
    Example: "Graph Neural Networks for X" -> gnn_for_x_1stwordyear (approx).
    Keep it conservative and ASCII-only.
    """
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    words = [w for w in t.split() if w]
    base = "_".join(words[:4]) if words else "unnamed"
    return f"{base}_tmpkey"



def _data_fetching(paragraph_key_ids, data_path, k_neighbour):
    """
    Given paragraph primary-key IDs (from table `paragraphs.id`), fetch and serialize,
    for each paragraph, the following into a JSONL at `data_path`:

      - figures: List[Dict] with id/label/caption/path
      - tables:  List[Dict] with id/label/text
      - abstracts + title of cited papers: List[Tuple[bib_key, title, abstract]]
      - prev_paras: List[str]
      - next_paras: List[str]
      - paper_section: str
      - num_char: int              (length of the original paragraph content)
      - title: str                 (paper title)
      - abstract: str              (paper abstract)
      - fig_labels: List[str]      (parsed from \\label{...} in figure labels)
      - table_labels: List[str]    (parsed from \\label{...} in table labels)
      - bib_keys: List[str]        (keys cited in this paragraph)
      - original_content: str      (the paragraph text itself)
      - paragraph_key_id: int
      - paper_arxiv_id: str
      - paragraph_id_local: int
    """
    # Ensure target directory exists
    Path(data_path).parent.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="cl195",
        password=PASSWORD,
        port="5433",
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Open (overwrite) JSONL for this run
    with open(data_path, "w", encoding="utf-8") as fp:
        for paragraph_key_id in paragraph_key_ids:
            # 1) Resolve (paper_arxiv_id, paragraph_id_local, paper_section) from PK
            got = _fetch_arxiv_id_paragraph_id_section_name(cur, paragraph_key_id)
            if not got:
                # Skip missing PKs; you can raise if you prefer strictness
                print(f"[WARN] paragraph id {paragraph_key_id} not found in `paragraphs`.")
                continue
            paper_arxiv_id, paragraph_id_local, paper_section = got

            # 2) Fetch the paragraph content (also returns arxiv id and local id)
            _arxiv_id, _local_id, paragraph_content = _fetch_context_for_paragraph(
                cur, arxiv_id=paper_arxiv_id, paragraph_id=paragraph_id_local, paper_section=paper_section
            )
            # 3) Adjacent context
            prev_paras, next_paras = _fetch_adjacent_paragraphs(
                cur, paper_arxiv_id, paper_section, paragraph_id_local, k=k_neighbour  # you can parameterize k if desired
            )

            # 4) Figures
            fig_ids = _fetch_global_refs_for_paragraph(
                cur=cur,
                paper_arxiv_id=paper_arxiv_id,
                paper_section=paper_section,
                paragraph_id=paragraph_id_local,
                ref_type="figure",
            )
            figures = _fetch_figures(cur, fig_ids)
            # Parse labels like \label{fig:foo} -> "fig:foo"
            fig_labels = []
            for f in figures:
                try:
                    lbl = collect_fig_labels(f.get("label", ""))
                except Exception:
                    lbl = None
                if lbl:
                    fig_labels.append(lbl)

            # 5) Tables
            tab_ids = _fetch_global_refs_for_paragraph(
                cur=cur,
                paper_arxiv_id=paper_arxiv_id,
                paper_section=paper_section,
                paragraph_id=paragraph_id_local,
                ref_type="table",
            )
            tables = _fetch_tables(cur, tab_ids)
            table_labels = []
            for t in tables:
                try:
                    tlbl = collect_fig_labels(t.get("label", ""))
                except Exception:
                    tlbl = None
                if tlbl:
                    table_labels.append(tlbl)

            # 6) Cited bib keys for this paragraph
            bib_keys = _fetch_cited_bib_key(
                cur=cur,
                citing_arxiv_id=paper_arxiv_id,
                paper_section=paper_section,
                paragraph_id=paragraph_id_local,
            ) or []

            # 7) Titles + (potential) arxiv ids for cited entries
            title_id_pairs = _fetch_titles(cur, paper_arxiv_id, bib_keys)
            # title_id_pairs: List[(bib_key, bib_title, cited_arxiv_id)]
            cited_arxiv_ids = [row[2] for row in title_id_pairs]

            # 8) Abstracts for cited arxiv ids (empty string if not found)
            cited_abstracts = _fetch_abstracts(cur, cited_arxiv_ids)  # List[str]
            # Build triples aligned to title_id_pairs
            cited_triples = []
            for (bib_key, bib_title, _arx), abs_txt in zip(title_id_pairs, cited_abstracts):
                cited_triples.append((bib_key, bib_title, abs_txt or ""))
        
            # 9) Paper title/abstract
            paper_title, paper_abs = _fetch_paper_title_abstract(cur=cur, arxiv_id=paper_arxiv_id) or ("", "")

            # 10) Compose record
            record = {
                "paragraph_key_id": int(paragraph_key_id),
                "paper_arxiv_id": paper_arxiv_id,
                "paragraph_id_local": int(paragraph_id_local),
                "paper_section": paper_section,
                "original_content": paragraph_content or "",
                "num_char": len(paragraph_content or ""),
                "title": paper_title or "",
                "abstract": paper_abs or "",
                "prev_paras": prev_paras,
                "next_paras": next_paras,
                "figures": figures,          # [{id,label,caption,path}, ...]
                "tables": tables,            # [{id,label,text}, ...]
                "fig_labels": fig_labels,    # ["fig:...", ...]
                "table_labels": table_labels,# ["tab:...", ...]
                "bib_keys": bib_keys,        # ["smith2024", ...]
                "cited_triples": cited_triples,  # [(bib_key, title, abstract), ...]
            }

            # 11) Write one line per paragraph_key_id
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    cur.close()
    conn.close()

    



def _format_block(label: str, lines: List[str]) -> str:
    if not lines:
        return ""
    header = f"{label}:\n"
    return header + "\n".join(lines)

# Here, we need to use pandas
def _fetch_context_for_paragraph(cur, arxiv_id, paragraph_id, paper_section):

    cur.execute(
        """
        SELECT paper_arxiv_id, paragraph_id, content
        FROM paragraphs
        WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s 
        """,
        (arxiv_id, paper_section, paragraph_id),
    )
    # The three inputs ensures the uniqueness of the row
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Arxiv id {arxiv_id} + Paragraph id {paragraph_id} not found.")
    return row  # (paper_arxiv_id, paper_section, paragraph_id_local)


def _fetch_adjacent_paragraphs(cur, paper_arxiv_id: str, paper_section: str, pivot_local_id: int, k: int) -> Tuple[List[str], List[str]]:
    """
    Returns (prev, next) paragraphs as lists of text.
    prev: earlier paragraphs in descending-to-ascending order (we'll re-order to natural ascending).
    next: following paragraphs in natural ascending order.
    """
    prev_ids = list(range(pivot_local_id - k, pivot_local_id))
    next_ids = list(range(pivot_local_id + 1, pivot_local_id + 1 + k))

    prev_texts = []
    for pid in prev_ids:
        if pid <= 0:
            continue
        cur.execute(
            """
            SELECT content FROM paragraphs
            WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s
            """,
            (paper_arxiv_id, paper_section, pid),
        )
        row = cur.fetchone()
        if row and row[0]:
            prev_texts.append(row[0])

    next_texts = []
    for nid in next_ids:
        cur.execute(
            """
            SELECT content FROM paragraphs
            WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s
            """,
            (paper_arxiv_id, paper_section, nid),
        )
        row = cur.fetchone()
        if row and row[0]:
            next_texts.append(row[0])

    # Ensure prev is in natural ascending order (older to newer)
    return (prev_texts, next_texts)


def _fetch_global_refs_for_paragraph(cur, paper_arxiv_id: str, paragraph_id: int, paper_section: str , ref_type: str) -> List[int]:
    """
    Uses your helper mapping. If you already have paragraph_ref_to_global_ref, call it;
    otherwise select from a junction table like paragraph_references(paragraph_id, reference_type, global_id).
    """

    # cur.execute(
    #     """
    #     SELECT id
    #     FROM paragraph_references
    #     WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s AND reference_type = %s;
    #     """,
    #     (paper_arxiv_id, paper_section, paragraph_id, ref_type),
    # )

    # rows = cur.fetchall()

    # for row in rows:
    #     paragraph_id = row[0]
    global_vals = paragraph_ref_to_global_ref(arxiv_id = paper_arxiv_id, paper_section=paper_section,paragraph_id=paragraph_id, ref_type=ref_type)
    
    # print(f"global_vals: {global_vals}")
    
    return global_vals


def _fetch_figures(cur, figure_ids: List[int]) -> List[Dict]:
    
    if not figure_ids:
        return []
    cur.execute(
        """
        SELECT id, label, caption, path
        FROM figures
        WHERE id = ANY(%s)
        """,
        (figure_ids,),
    )
    rows = cur.fetchall()
    figures = []
    for rid, label, caption, path in rows:
        figures.append({
            "id": rid,
            "label": label,
            "caption": caption or "",
            "path": path or ""
        })
    return figures


def _fetch_tables(cur, table_ids: List[int]) -> List[Dict]:
    if not table_ids:
        return []
    cur.execute(
        """
        SELECT id, label, table_text
        FROM tables
        WHERE id = ANY(%s)
        """,
        (table_ids,),
    )
    rows = cur.fetchall()
    tables = []
    for rid, label, ttext in rows:
        tables.append({
            "id": rid,
            "label": label,
            "text": ttext or ""
        })
    return tables


def _fetch_cited_bib_key(cur, citing_arxiv_id: str, paper_section: str, paragraph_id: int) -> List[Tuple[str, Optional[str]]]:
    """
    Returns list of (bib_key_or_none).
    If you already have paragraph_to_global_citation(paragraph_id) in Python, call that instead.
    """
    cur.execute(
        """
        SELECT bib_key
        FROM paragraph_citations
        WHERE citing_arxiv_id = %s AND paper_section= %s AND paragraph_id = %s
        """,
        (citing_arxiv_id, paper_section, paragraph_id),
    )
    return [(r[0]) for r in cur.fetchall()]

def _fetch_titles(cur, citing_arxiv_id, cited_bib_keys):

    """
    Return bib_key->bib_title pairs
    """
    keys = [str(k) for k in (cited_bib_keys or [])]
    if not keys:
        return {}  # nothing to fetch
    print(f"keys for query: {keys}")
    sql = """
    SELECT bib_key, bib_title, cited_arxiv_id
    FROM citations
    WHERE citing_arxiv_id = %s
      AND bib_key = ANY(%s::text[])
    """
    cur.execute(sql, (str(citing_arxiv_id), keys))
    return list(cur.fetchall())



def _fetch_abstracts(cur, arxiv_ids: List[str]) -> List[Tuple[str, str, str]]:
    """
    Returns list of (bib_key, title, abstract).
    We try exact id, and then the versionless id (split at 'v').
    """
    results = []
    for raw in arxiv_ids:
        if not raw:
            results.append("")
            continue
        # extract id if it's a URL
        if "/" in raw:
            arx = raw.strip().split("/")[-1]
        else:
            arx = raw.strip()

        versionless = arx.split("v", 1)[0] if "v" in arx else arx
        
        cur.execute("SELECT title, abstract FROM papers WHERE arxiv_id = %s", (arx,))
        row = cur.fetchone()
        if not row:
            cur.execute("SELECT title, abstract FROM papers WHERE arxiv_id = %s", (versionless,))
            row = cur.fetchone()

        if row:
            title, abstract = row
            print(f"row: {row}")
            results.append((abstract or ""))
        else:
            results.append("")
    return results

def _fetch_paper_title_abstract(cur, arxiv_id):
    """
    Given paper arxiv id, fetch its abstract for LLM reference
    """

    cur.execute("""
    SELECT title, abstract
    FROM papers
    WHERE arxiv_id = %s
    """, (arxiv_id,))
    row = cur.fetchone()
    if row:
        return row[0], row[1]
    else:
        return None


def _fetch_arxiv_id_paragraph_id_section_name(cur, id):
    """
    Given the id (PRIMARY KEY) of a paragraph, return its (paper_arxiv_id, paragraph_id).
    """
    cur.execute(
        """
        SELECT paper_arxiv_id, paragraph_id, paper_section
        FROM paragraphs
        WHERE id = %s
        """,
        (id,)
    )
    row = cur.fetchone()
    if row:
        return row[0], row[1], row[2]
    else:
        return None  # or raise an error, depending on your needs




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
    # Sufficient information needs to be provided for effective learning
#     prompt = f"""
# You are given the following inputs for reconstructing a missing paragraph in a research paper.

# Title of the paper:
# {paper_title}

# Abstract of the paper:
# {abstract}

# Section name of the paragraph:
# {paper_section}

# Figure (optional):
# {figure_block if figure_block else "(none)"}

# If figures are present, the images of them will be given following the order of figure blocks.

# Table (optional):
# {table_block if table_block else "(none)"}

# Title and abstract(s) of cited paper(s) if available:
# {abstracts_block if abstracts_block else "(none)"}

# {k}-Most Adjacent Paragraphs (context):
# {adjacent_paragraphs_block}

# Number of letters in the original paragraph:
# {num_char}

# # Task
# Write exactly one LaTeX-formatted paragraph that naturally fits between the adjacent paragraphs.

# # Requirements
# - If a figure is provided, explicitly reference it with \\label{{...}} using the label provided; remember to insert ALL THE FIGURES into the generated paragraph if provided.
# - If a table is provided, explicitly reference it with: \\label{{...}} using the label provided; remember to insert ALL THE TABLES into the generated paragraph if provided.
# - Incorporate all the citation information, provided the abstract(s) or title, and cite it with \\cite{{}}. Use the provided BibTeX key(s) if present; otherwise, use a stable placeholder key derived from title (e.g., {{derived_bib_key}}).  
# - Ensure the paragraph logically continues from and sets up the surrounding {k} adjacent paragraph(s).
# - Ensure that the generated paragraph has approximately the same length as the orginal answer.
# - Style: objective, concise, academic tone.
# - Formatting: produce a single LaTeX paragraph only (no section headers, lists, environments; math only if essential).

# # Output
# Return only the LaTeX paragraph text, nothing else.
# """.strip()
#     prompt = f"""
# You are given the following inputs for reconstructing a missing paragraph in a research paper.

# Title: {paper_title}
# Abstract: {abstract}
# Section name: {paper_section}

# Figure block (optional): {figure_block if figure_block else "(none)"}
# Table block (optional): {table_block if table_block else "(none)"}
# Cited paper titles/abstracts (optional): {abstracts_block if abstracts_block else "(none)"}

# k-most adjacent paragraphs (context): {adjacent_paragraphs_block}
# Target length (characters): {num_char}

# # Task
# Write exactly one LaTeX-formatted paragraph that naturally fits between the adjacent paragraphs.

# # HARD REQUIREMENTS (must all be satisfied)
# 1) If fig_labels is non-empty, include each label **exactly once** using \\label{{<label>}} in the paragraph text (e.g., "see Fig.~\\label{{fig:framework}}").
# 2) If table_labels is non-empty, include each label **exactly once** using \\label{{<label>}}.
# 3) If bib_keys is non-empty, **cite all of them** (you may group keys in a single \\cite{{...}}).
#    - If allow_derived_bib_keys=true and a cited item lacks a key, derive a stable placeholder from its title: lowercase, keep a-z0-9 and hyphens only.
#    - Do not invent any other keys.
# 4) Maintain objective, concise academic tone; ensure logical continuity with the provided context.
# 5) Length ≈ {num_char} (±15%). Produce exactly one paragraph (no lists/headers/environments).

# # ORDERING
# - When mentioning multiple figures/tables, follow the order in fig_labels/table_labels.

# # Output
# Return only the LaTeX paragraph text. No explanations, no markdown, no extra lines.
# """.strip()
    
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


def _figure_paths_to_embeddings(figure_paths):
    if not figure_paths:
        return []

    embeddings = []
    allowed = {".jpg", ".jpeg", ".png", ".pdf"}

    for fs_path in figure_paths:
        if not os.path.isfile(fs_path):
            print(f"[WARN] Not a file: {fs_path}")
            continue

        ext = Path(fs_path).suffix.lower()
        if ext not in allowed:
            print(f"[WARN] Skipping unsupported extension ({ext}): {fs_path}")
            continue

        try:
            if ext == ".pdf":
                # Convert each PDF page to JPEG and base64-encode
                with fitz.open(fs_path) as doc:
                    for page in doc:
                        pix = page.get_pixmap()               # render page
                        if pix.alpha:                          # drop alpha if present
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        jpeg_bytes = pix.tobytes("jpg")        # encode as JPEG bytes
                        embeddings.append(base64.b64encode(jpeg_bytes).decode("utf-8"))
            else:
                # Image file: read and base64-encode
                with open(fs_path, "rb") as f:
                    embeddings.append(base64.b64encode(f.read()).decode("utf-8"))

        except Exception as e:
            print(f"[ERROR] Failed to process {fs_path}: {e}")

    return embeddings


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

    

def paragraph_generation(args: Args) -> List[Dict[str, str]]:
    """
    For each paragraph_id in args.paragraph_ids:
      - gather figures/tables/cited abstracts
      - collect k-neighbour context
      - build a single prompt
      - optionally run an LLM via args.llm_generate(prompt)
    Returns a list of dicts: { "paragraph_id": ..., "prompt": ..., "llm_output": ... }
    """
    conn = psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="cl195",
        password=PASSWORD,
        port="5433",
    )

    conn.autocommit = True
    cur = conn.cursor()

    paper_arxiv_id = None
    
    outputs: List[Dict[str, str]] = []

    model_name = args.model_name

    for pid in args.paragraph_ids:
        paper_arxiv_id, paragraph_id, paper_section = _fetch_arxiv_id_paragraph_id_section_name(cur, pid)
        # context anchor
        paper_arxiv_id, pivot_local_id, paragraph_content = _fetch_context_for_paragraph(cur, arxiv_id = paper_arxiv_id, paragraph_id = paragraph_id, paper_section = paper_section)


        # figures
        figures = []
        if args.figure_available:
            fig_ids = _fetch_global_refs_for_paragraph(cur = cur, paper_arxiv_id = paper_arxiv_id, paper_section = paper_section, paragraph_id = paragraph_id, ref_type = "figure")
            #TODO remove it
            # Ensure that each figure is 
            # print(f"paper_arxiv_id: {paper_arxiv_id}")
            # print(f"paper_section: {paper_section}")
            # print(f"paragraph_id: {paragraph_id}")
            # print(f"Number of figure ids collected: {len(fig_ids)}")
            # print(f"fig_ids: {fig_ids}")
            figures = _fetch_figures(cur, fig_ids)
            print(f"figures: {figures}")
        
        # TODO: deal with this part
        figure_labels_raw = [figure["label"] for figure in figures]

        fig_labels = [collect_fig_labels(figure_label) for figure_label in figure_labels_raw]
        print(f"Collected figure labels: {fig_labels}")

        if args.table_available:
            tab_ids = _fetch_global_refs_for_paragraph(cur = cur, paper_arxiv_id = paper_arxiv_id, paper_section = paper_section, paragraph_id = paragraph_id, ref_type = "table")
            tables = _fetch_tables(cur, tab_ids)
            print(f"tables: {tables}")

        table_labels_raw = [table["label"] for table in tables]
    
        table_labels = [collect_fig_labels(table_label) for table_label in table_labels_raw]
        print(f"Collected table labels: {table_labels}")


        # print(f"Fetched figures: {figures}")
        # print(tables)
        # print(f"Fetched tables: {tables}")
        # citations → abstracts (+bib keys)
        cited_bib_keys = _fetch_cited_bib_key(cur=cur, citing_arxiv_id=paper_arxiv_id, paper_section=paper_section, paragraph_id = paragraph_id)  # (arxiv_or_url, bib_key_or_none)
        title_id_pairs = _fetch_titles(cur, paper_arxiv_id, cited_bib_keys)
        
        print(f"fetched title_id_pairs: {title_id_pairs}")
        
        
        cited_ids = [title_id_pair[0] for title_id_pair in title_id_pairs]
        # Then we fetch the arxiv ids of cited papers if exits

        abstracts = _fetch_abstracts(cur, cited_ids)
        print(f"fetched abstracts: {abstracts}")

        # Abstract of the paper
        paper_title, paper_abstract = _fetch_paper_title_abstract(cur=cur, arxiv_id=paper_arxiv_id)

        # adjacent paragraphs
        prev_paras, next_paras = _fetch_adjacent_paragraphs(
            cur, paper_arxiv_id, paper_section, pivot_local_id, args.k_neighbour
        )

        # print(f"Previous paragraphs: {prev_paras}")
        # print(f"Next paragraphs: {next_paras}")
        abstract_triple = []

        for i in range(len(title_id_pairs)):
            abstract_triple.append([title_id_pairs[i][0], title_id_pairs[i][1], abstracts[i]])

        print(f"triple: {abstract_triple}")

        num_char = len(paragraph_content)
        prompt = _build_prompt(
            k=args.k_neighbour,
            figures=figures,
            tables=tables,
            abstracts=abstract_triple,
            prev_paras=prev_paras,
            next_paras=next_paras,
            paper_section=paper_section,
            num_char=num_char,
            title=paper_title,
            abstract=paper_abstract,
            fig_labels=fig_labels,
            table_labels=table_labels,
            bib_keys=cited_bib_keys,
        )

        figure_paths = []
        download_path = args.download_path  
        #TODO remove it
        # print(f"Number of figures collected {len(figures)}")
        for figure in figures:
            latex_path = figure["path"]
            figure_path = figure_latex_path_to_path(path=download_path, arxiv_id = paper_arxiv_id, latex_path = latex_path)
            figure_paths.append(figure_path)


        image_embeddings = _figure_paths_to_embeddings(figure_paths)

        #TODO remove it
        print(f"Number of image embeddings collected: {len(image_embeddings)}")

        # sys.exit()
        
        # Dont need to print embeddings since we cannot understand it
        llm_output = llm_generate(prompt=prompt, image_embeddings=image_embeddings, model_name=model_name)

        outputs.append({
            "paragraph_id": str(pid),
            "prompt": prompt,
            "original content": paragraph_content,
            "llm_output": llm_output or ""
        })

    cur.close()
    conn.close()
    return outputs


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


