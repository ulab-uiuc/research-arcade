import os
import re
from typing import Dict, List, Tuple, Optional, Callable

import psycopg2
import psycopg2.extras

# ==============================
# Config
# ==============================
PASSWORD = os.getenv("PGPASSWORD", "REPLACE_ME")
OPENAI = False  # set True if using OpenAI adapter below


# ==============================
# LLM Adapters (plug your own)
# ==============================
def nvidia_llm_adapter(model: str = "meta/llama-3.1-405b-instruct", temperature: float = 0.2, top_p: float = 0.7, stream: bool = False):
    """
    NVIDIA integrate endpoint (text-only). For images, you’ll need an API that supports image parts.
    """
    from openai import OpenAI  # NVIDIA integrate uses OpenAI-compatible client
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("API_KEY", "REPLACE_ME"),
    )

    def _generate(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=1024,
            stream=stream,
        )
        if stream:
            chunks = []
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    chunks.append(delta)
            return "".join(chunks).strip()
        else:
            return resp.choices[0].message.content.strip()

    return _generate


def openai_llm_adapter(model: str = "gpt-4o-mini", temperature: float = 0.2):
    """
    OpenAI adapter (text-only here). If you need vision, use the images part in the messages payload.
    """
    from openai import OpenAI
    client = OpenAI()

    def _generate(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You strictly follow the output format."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    return _generate


# pick one
LLM_GENERATE: Callable[[str], str] = openai_llm_adapter() if OPENAI else nvidia_llm_adapter()


# ==============================
# Prompt helper
# ==============================
PROMPT_TEMPLATE = """You are given one reference extracted from a research paper (either a Figure, Table, or Citation), and a numbered list of the paper’s paragraphs.

Reference:
{reference_block}

Paragraphs:
{paragraphs_block}

Instructions:
- Read the reference carefully and compare it with each paragraph.
- Select ALL paragraphs where this reference most likely belongs (i.e., is introduced, described, or directly discussed).
- Output ONLY the paragraph indexes, comma-separated (e.g., 2,4,7).
- Do not include any words, explanations, or extra symbols.

Final Answer (only indexes, comma-separated):
""".strip()


def _build_paragraphs_block(paragraphs: List[str]) -> Tuple[str, Dict[int, int]]:
    """
    Returns paragraphs_block string and a mapping from global paragraph_id -> local index (1..N).
    (Caller prepares the (id, text) list and passes only text here.)
    """
    lines = []
    paragraph_to_idx: Dict[int, int] = {}
    for i, (pid, text) in enumerate(paragraphs, start=1):
        paragraph_to_idx[pid] = i
        # normalize whitespace, keep content readable
        t = (text or "").strip().replace("\n", " ")
        lines.append(f"{i}. {t}")
    return "\n".join(lines), paragraph_to_idx


def _derive_tmp_bib_key(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    words = [w for w in t.split() if w]
    base = "_".join(words[:4]) if words else "unnamed"
    return f"{base}_tmpkey"


# ==============================
# DB helpers (adjust table/column names if needed)
# ==============================
def _fetch_paragraphs_by_paper(cur, paper_arxiv_id: str) -> List[Tuple[int, str]]:
    cur.execute(
        """
        SELECT id, content
        FROM paragraphs
        WHERE paper_arxiv_id = %s
        ORDER BY paper_section, paragraph_id
        """,
        (paper_arxiv_id,),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def _fetch_figure(cur, global_id: int) -> Optional[dict]:
    cur.execute(
        """
        SELECT id, reference_label, caption, file_path, paper_arxiv_id
        FROM figures
        WHERE id = %s
        """,
        (global_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "label": row[1],
        "caption": row[2] or "",
        "file_path": row[3] or "",
        "paper_arxiv_id": row[4],
    }


def _fetch_table(cur, global_id: int) -> Optional[dict]:
    cur.execute(
        """
        SELECT id, reference_label, table_text, paper_arxiv_id
        FROM tables
        WHERE id = %s
        """,
        (global_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "label": row[1],
        "text": row[2] or "",
        "paper_arxiv_id": row[3],
    }


def _fetch_citation(cur, citation_id: int) -> Optional[dict]:
    """
    One citation row that points to a cited arXiv id and the host paper.
    """
    cur.execute(
        """
        SELECT cited_arxiv_id, paper_arxiv_id
        FROM paragraph_citations
        WHERE id = %s
        """,
        (citation_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    cited_arxiv_id, host_arxiv_id = row
    # fetch title/abstract of cited paper
    cur.execute(
        "SELECT title, abstract FROM papers WHERE arxiv_id = %s",
        (cited_arxiv_id,),
    )
    pr = cur.fetchone()
    title, abstract = (pr[0] if pr else ""), (pr[1] if pr else "")
    return {
        "cited_arxiv_id": cited_arxiv_id,
        "paper_arxiv_id": host_arxiv_id,
        "title": title or "",
        "abstract": abstract or "",
    }


def _ground_truth_paragraph_ids_for_ref(cur, ref_type: str, global_id: int) -> List[int]:
    """
    Junction tables:
      - paragraph_references(paragraph_id, reference_type, global_id)  -- for figures & tables
      - paragraph_citations_links(paragraph_id, citation_id)          -- for specific citations
    If you store citation placement in paragraph_citations directly (with paragraph_id), use that.
    """
    if ref_type in ("figure", "table"):
        cur.execute(
            """
            SELECT paragraph_id
            FROM paragraph_references
            WHERE reference_type = %s AND global_id = %s
            """,
            (ref_type, global_id),
        )
        return [r[0] for r in cur.fetchall()]
    elif ref_type == "citation":
        # if your schema stores per-citation placement:
        cur.execute(
            """
            SELECT paragraph_id
            FROM paragraph_citations_links
            WHERE citation_id = %s
            """,
            (global_id,),
        )
        return [r[0] for r in cur.fetchall()]
    return []


# ==============================
# Core: ref_insertion
# ==============================
def ref_insertion(args) -> List[dict]:
    """
    Args requirements:
      - args.data_type: "figure" | "table" | "citation"
      - args.data_ids: List[int]   # local IDs of the items you want to locate
      - args.paragraph_ref_id_to_global_ref: Callable[[int, str], int] or None
            maps local item id -> global id. If None, we assume the provided ids are already global.
      - (optional) args.model_generate: Callable[[str], str]  # override LLM_GENERATE
    Returns: list of dicts with fields:
      - data_id
      - prompt
      - answer_raw (model output)
      - pred_indices (List[int])  # paragraph local indices (1..N) predicted by model
      - pred_paragraph_ids (List[int])  # mapped back to paragraph ids
      - gt_paragraph_ids (List[int])
      - precision, recall, f1 (if you compute here)
    """
    data_type = args.data_type.lower()
    assert data_type in {"figure", "table", "citation"}, f"Unsupported data_type: {data_type}"

    data_ids: List[int] = list(args.data_ids)
    id_mapper: Optional[Callable[[int, str], int]] = getattr(args, "paragraph_ref_id_to_global_ref", None)
    generate = getattr(args, "model_generate", None) or LLM_GENERATE

    conn = psycopg2.connect(
        host="localhost", dbname="postgres", user="postgres", password=PASSWORD, port="5432"
    )
    conn.autocommit = True
    cur = conn.cursor()

    results = []

    for data_id in data_ids:
        # 1) resolve to global id
        global_id = id_mapper(data_id, data_type) if id_mapper else data_id

        # 2) fetch reference payload + host paper arxiv id
        reference_block = ""
        paper_arxiv_id = None

        if data_type == "figure":
            fg = _fetch_figure(cur, global_id)
            if not fg:
                results.append({"data_id": data_id, "error": "figure not found"})
                continue
            paper_arxiv_id = fg["paper_arxiv_id"]
            reference_block = (
                f"type: figure\n"
                f"label: {fg['label']}\n"
                f"caption: {fg['caption']}\n"
                f"path: {fg['file_path']}".strip()
            )

        elif data_type == "table":
            tb = _fetch_table(cur, global_id)
            if not tb:
                results.append({"data_id": data_id, "error": "table not found"})
                continue
            paper_arxiv_id = tb["paper_arxiv_id"]
            reference_block = (
                f"type: table\n"
                f"label: {tb['label']}\n"
                f"text: {tb['text']}".strip()
            )

        else:  # citation
            ct = _fetch_citation(cur, global_id)
            if not ct:
                results.append({"data_id": data_id, "error": "citation not found"})
                continue
            paper_arxiv_id = ct["paper_arxiv_id"]
            # derive key if you want, not needed for this task
            reference_block = (
                f"type: citation\n"
                f"cited_title: {ct['title']}\n"
                f"abstract: {ct['abstract']}\n"
                f"arxiv_id: {ct['cited_arxiv_id']}".strip()
            )

        # 3) fetch all paragraphs of that paper
        para_rows = _fetch_paragraphs_by_paper(cur, paper_arxiv_id)
        if not para_rows:
            results.append({"data_id": data_id, "error": "no paragraphs for host paper"})
            continue
        paragraphs_block, paragraph_to_idx = _build_paragraphs_block(para_rows)

        # 4) prepare prompt
        prompt = PROMPT_TEMPLATE.format(
            reference_block=reference_block,
            paragraphs_block=paragraphs_block,
        )

        # 5) model call
        answer_raw = generate(prompt)

        # 6) parse model output (keep only integers 1..N)
        N = len(para_rows)
        chosen_indices: List[int] = []
        if answer_raw:
            # accept patterns like "2,4, 7" or "2 , 4 ,7"
            tokens = re.split(r"[,\s]+", answer_raw.strip())
            for tok in tokens:
                if tok.isdigit():
                    idx = int(tok)
                    if 1 <= idx <= N:
                        chosen_indices.append(idx)
        chosen_indices = sorted(set(chosen_indices))

        # 7) map local indices -> paragraph_ids
        idx_to_pid = {i: pid for i, (pid, _) in enumerate(para_rows, start=1)}
        pred_paragraph_ids = [idx_to_pid[i] for i in chosen_indices]

        # 8) ground truth for evaluation
        gt_paragraph_ids = _ground_truth_paragraph_ids_for_ref(cur, data_type if data_type != "citation" else "citation", global_id)

        # Optional: compute simple precision/recall/f1
        gt_set = set(gt_paragraph_ids)
        pred_set = set(pred_paragraph_ids)
        tp = len(gt_set & pred_set)
        prec = tp / max(1, len(pred_set))
        rec = tp / max(1, len(gt_set))
        f1 = 2 * prec * rec / max(1e-9, (prec + rec))

        results.append({
            "data_id": data_id,
            "prompt": prompt,
            "answer_raw": answer_raw,
            "pred_indices": chosen_indices,
            "pred_paragraph_ids": pred_paragraph_ids,
            "gt_paragraph_ids": gt_paragraph_ids,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

    cur.close()
    conn.close()
    return results
