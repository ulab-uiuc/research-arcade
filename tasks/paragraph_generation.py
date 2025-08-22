import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict

import psycopg2
import psycopg2.extras


# ---------- Config ----------
PASSWORD = os.getenv("PGPASSWORD", "REPLACE_ME")


# ---------- Types ----------
@dataclass
class Args:
    paragraph_ids: List[int]
    k_neighbour: int = 2
    figure_available: bool = True
    table_available: bool = True
    # Optional: pass an LLM function that takes a prompt and returns text
    llm_generate: Optional[Callable[[str], str]] = None


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


def _format_block(label: str, lines: List[str]) -> str:
    if not lines:
        return ""
    header = f"{label}:\n"
    return header + "\n".join(lines)


def _fetch_context_for_paragraph(cur, paragraph_id: int) -> Tuple[str, str, int]:
    cur.execute(
        """
        SELECT paper_arxiv_id, paper_section, paragraph_id
        FROM paragraphs
        WHERE id = %s
        """,
        (paragraph_id,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Paragraph id {paragraph_id} not found.")
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


def _fetch_global_refs_for_paragraph(cur, paragraph_id: int, ref_type: str) -> List[int]:
    """
    Uses your helper mapping. If you already have paragraph_ref_to_global_ref, call it;
    otherwise select from a junction table like paragraph_references(paragraph_id, reference_type, global_id).
    """
    # If you do have a Python helper, swap this body out:
    cur.execute(
        """
        SELECT global_id
        FROM paragraph_references
        WHERE paragraph_id = %s AND reference_type = %s
        """,
        (paragraph_id, ref_type),
    )
    return [r[0] for r in cur.fetchall()]


def _fetch_figures(cur, figure_ids: List[int]) -> List[Dict]:
    if not figure_ids:
        return []
    cur.execute(
        """
        SELECT id, reference_label, caption, file_path
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
            "file_path": path or ""
        })
    return figures


def _fetch_tables(cur, table_ids: List[int]) -> List[Dict]:
    if not table_ids:
        return []
    cur.execute(
        """
        SELECT id, reference_label, table_text
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


def _fetch_cited_pairs(cur, paragraph_id: int) -> List[Tuple[str, Optional[str]]]:
    """
    Returns list of (arxiv_id_or_url, bib_key_or_none).
    If you already have paragraph_to_global_citation(paragraph_id) in Python, call that instead.
    """
    cur.execute(
        """
        SELECT cited_arxiv_id, bib_key
        FROM paragraph_citations
        WHERE paragraph_id = %s
        """,
        (paragraph_id,),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def _fetch_abstracts(cur, arxiv_ids_or_urls: List[str]) -> List[Tuple[str, str, str]]:
    """
    Returns list of (bib_key, title, abstract).
    We try exact id, and then the versionless id (split at 'v').
    """
    results = []
    for raw in arxiv_ids_or_urls:
        if not raw:
            continue
        # extract id if it's a URL
        if "/" in raw:
            arx = raw.strip().split("/")[-1]
        else:
            arx = raw.strip()

        versionless = arx.split("v", 1)[0] if "v" in arx else arx

        cur.execute("SELECT title, abstract, bib_key FROM papers WHERE arxiv_id = %s", (arx,))
        row = cur.fetchone()
        if not row:
            cur.execute("SELECT title, abstract, bib_key FROM papers WHERE arxiv_id = %s", (versionless,))
            row = cur.fetchone()

        if row:
            title, abstract, bib_key = row
            if not bib_key:
                bib_key = _derived_bib_key(title or versionless)
            results.append((bib_key, title or "", abstract or ""))

    return results


def _build_prompt(
    k: int,
    figures: List[Dict],
    tables: List[Dict],
    abstracts: List[Tuple[str, str, str]],
    prev_paras: List[str],
    next_paras: List[str],
) -> str:
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

    prompt = f"""
You are given the following inputs for reconstructing a missing paragraph in a research paper.

Figure (optional):
{figure_block if figure_block else "(none)"}

Table (optional):
{table_block if table_block else "(none)"}

Abstract(s) of cited paper(s):
{abstracts_block if abstracts_block else "(none)"}

{k}-Most Adjacent Paragraphs (context):
{adjacent_paragraphs_block}

# Task
Write exactly one LaTeX-formatted paragraph that naturally fits between the adjacent paragraphs.

# Requirements
- If a figure is provided, explicitly reference it with: Figure~\\ref{{{{{ '{' }figure_label{ '}' } }}}}, and incorporate at least one concrete detail from the figure’s content or caption.
- If a table is provided, explicitly reference it with: Table~\\ref{{{{{ '{' }table_label{ '}' } }}}}, and incorporate at least one concrete detail from the table’s content.
- Incorporate at least one core claim or finding from the abstract(s) and cite it with \\citep{{{{{ '{' }bib_key{ '}' } }}}}. Use the provided BibTeX key(s) if present; otherwise, use a stable placeholder key derived from title (e.g., {{derived_bib_key}}).
- Ensure the paragraph logically continues from and sets up the surrounding {k} adjacent paragraph(s).
- Style: objective, concise, academic tone; ~120–180 words.
- Formatting: produce a single LaTeX paragraph only (no section headers, lists, environments; math only if essential).
- Constraints: do not include \\label{{...}}; do not write “Figure X”/“Table Y”; do not copy raw table/figure content verbatim; summarize/interpret key points.

# Output
Return only the LaTeX paragraph text, nothing else.
""".strip()

    return prompt


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
        user="postgres",
        password=PASSWORD,
        port="5432",
    )
    conn.autocommit = True
    cur = conn.cursor()

    outputs: List[Dict[str, str]] = []

    for pid in args.paragraph_ids:
        # context anchor
        paper_arxiv_id, paper_section, pivot_local_id = _fetch_context_for_paragraph(cur, pid)

        # figures
        figures = []
        if args.figure_available:
            fig_ids = _fetch_global_refs_for_paragraph(cur, pid, "figure")
            figures = _fetch_figures(cur, fig_ids)

        # tables
        tables = []
        if args.table_available:
            tab_ids = _fetch_global_refs_for_paragraph(cur, pid, "table")
            tables = _fetch_tables(cur, tab_ids)

        # citations → abstracts (+bib keys)
        cited_pairs = _fetch_cited_pairs(cur, pid)  # (arxiv_or_url, bib_key_or_none)
        cited_ids = [cp[0] for cp in cited_pairs]
        abstracts = _fetch_abstracts(cur, cited_ids)

        # adjacent paragraphs
        prev_paras, next_paras = _fetch_adjacent_paragraphs(
            cur, paper_arxiv_id, paper_section, pivot_local_id, args.k_neighbour
        )

        prompt = _build_prompt(
            k=args.k_neighbour,
            figures=figures,
            tables=tables,
            abstracts=abstracts,
            prev_paras=prev_paras,
            next_paras=next_paras,
        )

        llm_output = None
        if args.llm_generate:
            llm_output = args.llm_generate(prompt)

        outputs.append({
            "paragraph_id": str(pid),
            "prompt": prompt,
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


# ---------- Usage example ----------
if __name__ == "__main__":
    # Plug your own LLM adapter here if desired
    # llm_fn = example_openai_adapter()
    llm_fn = None  # build prompts only

    args = Args(
        paragraph_ids=[12345, 12346],
        k_neighbour=2,
        figure_available=True,
        table_available=True,
        llm_generate=llm_fn,
    )

    results = paragraph_generation(args)
    for r in results:
        print("\n=== Paragraph ID:", r["paragraph_id"], "===\n")
        print("PROMPT:\n", r["prompt"])
        if r["llm_output"]:
            print("\nLLM OUTPUT:\n", r["llm_output"])
