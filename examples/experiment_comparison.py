#!/usr/bin/env python3
"""
Batch runner for paragraph_generation:
- Select top-N paragraph IDs from PostgreSQL
- Run under 4 conditions (figure+table, figure-only, table-only, none)
- Evaluate with ROUGE, SBERT, and GPT evaluation
- Save JSONL of full records and a CSV summary per condition + a combined CSV

Environment variables for DB connection (defaults in parens):
  PGHOST (localhost), PGPORT (5432), PGUSER (postgres), PGPASSWORD (required if needed), PGDATABASE (postgres)

Requires:
  psycopg2
  Your repo layout such that `tasks.paragraph_generation` and evaluation funcs are importable.
"""

import os
import sys
import csv
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Make repo root importable (adjust if this file lives elsewhere)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import psycopg2
from dataclasses import dataclass
from tasks.paragraph_generation import paragraph_generation
from tasks.generated_paragraph_evaluation import rouge_similarity, sbert_similarity, gpt_evaluation

# ----- Args dataclass must match the signature your paragraph_generation expects -----
@dataclass
class Args:
    paragraph_ids: List[int]
    model_name: str
    k_neighbour: int = 2
    figure_available: bool = True
    table_available: bool = True
    download_path: str = "./download"


# ----------------------- Configuration -----------------------
TOP_N = 100
MODEL_NAME = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
K_NEIGHBOUR = 4
DOWNLOAD_PATH = "./download"

# Four experimental conditions
CONDITIONS = [
    ("both",        True,  True),
    ("figure_only", True,  False),
    ("table_only",  False, True),
    ("none",        False, False),
]


csv_path = "./csv/paragraphs_with_figures_tables_citations_csv.csv"
# ----------------------- Utilities -----------------------


def fetch_top_paragraph_ids(limit: int = TOP_N) -> List[int]:
    """
    Returns the first `limit` paragraph primary keys.
    Adjust ORDER BY to your preferred criterion if needed.
    """
    ids: List[int] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(ids) >= limit:
                break
            val = (row["id"] or "").strip()
            if val:
                ids.append(int(val))
    print(ids)
    return ids


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def jsonl_write(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def csv_write(path: Path, rows: List[Dict[str, Any]], field_order: Optional[List[str]] = None):
    if not rows:
        # write header only if we have a field order
        if field_order:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=field_order)
                writer.writeheader()
        return
    if field_order is None:
        # derive union of keys (stable-ish)
        keys = set()
        for r in rows:
            keys.update(r.keys())
        field_order = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in field_order})


def safe_str(x: Any) -> str:
    try:
        if isinstance(x, (dict, list)):
            return json.dumps(x, ensure_ascii=False)
        return "" if x is None else str(x)
    except Exception:
        return repr(x)


# ----------------------- Core Run Logic -----------------------
def evaluate_one(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach evaluation scores to a single result row.
    """
    prompt = result.get("prompt", "")
    gen = result.get("llm_output") or ""
    gold = result.get("original content") or ""

    row = {
        "paragraph_id": result.get("paragraph_id"),
        "prompt": prompt,
        "generated": gen,
        "original": gold,
    }

    # Compute metrics robustly
    try:
        row["rouge"] = rouge_similarity(gen, gold)
    except Exception as e:
        row["rouge"] = None
        row["rouge_error"] = f"{type(e).__name__}: {e}"

    try:
        row["sbert"] = sbert_similarity(gen, gold)
    except Exception as e:
        row["sbert"] = None
        row["sbert_error"] = f"{type(e).__name__}: {e}"

    try:
        # Whatever your gpt_evaluation returns, keep it as-is (string/dict)
        ge = gpt_evaluation(gen, gold)
        row["gpt_eval"] = ge
    except Exception as e:
        row["gpt_eval"] = None
        row["gpt_eval_error"] = f"{type(e).__name__}: {e}"

    return row


def run_condition(
    paragraph_ids: List[int],
    model_name: str,
    k_neighbour: int,
    download_path: str,
    fig_avail: bool,
    tbl_avail: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Execute paragraph_generation for a set of paragraph ids and one (fig,tbl) condition,
    then evaluate and return (raw_results, evaluated_rows).
    """
    args = Args(
        paragraph_ids=paragraph_ids,
        model_name=model_name,
        k_neighbour=k_neighbour,
        figure_available=fig_avail,
        table_available=tbl_avail,
        download_path=download_path,
    )

    # Run the generation
    raw_results = paragraph_generation(args)  # expected: List[dict] with keys: paragraph_id, prompt, original content, llm_output

    # Evaluate each
    evaluated = []
    for r in raw_results:
        try:
            row = evaluate_one(r)
        except Exception as e:
            row = {
                "paragraph_id": r.get("paragraph_id"),
                "prompt": safe_str(r.get("prompt")),
                "generated": safe_str(r.get("llm_output")),
                "original": safe_str(r.get("original content")),
                "evaluation_error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=1),
            }
        evaluated.append(row)

    return raw_results, evaluated


def main():
    tstamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(f"./runs/paragraph_generation_{tstamp}")
    ensure_dir(out_dir)
    ensure_dir(Path(DOWNLOAD_PATH))

    print(f"[INFO] Output dir: {out_dir.resolve()}")

    # 1) Pick top-N paragraphs
    print(f"[INFO] Fetching top {TOP_N} paragraph IDs from DB...")
    pids = fetch_top_paragraph_ids(TOP_N)
    print(f"[INFO] Got {len(pids)} IDs. Example: {pids[:10]}")

    all_evals: List[Dict[str, Any]] = []

    # 2) Iterate four conditions
    for label, fig_avail, tbl_avail in CONDITIONS:
        print(f"\n[RUN] Condition = {label} (figure_available={fig_avail}, table_available={tbl_avail})")
        start = time.time()
        raw, evaled = run_condition(
            paragraph_ids=pids,
            model_name=MODEL_NAME,
            k_neighbour=K_NEIGHBOUR,
            download_path=DOWNLOAD_PATH,
            fig_avail=fig_avail,
            tbl_avail=tbl_avail,
        )
        dur = time.time() - start
        print(f"[DONE] {label}: {len(raw)} results in {dur:.1f}s")

        # Tag and persist
        for r in raw:
            r["condition"] = label
            r.setdefault("figure_available", fig_avail)
            r.setdefault("table_available", tbl_avail)
        for e in evaled:
            e["condition"] = label
            e["figure_available"] = fig_avail
            e["table_available"] = tbl_avail

        # Write per-condition outputs
        jsonl_write(out_dir / f"{label}_raw.jsonl", raw)
        jsonl_write(out_dir / f"{label}_eval.jsonl", evaled)

        # Also a per-condition CSV (scores + minimal fields)
        summary = []
        for e in evaled:
            summary.append({
                "paragraph_id": e.get("paragraph_id"),
                "condition": label,
                "figure_available": fig_avail,
                "table_available": tbl_avail,
                "rouge": safe_str(e.get("rouge")),
                "sbert": safe_str(e.get("sbert")),
                "gpt_eval": safe_str(e.get("gpt_eval")),
            })
        csv_write(out_dir / f"{label}_summary.csv", summary, field_order=[
            "paragraph_id", "condition", "figure_available", "table_available", "rouge", "sbert", "gpt_eval"
        ])

        all_evals.extend(evaled)

    # 3) Combined CSV across all conditions
    combined_rows = []
    for e in all_evals:
        combined_rows.append({
            "paragraph_id": e.get("paragraph_id"),
            "condition": e.get("condition"),
            "figure_available": e.get("figure_available"),
            "table_available": e.get("table_available"),
            "rouge": safe_str(e.get("rouge")),
            "sbert": safe_str(e.get("sbert")),
            "gpt_eval": safe_str(e.get("gpt_eval")),
            "evaluation_error": e.get("evaluation_error"),
        })
    csv_write(out_dir / "combined_summary.csv", combined_rows, field_order=[
        "paragraph_id", "condition", "figure_available", "table_available",
        "rouge", "sbert", "gpt_eval", "evaluation_error"
    ])

    print(f"\n[OK] Wrote outputs to: {out_dir.resolve()}")
    print(f"     - Per-condition JSONL: *_raw.jsonl, *_eval.jsonl")
    print(f"     - Per-condition CSV: *_summary.csv")
    print(f"     - Combined CSV: combined_summary.csv")


if __name__ == "__main__":
    main()
