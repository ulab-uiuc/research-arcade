from paperscraper.get_dumps import arxiv
from .multi_input import MultiInput
import json
from typing import List
import os


def download_with_time(start_date, end_date=None, save_path="./download"):

    # os.mkdir(save_path)
    os.makedirs(save_path, exist_ok=True)

    save_path = f"{save_path}/arxiv_metadata_{start_date}_{end_date}.jsonl"
    print(save_path)
    with open(save_path, 'w') as f:
        f.write("")

    arxiv(start_date=start_date, end_date=end_date, save_path=save_path) # scrapes all metadata from 2024 until today.
    return save_path


def extract_arxiv_ids(file_path: str) -> List[str]:
    """
    Reads either NDJSON (one JSON object per line) or a JSON array file,
    pulls 'doi' (or common variants), converts to arXiv IDs via MultiInput.doi_to_id,
    and returns a de-duplicated list preserving order.
    """
    mi = MultiInput()
    arxiv_ids: List[str] = []
    seen = set()

    def try_add(doi):
        if not doi:
            return
        try:
            arx_id = mi.doi_to_id(doi)
        except Exception:
            return
        if arx_id not in seen:
            seen.add(arx_id)
            arxiv_ids.append(arx_id)

    with open(file_path, "r", encoding="utf-8") as f:
        # Peek first non-whitespace char to detect JSON array vs NDJSON
        start = f.read(1)
        while start and start.isspace():
            start = f.read(1)
        f.seek(0)

        if start == "[":  # JSON array
            data = json.load(f)
            for obj in data:
                if isinstance(obj, dict):
                    doi = obj.get("doi") or obj.get("DOI") or obj.get("paper_doi")
                    try_add(doi)
        else:  # NDJSON
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # skip malformed line
                    continue
                if isinstance(obj, dict):
                    doi = obj.get("doi") or obj.get("DOI") or obj.get("paper_doi")
                    try_add(doi)

    return arxiv_ids

