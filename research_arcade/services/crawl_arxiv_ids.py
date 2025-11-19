from paperscraper.get_dumps import arxiv
import json
from typing import List
import re

def doi_to_id(self, arxiv_doi: str) -> str:

    if not isinstance(arxiv_doi, str):
        raise TypeError("arxiv_doi must be a string")

    s = arxiv_doi.strip()

    s = re.sub(r'^(https?://(?:dx\.)?doi\.org/)', '', s, flags=re.I)

    m = re.match(r'^10\.48550/ARXIV\.(?P<id>[^?#]+)$', s, flags=re.I)
    if not m:
        raise ValueError(f"Not an arXiv DOI: {arxiv_doi!r}")

    arxiv_id = m.group('id')

    # remove optional version suffix, e.g., 'v3'
    arxiv_id = re.sub(r'v\d+$', '', arxiv_id, flags=re.I)

    return arxiv_id


def download_with_time(start_date, end_date=None, save_path="./download"):

    save_path = f"{save_path}/arxiv_metadata_{start_date}_{end_date}.jsonl"

    arxiv(start_date=start_date, end_date=end_date, save_path=save_path) # scrapes all metadata from 2024 until today.
    return save_path

def extract_arxiv_ids(file_path: str) -> List[str]:
    """
    Reads either NDJSON (one JSON object per line) or a JSON array file,
    pulls 'doi' (or common variants), converts to arXiv IDs via MultiInput.doi_to_id,
    and returns a de-duplicated list preserving order.
    """
    arxiv_ids: List[str] = []
    seen = set()

    def try_add(doi):
        if not doi:
            return
        try:
            arx_id = doi_to_id(doi)
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


def crawl_recent_arxiv_paper_new( start_date, end_date=None, path=None):
    save_path = download_with_time(start_date=start_date, end_date=end_date, save_path=path)
    arxiv_ids = extract_arxiv_ids(file_path=save_path)
    return arxiv_ids

