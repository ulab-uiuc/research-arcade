import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import List

from paperscraper.get_dumps import arxiv
from multi_input.multi_input import MultiInput  # Adjust import path as needed


def valid_date(s: str) -> str:
    # Accept YYYY-MM-DD or YYYY-MM; store as YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            d = datetime.strptime(s, fmt)
            if fmt == "%Y-%m":
                d = d.replace(day=1)
            return d.date().isoformat()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError("Use YYYY-MM-DD or YYYY-MM")


def extract_arxiv_ids(file_path: str) -> List[str]:
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

# [start_date, end_date], both ends inclusive
def arxiv_id_crawler(start_date, end_date, save_path="."):
    save_path = f"{save_path}/arxiv_metadata_{start_date}_{end_date}.jsonl"
    
    arxiv(start_date=start_date, end_date=end_date, save_path=save_path)
    arxiv_ids = extract_arxiv_ids(file_path=save_path)
    
    return arxiv_ids, save_path




# Testing passed!

# def main():
#     parser = argparse.ArgumentParser(description="Crawl arXiv papers by date range")
#     parser.add_argument("--start-date", type=valid_date, required=True, 
#                         help="Start date (YYYY-MM-DD or YYYY-MM)")
#     parser.add_argument("--end-date", type=valid_date, required=True,
#                         help="End date (YYYY-MM-DD or YYYY-MM)")
#     parser.add_argument("--field", type=str, help="Field filter (optional)")
#     parser.add_argument("--save-path", type=str, default="./data", 
#                         help="Directory to save results")
    
#     args = parser.parse_args()
    
#     arxiv_ids, output_file = arxiv_id_crawler(
#         start_date=args.start_date,
#         end_date=args.end_date,
#         field=args.field,
#         save_path=args.save_path
#     )
    
#     print(f"Extracted {len(arxiv_ids)} arXiv IDs")
#     print(f"Saved to: {output_file}")
    

# if __name__ == "__main__":
#     main()