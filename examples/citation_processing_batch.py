import sys
import os
import argparse
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paper_crawler.crawler_job import CrawlerJob

import csv, re


def load_arxiv_ids(file_name: str):
    """Load arxiv IDs from a CSV file with a column named 'arxiv_id'."""
    ids = []
    with open(file_name, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "[" not in reader.fieldnames:
            raise ValueError(f"CSV must contain a column named 'arxiv_id'. Found: {reader.fieldnames}")
        for row in reader:
            val = (row["["] or "").strip()
            val = normalize_arxiv_id(val)
            if val:
                ids.append(val)
    return ids[1:-1:1]

def normalize_arxiv_id(s: str) -> str:
    s = s.strip()
    s = s.strip('"').strip("'")                            # drop wrapping quotes
    s = re.sub(r'^https?://arxiv\.org/abs/', '', s)        # drop URL
    s = re.sub(r'^arXiv:', '', s, flags=re.I)              # drop "arXiv:" prefix
    return s


def main():
    parser = argparse.ArgumentParser(description="Process paper citations for a list of arXiv IDs.")
    parser.add_argument("--source", required=True, help="Path to the CSV file containing arxiv IDs.")
    parser.add_argument("--dest", default="./download", help="Destination directory for crawler outputs.")
    args = parser.parse_args()

    arxiv_ids = load_arxiv_ids(args.source)
    if not arxiv_ids:
        raise ValueError(f"No arxiv IDs found in {args.source}.")

    cj = CrawlerJob(dest_dir=args.dest)
    print(f"Loaded {len(arxiv_ids)} arxiv IDs from {args.source}")
    processed_ids = cj.process_paper_citations(arxiv_ids=arxiv_ids)
    print(processed_ids)


if __name__ == "__main__":
    main()
