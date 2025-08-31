import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from datetime import date, datetime
from pathlib import Path
import json

from paper_crawler.crawler_job import CrawlerJob


def main():
    parser = argparse.ArgumentParser(description="Crawl arXiv and run pipeline.")
    parser.add_argument("--source", default=None, help="Arxiv id source directory")
    parser.add_argument("--dest", default="./download", help="Output directory")
    
    args = parser.parse_args()

    source_path = args.source
    dest_dir = (args.dest)
    
    cj = CrawlerJob(dest_dir=str(dest_dir)) 

    ids_raw = []

    # Open the json file and read arxiv ids from it.
    with open(source_path, "r") as f:
        ids_raw = json.load(f)

    
    print(f"Papers from file {source_path}: {len(ids_raw)} found")
    
    # Initialize tasks  
    init_res = cj.initialize_paper_tasks(arxiv_ids=ids_raw, category=None)
    if isinstance(init_res, dict):
        ids_to_process = init_res.get("added", []) or init_res.get("ids", [])
        skipped = init_res.get("skipped", [])
        print(f"Paper tasks initiated. Added: {len(ids_to_process)} | Skipped: {len(skipped)}")
    else:
        ids_to_process = list(init_res)  # assume it's an iterable of IDs
        print(f"Paper tasks initiated. Added: {len(ids_to_process)}")

    if not ids_to_process:
        print("Nothing to process. Exiting.")
        return
    
    # Pipeline
    print("Downloading latex papers")
    cj.download_papers(arxiv_ids=ids_to_process)
    
    print("Building paper graph")
    ids_to_process2 = cj.process_paper_graphs(arxiv_ids=ids_to_process)

    print("Extracting paragraphs")
    cj.process_paper_paragraphs(arxiv_ids=ids_to_process2)
    print("Done")
    
if __name__ == "__main__":
    main()
