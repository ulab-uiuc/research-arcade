import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from datetime import date, datetime
from pathlib import Path

from paper_crawler.crawler_job import CrawlerJob

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

def main():
    parser = argparse.ArgumentParser(description="Crawl arXiv and run pipeline.")
    parser.add_argument("--start-date", required=True, type=valid_date)
    parser.add_argument("--end-date", type=valid_date, help="Defaults to today")
    parser.add_argument("--field", default=None, help="e.g., cs.AI, cs.CV, stat.ML")
    parser.add_argument("--dest", default="download", help="Output directory")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date or date.today().isoformat()
    field = args.field
    dest_dir = (args.dest)
    
    cj = CrawlerJob(dest_dir=str(dest_dir))

    # If your crawler supports field/category, pass it; otherwise remove 'field'
    # ids_raw may be a list of IDs (e.g., ['2508.01234', ...])
    ids_raw = cj.crawl_recent_arxiv_paper_new(
        start_date=start_date,
        end_date=end_date,
        path=str(dest_dir)
    )
    
    print(f"Papers from {start_date} to {end_date}: {len(ids_raw)} found")

    # Initialize tasks
    init_res = cj.initialize_paper_tasks(arxiv_ids=ids_raw, category=field)
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
