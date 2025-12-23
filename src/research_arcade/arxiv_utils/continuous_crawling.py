import sys
import os
import time
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, '/app')

from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from research_arcade.arxiv_utils.multi_input.arxiv_crawler_new import download_with_time, extract_arxiv_ids


def crawl_recent_arxiv_paper_new(start_date, end_date=None, path=None):
    """Download papers and extract arXiv IDs."""
    save_path = download_with_time(start_date=start_date, end_date=end_date, save_path=path)
    print(save_path)
    arxiv_ids = extract_arxiv_ids(file_path=save_path)
    return arxiv_ids


def valid_date(s: str) -> str:
    """Parse date string in YYYY-MM-DD or YYYY-MM format."""
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            d = datetime.strptime(s, fmt)
            if fmt == "%Y-%m":
                d = d.replace(day=1)
            return d.date().isoformat()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError("Use YYYY-MM-DD or YYYY-MM")


def save_arxiv_ids(arxiv_ids, dest_dir, start_date, end_date, field=None):
    """Save processed arXiv IDs to a file."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    field_suffix = f"_{field}" if field else ""
    filename = f"processed_ids_{start_date}_to_{end_date}{field_suffix}.txt"
    filepath = dest_path / filename

    with open(filepath, 'w') as f:
        for arxiv_id in arxiv_ids:
            f.write(f"{arxiv_id}\n")

    print(f"Saved {len(arxiv_ids)} processed arXiv IDs to {filepath}")
    return filepath


def get_interval_seconds(interval: int) -> int:
    """Convert interval string to seconds."""
    return int(interval) * 86400


def run_single_crawl(start_date, end_date, paper_category, dest_dir, arxiv_id_dest):
    """
    Run a single crawl iteration. Returns arxiv_ids for caller to process.
    
    Returns:
        list: List of arxiv_ids if successful, None if failed
    """
    print(f"\n{'='*60}")
    print(f"[{datetime.now()}] Starting crawl: {start_date} to {end_date}")
    print(f"{'='*60}")
    
    try:
        ids_raw = crawl_recent_arxiv_paper_new(
            start_date=start_date,
            end_date=end_date,
            path=str(dest_dir)
        )
        
        print(f"Papers from {start_date} to {end_date}: {len(ids_raw)} found")
        
        save_arxiv_ids(
            arxiv_ids=ids_raw,
            dest_dir=arxiv_id_dest,
            start_date=start_date,
            end_date=end_date,
            field=paper_category
        )
        
        print(f"[{datetime.now()}] Crawl completed successfully")
        return ids_raw
        
    except Exception as e:
        print(f"[{datetime.now()}] Crawl failed: {e}")
        return None