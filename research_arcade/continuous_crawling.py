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
# from research_arcade.research_arcade import ResearchArcade
from research_arcade.research_arcade import ResearchArcade


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


def process_papers(arxiv_ids, db_type):
    """Process papers and store in database."""
    if db_type == "csv":
        config = {'csv_dir': os.getenv('CSV_DATASET_FOLDER_PATH')}
    elif db_type == "sql":
        config = {}
    else:
        config = {}

    ra = ResearchArcade(db_type=db_type, config=config)

    config = {
        'arxiv_ids': arxiv_ids,
        'dest_dir': os.getenv('PAPER_FOLDER_PATH')
    }
    ra.construct_tables_from_arxiv_ids(config=config)


def get_interval_seconds(interval: int) -> int:
    """Convert interval string to seconds."""
    return int(interval) * 86400

def run_single_crawl(start_date, end_date, field, dest_dir, arxiv_id_dest, db_type):
    """Run a single crawl iteration."""
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
            field=field
        )
        
        process_papers(arxiv_ids=ids_raw, db_type=db_type)
        
        print(f"[{datetime.now()}] Crawl completed successfully")
        return True
        
    except Exception as e:
        print(f"[{datetime.now()}] Crawl failed: {e}")
        return False


def continuous_crawling(interval_days, delay_days, field, dest_dir, arxiv_id_dest, db_type):
    """
    Runs the crawl process in an infinite loop.
    
    Args:
        interval_days (int): How many days of papers to fetch per crawl.
        delay_days (int): How many days to lag behind 'today' (to account for arXiv processing).
        field (str): The arXiv category filter.
        dest_dir (str): Directory to save downloaded papers.
        arxiv_id_dest (str): Directory to save the list of processed IDs.
        db_type (str): "csv" or "sql".
    """
    interval_seconds = get_interval_seconds(interval_days)

    print(f"Starting continuous crawl mode")
    print(f"  Interval: {interval_days} days")
    print(f"  Delay: {delay_days} days")
    print(f"  Field: {field or 'all'}")
    print(f"  Database Type: {db_type}")

    while True:
        # Calculate dynamic date window based on current time
        # Example: if interval=2 and delay=2, it fetches papers from 5 days ago to 2 days ago.
        start_date = (date.today() - timedelta(days=interval_days + delay_days + 1)).isoformat()
        end_date = (date.today() - timedelta(days=delay_days)).isoformat() 
        
        success = run_single_crawl(
            start_date=start_date,
            end_date=end_date,
            field=field,
            dest_dir=dest_dir,
            arxiv_id_dest=arxiv_id_dest,
            db_type=db_type
        )
        
        if success:
            print(f"[{datetime.now()}] Batch completed. Sleeping for {interval_days} days...")
        else:
            print(f"[{datetime.now()}] Batch failed. Will retry after sleep.")
            
        time.sleep(interval_seconds)

