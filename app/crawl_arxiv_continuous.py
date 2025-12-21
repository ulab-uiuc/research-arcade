import sys
import os
import time
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, '/app')

from dotenv import load_dotenv
load_dotenv()

from research_arcade.arxiv_utils.multi_input.arxiv_crawler_new import download_with_time, extract_arxiv_ids
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


def get_interval_seconds(interval: str) -> int:
    """Convert interval string to seconds."""
    if interval == "hourly":
        return 3600
    elif interval == "daily":
        return 86400
    elif interval == "weekly":
        return 604800
    else:
        # Assume it's a number (seconds)
        try:
            return int(interval)
        except ValueError:
            return 86400  # Default to daily


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


def main():
    parser = argparse.ArgumentParser(description="Continuous arXiv crawler (no Docker required)")
    parser.add_argument("--start-date", type=valid_date, help="Initial start date (defaults to lookback from today)")
    parser.add_argument("--end-date", type=valid_date, help="End date (defaults to today)")
    parser.add_argument("--field", default=None, help="arXiv category, e.g., cs.AI, cs.CV, stat.ML")
    parser.add_argument("--dest", default="download", help="Output directory for downloads")
    parser.add_argument("--arxiv_id_dest", default="arxiv_ids", help="Directory for arXiv ID files")
    parser.add_argument("--db-type", default="csv", choices=["csv", "sql"], help="Database type")
    
    # Continuous mode options
    parser.add_argument("--continuous", action="store_true", help="Run continuously in a loop")
    parser.add_argument("--interval", default="daily", help="Crawl interval: hourly, daily, weekly, or seconds")
    parser.add_argument("--lookback-days", type=int, default=1, help="Days to look back for each crawl")
    
    args = parser.parse_args()

    # Single run mode
    if not args.continuous:
        start_date = args.start_date or (date.today() - timedelta(days=1)).isoformat()
        end_date = args.end_date or date.today().isoformat()
        
        run_single_crawl(
            start_date=start_date,
            end_date=end_date,
            field=args.field,
            dest_dir=args.dest,
            arxiv_id_dest=args.arxiv_id_dest,
            db_type=args.db_type
        )
        print("Done")
        return

    # Continuous mode
    interval_seconds = get_interval_seconds(args.interval)
    print(f"Starting continuous crawl mode")
    print(f"  Interval: {args.interval} ({interval_seconds} seconds)")
    print(f"  Lookback: {args.lookback_days} days")
    print(f"  Field: {args.field or 'all'}")
    print(f"  Destination: {args.dest}")

    while True:
        start_date = (date.today() - timedelta(days=args.lookback_days)).isoformat()
        end_date = date.today().isoformat()
        
        run_single_crawl(
            start_date=start_date,
            end_date=end_date,
            field=args.field,
            dest_dir=args.dest,
            arxiv_id_dest=args.arxiv_id_dest,
            db_type=args.db_type
        )
        
        print(f"[{datetime.now()}] Sleeping for {interval_seconds} seconds...")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()