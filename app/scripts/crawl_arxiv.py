import sys
import os

sys.path.insert(0, '/app')

import argparse
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# from research_arcade.arxiv_utils.paper_crawler.crawler_job import CrawlerJob
from research_arcade.arxiv_utils.multi_input.arxiv_crawler_new import download_with_time, extract_arxiv_ids
from research_arcade.research_arcade import ResearchArcade

def crawl_recent_arxiv_paper_new(start_date, end_date=None, path=None):
    # Create the save path target file if not exists

    save_path = download_with_time(start_date=start_date, end_date=end_date, save_path=path)
    print(save_path)
    arxiv_ids = extract_arxiv_ids(file_path=save_path)
    return arxiv_ids


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

def save_arxiv_ids(arxiv_ids, dest_dir, start_date, end_date, field=None):
    """Save processed arXiv IDs to a file."""
    # make directory first if it does not exist
    

    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename with date range and optional field
    field_suffix = f"_{field}" if field else ""
    filename = f"processed_ids_{start_date}_to_{end_date}{field_suffix}.txt"
    filepath = dest_path / filename

    with open(filepath, 'w') as f:
        for arxiv_id in arxiv_ids:
            f.write(f"{arxiv_id}\n")
    
    print(f"Saved {len(arxiv_ids)} processed arXiv IDs to {filepath}")
    return filepath

def process_papers(arxiv_ids, db_type):

    if db_type == "csv":
        # load the path from environment
        config = {
            'csv_dir': os.getenv('CSV_DATASET_FOLDER_PATH')
        }
    if db_type == "sql":
        # we load from env
        config = {

        }

    ra = ResearchArcade(db_type=db_type, config=config)

    # Based on the arxiv ids we crawl, we can further download the papers and convert them into graph structures, store them into the database.

    config = {
        'arxiv_ids': arxiv_ids,
        'dest_dir': os.getenv('PAPER_FOLDER_PATH')
    }

    ra.construct_tables_from_arxiv_ids(config=config)


def main():
    parser = argparse.ArgumentParser(description="Crawl arXiv and run pipeline.")
    parser.add_argument("--start-date", required=True, type=valid_date)
    parser.add_argument("--end-date", type=valid_date, help="Defaults to today")
    parser.add_argument("--field", default=None, help="e.g., cs.AI, cs.CV, stat.ML")
    parser.add_argument("--dest", default="download", help="Output directory")
    parser.add_argument("--arxiv_id_dest", default="arxiv_ids", help="Directory that stores downloaded arxiv ids")
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date or date.today().isoformat()
    field = args.field
    dest_dir = args.dest
    arxiv_id_dest = args.arxiv_id_dest

    print(dest_dir)
    print(field)
    # sys.exit()

    # cj = CrawlerJob(dest_dir=str(dest_dir))

    # If your crawler supports field/category, pass it; otherwise remove 'field'
    # ids_raw may be a list of IDs (e.g., ['2508.01234', ...])
    ids_raw = crawl_recent_arxiv_paper_new(
        start_date=start_date,
        end_date=end_date,
        path=str(dest_dir)
    )

    print(f"Papers from {start_date} to {end_date}: {len(ids_raw)} found")

    # We should flter papers of specific fields/categories
    # Otherwise, there will be repetition


    # Save arXiv IDs to file
    save_arxiv_ids(
        arxiv_ids=ids_raw,
        dest_dir=arxiv_id_dest,
        start_date=start_date,
        end_date=end_date,
        field=field
    )

    # process the paper with ids

    process_papers(arxiv_ids=ids_raw, db_type='csv')
    
    print("Done")
    
if __name__ == "__main__":
    main()
