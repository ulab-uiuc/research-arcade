import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# from research_arcade.arxiv_utils.paper_crawler.crawler_job import CrawlerJob
from research_arcade.arxiv_utils.multi_input.arxiv_crawler_new import download_with_time, extract_arxiv_ids
from research_arcade.research_arcade import ResearchArcade

def crawl_recent_arxiv_paper_new(start_date, end_date=None, path=None):
    # Create the save path target file if not exists

    save_path = download_with_time(start_date=start_date, end_date=end_date, save_path=path)
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
    # process the paper with ids

    ids_raw = ['1802.08773', '1806.02473']

    process_papers(arxiv_ids=ids_raw, db_type='csv')
    
    print("Done")
    
if __name__ == "__main__":
    main()


self.arxiv_citation.construct_citations_table_from_api(**config)
self.arxiv_paragraphs.construct_paragraphs_table_from_api(**config)
self.arxiv_paragraph_reference.construct_paragraph_references_table_from_api(**config)
self.arxiv_paragraph_citation.construct_citations_table_from_api(**config)
