import sys
import os
import json  # Added missing json import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from paper_crawler.crawler_job import CrawlerJob

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ArXiv Author Information Crawler')
    
    parser.add_argument('--file_id', '-f', 
                       type=str, 
                       default="1",
                       help='File ID for input JSON file (default: 1)')
    
    return parser.parse_args()



def main():
    args = parse_args()

    cj = CrawlerJob(dest_dir="download")
    file_id = args.file_id
    dest_dir = "./json/"
    # i \in.[1,5]
    dest_path = f"{dest_dir}arxiv_id_author_{file_id}.json"

    with open(dest_path, 'r') as f:
        base_arxiv_ids = json.load(f)

    # cj.process_paper_paragraphs(arxiv_ids=base_arxiv_ids)
    for i, base_arxiv_id in enumerate(base_arxiv_ids):
        print(f"Currently processing the {i}th paper")
        result = cj.process_paper_authors(arxiv_ids=[base_arxiv_id])

        if result:
            print(f"Successfully processed the {i}th paper")

if __name__ == "__main__":
    main()