"""
Why only 1000 tasks initiated?
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multi_input.arxiv_crawler_new import download_with_time, extract_arxiv_ids
from paper_crawler.crawler_job import CrawlerJob


arxiv_ids = extract_arxiv_ids(file_path = "./download/arxiv_metadata_2020-08-10_2020-08-12.jsonl")

print(len(arxiv_ids))

cj = CrawlerJob(dest_dir="download")

cj.initialize_paper_tasks(arxiv_ids=arxiv_ids)
