import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multi_input.arxiv_crawler_new import extract_arxiv_ids, download_with_time

file_path = "download/arxiv_metadata.jsonl"

arxiv_ids = extract_arxiv_ids(file_path)

print(arxiv_ids)

