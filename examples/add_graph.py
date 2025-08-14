import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper_crawler.crawler_job import CrawlerJob

dest_dir = "./download"
cj = CrawlerJob(dest_dir=dest_dir)

arxiv_ids = cj.add_existing_paper_graphs()

print("Added arxiv ids:")
print(arxiv_ids)
