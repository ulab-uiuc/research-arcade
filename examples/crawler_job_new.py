import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper_crawler.crawler_job import CrawlerJob

cj = CrawlerJob(dest_dir="download")

arxiv_ids = cj.crawl_recent_arxiv_paper(year=2025, month=1, day=1, max_result=10)

print(arxiv_ids)

cj.process_paper_graphs(arxiv_ids=arxiv_ids)