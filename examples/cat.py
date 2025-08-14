import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper_crawler.crawler_job import CrawlerJob

path = ".\download"

cj = CrawlerJob(dest_dir = path)

ids = ["2505.22929", "2507.10539"]

result1 = cj.ids_with_major_category(arxiv_ids = ids, category = "cs")
result2 = cj.ids_with_major_category(arxiv_ids = ids, category = "math")

print(result1)
print(result2)