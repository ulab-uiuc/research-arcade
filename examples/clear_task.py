import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from paper_crawler.crawler_job import CrawlerJob

cj = CrawlerJob(dest_dir="download")


with open("download/output/history.json", "w") as f:
    json.dump([], f)

cj.drop_task_database()
cj.create_task_database()
