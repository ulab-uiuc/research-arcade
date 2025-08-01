import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper_crawler.crawler_job import CrawlerJob

cj = CrawlerJob(dest_dir="download")

cj.drop_task_database()
cj.create_task_database()
cj.db.drop_all()
cj.db.create_all()
