import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper_crawler.crawler_job import CrawlerJob

arxiv_ids_fake = ['2008.03842', '2008.03843', '2008.03844', '2008.03845', '2008.03846', '2008.03847', '2008.03848', '2008.03849', '2008.03850', '2008.03851', '2008.03842', "2008.03846"]

cj = CrawlerJob(dest_dir="./download")

task_results = cj.initialize_paper_tasks(arxiv_ids=arxiv_ids_fake)
print(task_results)
print(task_results['added'])
print(task_results['skipped'])