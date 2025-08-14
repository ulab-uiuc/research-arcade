import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multi_input.arxiv_crawler_new import extract_arxiv_ids, download_with_time

from multi_input.multi_download import MultiDownload

from paper_crawler.crawler_job import CrawlerJob


path = "./download"
cj = CrawlerJob(dest_dir=path)


arxiv_ids = cj.select_unproceeded_task(task_type="paragraph", max_results=1000)
print(arxiv_ids)
cj.process_paper_paragraphs(arxiv_ids=arxiv_ids)

