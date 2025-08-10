import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multi_input.arxiv_crawler_new import extract_arxiv_ids, download_with_time

from multi_input.multi_download import MultiDownload

from paper_crawler.crawler_job import CrawlerJob

cj = CrawlerJob(dest_dir="download")

start_date = "2020-08-10"
end_date = "2020-08-10"
path = "./download"

file_path = f"./download/arxiv_metadata_{start_date}_{end_date}.jsonl"

arxiv_ids = cj.crawl_recent_arxiv_paper_new(start_date=start_date, end_date=end_date, path=path)

# arxiv_ids = extract_arxiv_ids(file_path)

print(arxiv_ids)

task_results = cj.initialize_paper_tasks(arxiv_ids=arxiv_ids)

arxiv_ids = task_results['added']

# arxiv_ids = cj.select_unproceeded_task(task_type="downloaded", max_results=10)

print(arxiv_ids)

# arxiv_ids = ['2008.03842', '2008.03843', '2008.03844', '2008.03845', '2008.03846', '2008.03847', '2008.03848', '2008.03849', '2008.03850', '2008.03851']
# print(arxiv_ids)

cj.download_papers(arxiv_ids=arxiv_ids)

cj.process_paper_graphs(arxiv_ids=arxiv_ids)

cj.process_paper_paragraphs(arxiv_ids=arxiv_ids)

# arxiv_ids = cj.select_unproceeded_task(task_type="citation", max_results=1000)

# cj.process_paper_citations(arxiv_ids=arxiv_ids)
