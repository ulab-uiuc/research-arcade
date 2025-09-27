
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper_crawler.crawler_job import CrawlerJob

cj = CrawlerJob(dest_dir="download")


arxiv_ids = cj.crawl_recent_arxiv_paper(year=2020, month=1, day=1, max_result=1000)

print(arxiv_ids)
# print(len(arxiv_ids))

arxiv_ids = {'2412.17767v2'}

cj.initialize_paper_tasks(arxiv_ids=arxiv_ids)

arxiv_ids = cj.select_unproceeded_task(task_type="paper_graph", max_results=40000)
# print(f"arxiv ids: {arxiv_ids}")
print(f"Number of arxiv papers selected {len(arxiv_ids)}")


# print(f"arxiv ids: {arxiv_ids}")

# cj.download_papers(arxiv_ids=arxiv_ids)


# arxiv_ids = cj.select_proceeded_task(task_type="paper_graph", max_results=10)
# arxiv_ids = cj.select_unproceeded_task(task_type="paper_graph", max_results=40000)
# print(f"arxiv ids: {arxiv_ids}")
# cj.process_paper_graphs(arxiv_ids=arxiv_ids)


arxiv_ids = cj.select_unproceeded_task(task_type="paragraph", max_results=1000)
print(f"arxiv ids: {arxiv_ids}")
cj.process_paper_paragraphs(arxiv_ids=arxiv_ids)

# cj.process_paper_authors(arxiv_ids=arxiv_ids)

arxiv_ids = cj.select_unproceeded_task(task_type="citation", max_results=1000)

cj.process_paper_citations(arxiv_ids=arxiv_ids)

