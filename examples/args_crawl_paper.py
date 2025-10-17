import sys
import os
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paper_crawler.crawler_job import CrawlerJob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", required=True, help="Path to JSON file with arxiv IDs")
    args = parser.parse_args()
    
    cj = CrawlerJob(dest_dir="download")

    # Load JSON file properly
    with open(args.file_path, "r") as f:
        json_file = json.load(f)

    # Collect arxiv IDs
    arxiv_ids = list(json_file)
    
    print("Number of loaded arxiv IDs:", len(arxiv_ids))

    # sys.exit()

    # Run crawler tasks
    # task_results = cj.initialize_paper_tasks(arxiv_ids=arxiv_ids)
    # arxiv_ids = task_results['added']
    
    # for arxiv_id in arxiv_ids:
    #     try:
    #         path = f"./download/output/{arxiv_id}.json"
    #         if not os.path.exists(path):
    #             cj.process_paper_graphs(arxiv_ids=[arxiv_id])
    #     except FileNotFoundError as e:
    #         print(f"File not found for arxiv_id {arxiv_id}: {e}")
    #     except Exception as e:
    #         print(f"Error processing arxiv_id {arxiv_id}: {e}")
    # We process those stuffs finally
        # DONT FORGET TO PROCESS THE PARAGRAPHS!!!
    cj.process_paper_paragraphs(arxiv_ids=arxiv_ids)
        # cj.process_paper_citations(arxiv_ids=arxiv_ids)


if __name__ == "__main__":
    main()
