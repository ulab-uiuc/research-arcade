import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.util import select_papers_with_criteria

import argparse

def main():
    parser = argparse.ArgumentParser(description="Crawl arXiv and run pipeline.")
    parser.add_argument("--para_num", default=30)

    args = parser.parse_args()

    min_para_num = args.para_num

    selected_arxiv_ids = select_papers_with_criteria(min_paragraph=min_para_num)

    print(selected_arxiv_ids)

if __name__  == "__main__":
    main()

