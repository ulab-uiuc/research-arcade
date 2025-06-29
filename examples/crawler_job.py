import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime

from graph_constructor.node_processor import NodeConstructor
from multi_input.multi_download import MultiDownload

PATH = "download"
FIELD = "cs.AI"

def main():
    md = MultiDownload()
    nc = NodeConstructor()

    two_days_ago_str = k_days_ago_str(2)

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    ids = md.download_papers_by_field_and_date(
        field=FIELD,
        start_date=two_days_ago_str,
        dest_dir=PATH,
        max_results = 10
    )

    nc.drop_tables()
    nc.create_tables()


    for arxiv_id in ids:
        try:
            # First build the json file of paper using knowledge debugger
            md.build_paper_graph(
                input=arxiv_id,
                input_type="id",
                dest_dir=PATH
            )
            # Then we process the paper based on it
            nc.process_paper(arxiv_id=arxiv_id, dir_path=PATH)
        except Exception as e:
            print(f"[Warning] Failed to process {arxiv_id}: {e}")
    print(f"Cron job completed at {datetime.datetime.now().isoformat()}, processed {len(ids)} papers.")


def k_days_ago_str(k: int) -> str:
    """
    Return the date k days before today, formatted as 'YYYY-MM-DD'.
    """
    return (datetime.date.today() - datetime.timedelta(days=k)).strftime("%Y-%m-%d")

if __name__ == "__main__":
    main()
