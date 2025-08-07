import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime
import time

from graph_constructor.node_processor import NodeConstructor
from multi_input.multi_download import MultiDownload
from graph_constructor.paragraph_processor import ParagraphProcessor

PATH = "download"
FIELD = "cs.AI"

def main():
    md = MultiDownload()
    nc = NodeConstructor()  
    data_dir = f"{PATH}/output/endpoints"
    figures_dir = f"{PATH}/output/figures"
    output_dir = f"{PATH}/output/paragraphs"

    pp = ParagraphProcessor(data_dir=data_dir, figures_dir=figures_dir, output_dir=output_dir)

    nc.create_tables()
    time_paper_graph = []
    time_paper_database = []

    days_ago_str = k_days_ago_str(30)

    print(f"Day Query: {days_ago_str}")

    today_str = datetime.date.today().strftime("%Y-%m-%d")

    # Here, we use ascending to download older paper first
    ids = md.download_papers_by_field_and_date(
        field=FIELD,
        start_date=days_ago_str,
        dest_dir=PATH,
        max_results = 10,
        sort_order="descending"
    )


    for arxiv_id in ids:
        try:
            # First build the json file of paper using knowledge debugger
            t0 = time.perf_counter()
            md.build_paper_graph(
                input=arxiv_id,
                input_type="id",
                dest_dir=PATH
            )
            t = time.perf_counter() - t0
            time_paper_graph.append(t)
            
            t0 = time.perf_counter()
            # Then we process the paper based on it
            nc.process_paper(arxiv_id=arxiv_id, dir_path=PATH)
            t = time.perf_counter() - t0
            time_paper_database.append(t)
        except Exception as e:
            print(f"[Warning] Failed to process {arxiv_id}: {e}")
    print(f"Cron job completed at {datetime.datetime.now().isoformat()}, processed {len(ids)} papers.")
    print(f"Time needed in each loop for json processing: {time_paper_graph}")
    print(f"Time needed in each loop for graph database processing: {time_paper_database}")

    # After we build all the jobs, we extract the paragraph informaiton

    pp.extract_paragraphs()

    nc.process_paragraphs("download")


def k_days_ago_str(k: int) -> str:
    """
    Return the date k days before today, formatted as 'YYYY-MM-DD'.
    """
    return (datetime.date.today() - datetime.timedelta(days=k)).strftime("%Y-%m-%d")

if __name__ == "__main__":
    main()
