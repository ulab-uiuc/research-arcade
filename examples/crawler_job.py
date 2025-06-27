import datetime
from graph_constructor.node_processor import NodeConstructor
from multi_input.multi_download import MultiDownload

PATH = "download"
FIELD = "cs.AI"

def main():
    md = MultiDownload()
    nc = NodeConstructor()

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    ids = md.download_papers_by_field_and_date(
        field=FIELD,
        start_date=today_str,
        dest_dir=PATH
    )
    for arxiv_id in ids:
        try:
            nc.process_paper(arxiv_id=arxiv_id, dir_path=PATH)
        except Exception as e:
            print(f"[Warning] Failed to process {arxiv_id}: {e}")
    print(f"Cron job completed at {datetime.datetime.now().isoformat()}, processed {len(ids)} papers.")

if __name__ == "__main__":
    main()
