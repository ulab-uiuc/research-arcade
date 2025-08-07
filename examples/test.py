import datetime
from graph_constructor.node_processor import NodeConstructor
from multi_input.multi_download import MultiDownload
import time
import os


PATH = "download"
FIELD = "cs.AI"

md = MultiDownload()
nc = NodeConstructor()

today_str = datetime.date.today().strftime("%Y-%m-%d")
yesterday = datetime.date.today() - datetime.timedelta(days=3)
yesterday_str = yesterday.strftime("%Y-%m-%d")

ids = md.download_papers_by_field_and_date(
    field=FIELD,
    start_date=yesterday_str,
    dest_dir=PATH,
    max_results=5
)

print(f"Fetched papers: {ids}")

for arxiv_id in ids:
    
    metadata_path = os.path.join(PATH, arxiv_id, f"{arxiv_id}_metadata.json")
    timeout = 10
    interval = 0.5
    waited = 0
    while not os.path.exists(metadata_path) and waited < timeout:
        time.sleep(interval)
        waited += interval
    if not os.path.exists(metadata_path):
        print(f"[Error] Metadata for {arxiv_id} not found after waiting")
        continue
    try:
        nc.process_paper(arxiv_id=arxiv_id, dir_path=PATH)
    except Exception as e:
        print(f"[Warning] Failed to process {arxiv_id}: {e}")

    try:
        nc.process_paper(arxiv_id=arxiv_id, dir_path=PATH)
    except Exception as e:
        print(f"[Warning] Failed to process {arxiv_id}: {e}")
print(f"Cron job completed at {datetime.datetime.now().isoformat()}, processed {len(ids)} papers.")

