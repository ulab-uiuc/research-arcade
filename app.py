import datetime
from flask import Flask
from graph_constructor.node_processor import NodeConstructor
from multi_input.multi_download import MultiDownload
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import utc

app = Flask(__name__)
PATH = "download"
FIELD = "cs.AI"


"""
Note that this program is a in app service. For Kubernetes CronJob, we have the crawler_job.py for it.
Ignore this file. We don't need to host the service for now.
"""


@app.route("/")
def hello():
    return "Hello from Minikube!"

def service():
    md = MultiDownload()
    nc = NodeConstructor()

    # Use today's date in "YYYY-MM-DD" format
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

    print(f"Service run completed at {datetime.datetime.now().isoformat()}, processed {len(ids)} papers.")

if __name__ == "__main__":

    scheduler = BackgroundScheduler(timezone=utc)

    scheduler.add_job(service, 'cron', hour=23, minute=59)
    scheduler.start()

    app.run(host="0.0.0.0", port=5000)
