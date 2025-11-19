import os
import json
from datetime import date, datetime, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from research_arcade.research_arcade import ResearchArcade
from services.crawl_arxiv_ids import crawl_recent_arxiv_paper_new

CATEGORIES = [
    "arxiv_papers", "arxiv_authors", "arxiv_categories", "arxiv_figures",
    "arxiv_tables", "arxiv_sections", "arxiv_paragraphs",
    "arxiv_paper_citation", "arxiv_paper_author", "arxiv_paper_category",
    "arxiv_paper_figure", "arxiv_paper_table", "arxiv_paragraph_reference"
]

LAST_RUN_FILE = "/app/logs/last_crawl.json"


def load_last_run():
    if not os.path.exists(LAST_RUN_FILE):
        return None
    try:
        with open(LAST_RUN_FILE, "r") as f:
            data = json.load(f)
            return date.fromisoformat(data["last_run_date"])
    except:
        return None


def save_last_run(d: date):
    os.makedirs(os.path.dirname(LAST_RUN_FILE), exist_ok=True)
    with open(LAST_RUN_FILE, "w") as f:
        json.dump({"last_run_date": d.isoformat()}, f)


def crawl_once():
    """Main scheduled job: crawl arXiv + ingest into ResearchArcade."""
    print("🚀 Starting scheduled arXiv crawl job...")

    today = date.today()
    last_run = load_last_run()

    # Determine crawling date range
    start_date = last_run or today - timedelta(days=1)
    end_date = today

    print(f"📅 Crawling from {start_date} → {end_date}")

    # 1. Crawl metadata and extract arXiv IDs
    arxiv_ids = crawl_recent_arxiv_paper_new(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        path="/app/download"
    )

    print(f"📥 Extracted {len(arxiv_ids)} arXiv IDs")
    
    # 2. Load ResearchArcade
    research_arcade = ResearchArcade(
        db_type=os.getenv("DB_TYPE", "csv"),
        config={"csv_dir": "/app/data/csv"}
    )

    # 3. Ingest into tables
    config = {"arxiv_ids": arxiv_ids, "dest_dir": "/app/download"}

    for cat in CATEGORIES:
        print(f"📚 Constructing table: {cat}")
        research_arcade.construct_table_from_api(cat, config)

    # 4. Save last run date
    save_last_run(today)
    print("✅ Crawl job completed")


def main():
    scheduler = BlockingScheduler()

    schedule_type = os.getenv("SCHEDULE_TYPE", "interval")

    if schedule_type == "interval":
        minutes = int(os.getenv("CRAWL_INTERVAL", 5))
        scheduler.add_job(crawl_once, IntervalTrigger(minutes=minutes))

    else:  # daily cron
        hour = int(os.getenv("CRAWL_HOUR", 2))
        minute = int(os.getenv("CRAWL_MINUTE", 0))
        scheduler.add_job(crawl_once, CronTrigger(hour=hour, minute=minute))

    # Run immediately at startup if configured
    if os.getenv("RUN_ON_STARTUP", "true").lower() == "true":
        crawl_once()

    print("⏳ Scheduler running...")
    scheduler.start()


if __name__ == "__main__":
    main()
