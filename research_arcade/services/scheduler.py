import logging
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger  # <-- Changed
from dotenv import load_dotenv

from .continuous_crawler import ContinuousCrawlerService

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the scheduler for continuous arXiv crawling."""
    
    # Get configuration from environment
    db_type = os.getenv("DB_TYPE", "csv")
    dest_path = os.getenv("DOWNLOAD_PATH", "./download")
    crawl_interval = int(os.getenv("CRAWL_INTERVAL", "7"))
    run_every_days = int(os.getenv("RUN_EVERY_DAYS", "5"))  # <-- New
    run_on_startup = os.getenv("RUN_ON_STARTUP", "false").lower() == "true"
    
    # Initialize the crawler service
    logger.info("Initializing ContinuousCrawlerService...")
    logger.info(f"Config: db_type={db_type}, crawl_interval={crawl_interval} days")
    
    service = ContinuousCrawlerService(
        dest_path=dest_path,
        db_type=db_type,
        crawl_interval=crawl_interval
    )
    
    # Create scheduler
    scheduler = BlockingScheduler()
    
    # Schedule to run every 5 days
    scheduler.add_job(
        service.run_daily_crawl,
        trigger=IntervalTrigger(days=run_every_days),  # <-- Run every 5 days
        id='periodic_arxiv_crawl',
        name=f'Periodic arXiv paper crawl (every {run_every_days} days)',
        replace_existing=True
    )
    
    logger.info(f"Scheduler started. Will run every {run_every_days} days")
    
    # Run once on startup (recommended for interval-based scheduling)
    if run_on_startup:
        logger.info("Running initial crawl on startup...")
        try:
            stats = service.run_once()
            logger.info(f"Initial crawl completed: {stats}")
        except Exception as e:
            logger.error(f"Initial crawl failed: {e}")
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()