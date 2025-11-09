import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv

from ..arxiv_utils.arxiv_id_crawler import arxiv_id_crawler
from ..research_arcade import ResearchArcade

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ContinuousCrawlerService:
    """Service for continuous arXiv paper crawling."""
    
    def __init__(self, dest_path: str = "./download", db_type: str = "csv", crawl_interval: int = 7):
        """
        Initialize the continuous crawler service.
        
        Args:
            dest_path: Directory to download papers
            db_type: Database type ('csv' or 'sql')
            crawl_interval: Number of days to look back for papers
        """
        # Length of crawling interval in days
        self.crawl_interval = crawl_interval

        db_type = db_type.lower()
        if db_type not in {"csv", "sql"}:
            raise ValueError(f"Database type '{db_type}' not supported. Use 'csv' or 'sql'.")
        
        self.db_type = db_type
        self.dest_path = dest_path
        
        # Initialize database configuration
        self.config = self._load_config()
        
        # Track statistics
        self.last_run: Optional[datetime] = None
        
    def _load_config(self) -> Dict:
        """Load database configuration from environment variables."""
        if self.db_type == "csv":
            csv_dir = os.getenv("CSV_DATASET_FOLDER_PATH")
            if not csv_dir:
                raise ValueError("CSV_DATASET_FOLDER_PATH not set in environment")
            return {"csv_dir": csv_dir}
        
        elif self.db_type == "sql":
            config = {
                "host": os.getenv("HOST"),
                "port": os.getenv("PORT"),
                "dbname": os.getenv("DBNAME"),
                "user": os.getenv("USER"),
                "password": os.getenv("PASSWORD", None)  # Optional
            }
            
            missing = [k for k, v in config.items() if v is None and k != "password"]
            if missing:
                raise ValueError(f"Missing required SQL environment variables: {missing}")
            
            return config
        
        return {}
    
    def _calculate_date_range(self) -> tuple[str, str]:

        end_date = datetime.today()
        start_date = end_date - timedelta(days=self.crawl_interval - 1)
        

        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        
    def run_daily_crawl(self) -> Dict:
        """Execute daily crawl and return statistics."""
        logger.info(f"Starting crawl at {datetime.now()}")
        
        try:
            # Calculate date range
            start_date, end_date = self._calculate_date_range()
            logger.info(f"Crawling papers from {start_date} to {end_date}")
            
            # Get arXiv IDs for the date range
            arxiv_ids = arxiv_id_crawler(start_date=start_date, end_date=end_date)
            logger.info(f"Found {len(arxiv_ids)} papers")
            
            if not arxiv_ids:
                logger.info("No papers found for the specified date range")
                return {
                    'total_found': 0,
                    'processed': 0,
                    'failed': 0,
                    'failed_ids': []
                }
            
            # Initialize ResearchArcade
            ra = ResearchArcade(db_type=self.db_type, config=self.config)
            
            # Prepare input for batch processing
            input_config = {
                "arxiv_ids": arxiv_ids,
                "dest_dir": self.dest_path
            }
            
            # Process papers
            logger.info("Starting paper processing...")
            result = ra.construct_arxiv_tables_from_api(config=input_config)
            
            # Update last run timestamp
            self.last_run = datetime.now()
            
            success_count = len(arxiv_ids)  # Assuming all succeed for now
            failed_ids = []  # You may want to track failures in ResearchArcade
            
            
            stats = {
                'timestamp': self.last_run.isoformat(),
                'date_range': {'start': start_date, 'end': end_date},
                'total_found': len(arxiv_ids),
                'processed': success_count,
                'failed': len(failed_ids),
                'failed_ids': failed_ids
            }
            
            logger.info(f"Crawl completed successfully: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Daily crawl failed: {e}", exc_info=True)
            raise
    
    def run_once(self) -> Dict:
        """
        Run the crawler once immediately (useful for testing).
        Alias for run_daily_crawl.
        """
        return self.run_daily_crawl()
    
    def get_status(self) -> Dict:
        """Get current status of the crawler service."""
        return {
            'db_type': self.db_type,
            'dest_path': self.dest_path,
            'crawl_interval': self.crawl_interval,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'config_loaded': bool(self.config)
        }