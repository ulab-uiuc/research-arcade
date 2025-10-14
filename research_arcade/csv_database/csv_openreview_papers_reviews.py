from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
from typing import Optional

class CSVOpenReviewPapersReviews:
    def __init__(self, csv_path: str = "papers_reviews.csv", 
                 papers_csv: Optional[str] = "papers.csv", 
                 reviews_csv: Optional[str] = "reviews.csv"):
        self.csv_path = csv_path
        self.papers_csv = papers_csv
        self.reviews_csv = reviews_csv
        self.openreview_crawler = OpenReviewCrawler()
        
        # 如果CSV文件不存在，创建空的DataFrame
        if not os.path.exists(csv_path):
            self.create_papers_reviews_table()
    
    def create_papers_reviews_table(self):
        columns = ['venue', 'paper_openreview_id', 'review_openreview_id', 
                   'title', 'time']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(self.csv_path, index=False)
        print(f"Created empty CSV file at {self.csv_path}")
    
    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            return df
    
    def _save_data(self, df: pd.DataFrame):
        df.to_csv(self.csv_path, index=False)
    
    def check_paper_exists(self, paper_openreview_id: str) -> bool:
        if self.papers_csv is None or not os.path.exists(self.papers_csv):
            # 如果没有指定论文表，跳过验证
            return False
        
        papers_df = pd.read_csv(self.papers_csv)
        return (papers_df['paper_openreview_id'] == paper_openreview_id).any()
    
    def check_review_exists(self, review_openreview_id: str) -> bool:
        if self.reviews_csv is None or not os.path.exists(self.reviews_csv):
            # 如果没有指定评审表，跳过验证
            return False
        
        reviews_df = pd.read_csv(self.reviews_csv)
        return (reviews_df['review_openreview_id'] == review_openreview_id).any()
    
    def insert_paper_reviews(self, venue: str, paper_openreview_id: str, 
                            review_openreview_id: str, title: str, 
                            time: str) -> Optional[tuple]:
        # 验证论文和评审是否存在
        if not self.check_paper_exists(paper_openreview_id):
            print(f"The paper {paper_openreview_id} does not exist in the database.")
            return None
        
        if not self.check_review_exists(review_openreview_id):
            print(f"The review {review_openreview_id} does not exist in the database.")
            return None
        
        df = self._load_data()
        
        # 检查是否已存在（基于三个字段的组合键）
        exists = ((df['venue'] == venue) & 
                 (df['paper_openreview_id'] == paper_openreview_id) &
                 (df['review_openreview_id'] == review_openreview_id)).any()
        
        if exists:
            return None
        
        # 创建新行
        new_row = pd.DataFrame([{
            'venue': self._clean_string(venue),
            'paper_openreview_id': self._clean_string(paper_openreview_id),
            'review_openreview_id': self._clean_string(review_openreview_id),
            'title': self._clean_string(title),
            'time': self._clean_string(time)
        }])
        
        # 添加到DataFrame并保存
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        
        return (venue, paper_openreview_id, review_openreview_id)
    
    def delete_paper_review_by_id(self, paper_openreview_id: str, 
                                  review_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = ((df['paper_openreview_id'] == paper_openreview_id) & 
                (df['review_openreview_id'] == review_openreview_id))
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No connection found between paper {paper_openreview_id} and review {review_openreview_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"The connection between paper {paper_openreview_id} and review {review_openreview_id} is deleted successfully.")
        return deleted_rows
    
    def delete_papers_reviews_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['venue'] == venue
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No connections found in venue {venue}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"All connections in venue {venue} deleted successfully.")
        return deleted_rows
    
    def get_papers_reviews_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['venue'] == venue
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_all_papers_reviews(self) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        # 按paper_openreview_id排序
        df = df.sort_values('paper_openreview_id')
        return df.copy()
    
    def check_paper_review_exists(self, paper_openreview_id: str, 
                                  review_openreview_id: str) -> bool:
        df = self._load_data()
        exists = ((df['paper_openreview_id'] == paper_openreview_id) & 
                 (df['review_openreview_id'] == review_openreview_id)).any()
        return exists
    
    def get_paper_neighboring_reviews(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['paper_openreview_id'] == paper_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_review_neighboring_papers(self, review_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['review_openreview_id'] == review_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def construct_papers_reviews_table_from_api(self, venue: str):
        # 从API爬取数据
        print(f"Crawling paper-review connections for venue: {venue}...")
        papers_reviews_data = self.openreview_crawler.crawl_papers_reviews_from_api(venue)
        
        if len(papers_reviews_data) > 0:
            print(f"Inserting paper-review connections into CSV file for venue: {venue}...")
            for data in tqdm(papers_reviews_data):
                self.insert_paper_reviews(**data)
        else:
            print(f"No paper-review connections found for venue: {venue}.")
    
    def construct_papers_reviews_table_from_csv(self, csv_file: str):
        print(f"Reading paper-review connection data from {csv_file}...")
        import_df = pd.read_csv(csv_file)
        papers_reviews_data = import_df.to_dict(orient='records')
        
        if len(papers_reviews_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(papers_reviews_data):
                self.insert_paper_reviews(**data)
        else:
            print("No new paper-review connection data to insert.")
    
    def construct_papers_reviews_table_from_json(self, json_file: str):
        print(f"Reading paper-review connection data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            papers_reviews_data = json.load(f)
        
        if len(papers_reviews_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(papers_reviews_data):
                self.insert_paper_reviews(**data)
        else:
            print("No new paper-review connection data to insert.")
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return re.sub(r'[\x00-\x1F\x7F]', '', s)
        return s