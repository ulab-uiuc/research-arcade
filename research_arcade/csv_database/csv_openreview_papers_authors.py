from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
from typing import Optional

class CSVOpenReviewPapersAuthors:
    def __init__(self, csv_path: str = "papers_authors.csv", 
                 papers_csv: Optional[str] = "papers.csv", 
                 authors_csv: Optional[str] = "authors.csv"):
        self.csv_path = csv_path
        self.papers_csv = papers_csv
        self.authors_csv = authors_csv
        self.openreview_crawler = OpenReviewCrawler()
        
        # 如果CSV文件不存在，创建空的DataFrame
        if not os.path.exists(csv_path):
            self.create_papers_authors_table()
    
    def create_papers_authors_table(self):
        columns = ['venue', 'paper_openreview_id', 'author_openreview_id']
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
            return False
        
        papers_df = pd.read_csv(self.papers_csv)
        return (papers_df['paper_openreview_id'] == paper_openreview_id).any()
    
    def check_author_exists(self, author_openreview_id: str) -> bool:
        if self.authors_csv is None or not os.path.exists(self.authors_csv):
            return False
        
        authors_df = pd.read_csv(self.authors_csv)
        return (authors_df['author_openreview_id'] == author_openreview_id).any()
    
    def insert_paper_authors(self, venue: str, paper_openreview_id: str, 
                            author_openreview_id: str) -> Optional[tuple]:
        # 验证论文和作者是否存在
        if not self.check_paper_exists(paper_openreview_id):
            print(f"The paper {paper_openreview_id} does not exist in the database.")
            return None
        
        if not self.check_author_exists(author_openreview_id):
            print(f"The author {author_openreview_id} does not exist in the database.")
            return None
        
        df = self._load_data()
        
        # 检查是否已存在（基于三个字段的组合键）
        exists = ((df['venue'] == venue) & 
                 (df['paper_openreview_id'] == paper_openreview_id) &
                 (df['author_openreview_id'] == author_openreview_id)).any()
        
        if exists:
            return None
        
        # 创建新行
        new_row = pd.DataFrame([{
            'venue': self._clean_string(venue),
            'paper_openreview_id': self._clean_string(paper_openreview_id),
            'author_openreview_id': self._clean_string(author_openreview_id)
        }])
        
        # 添加到DataFrame并保存
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        
        return (venue, paper_openreview_id, author_openreview_id)
    
    def delete_paper_author_by_id(self, paper_openreview_id: str, 
                                  author_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = ((df['paper_openreview_id'] == paper_openreview_id) & 
                (df['author_openreview_id'] == author_openreview_id))
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No connection found between paper {paper_openreview_id} and author {author_openreview_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"The connection between paper {paper_openreview_id} and author {author_openreview_id} is deleted successfully.")
        return deleted_rows
    
    def delete_papers_authors_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
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
    
    def get_paper_neighboring_authors(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['paper_openreview_id'] == paper_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_author_neighboring_papers(self, author_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['author_openreview_id'] == author_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_papers_authors_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['venue'] == venue
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_all_papers_authors(self) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        # 按paper_openreview_id排序
        df = df.sort_values('paper_openreview_id')
        return df.copy()
    
    def check_paper_author_exists(self, paper_openreview_id: str, 
                                  author_openreview_id: str) -> bool:
        df = self._load_data()
        exists = ((df['paper_openreview_id'] == paper_openreview_id) & 
                 (df['author_openreview_id'] == author_openreview_id)).any()
        return exists
    
    def construct_papers_authors_table_from_api(self, venue: str):
        # 从API爬取数据
        print(f"Crawling papers-authors data for venue {venue} from OpenReview API...")
        papers_authors_data = self.openreview_crawler.crawl_papers_authors_data_from_api(venue)
        
        if len(papers_authors_data) > 0:
            print(f"Inserting papers-authors data for venue {venue}...")
            for data in tqdm(papers_authors_data):
                self.insert_paper_authors(**data)
        else:
            print(f"No papers-authors data found for venue {venue}.")
    
    def construct_papers_authors_table_from_csv(self, csv_file: str):
        print(f"Reading papers-authors data from {csv_file}...")
        import_df = pd.read_csv(csv_file)
        papers_authors_data = import_df.to_dict(orient='records')
        
        if len(papers_authors_data) > 0:
            print(f"Inserting papers-authors data from {csv_file}...")
            for data in tqdm(papers_authors_data):
                self.insert_paper_authors(**data)
        else:
            print(f"No papers-authors data found in {csv_file}.")
    
    def construct_papers_authors_table_from_json(self, json_file: str):
        print(f"Reading papers-authors data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            papers_authors_data = json.load(f)
        
        if len(papers_authors_data) > 0:
            print(f"Inserting papers-authors data from {json_file}...")
            for data in tqdm(papers_authors_data):
                self.insert_paper_authors(**data)
        else:
            print(f"No papers-authors data found in {json_file}.")
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s