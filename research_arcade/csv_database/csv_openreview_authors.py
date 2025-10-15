from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
from typing import Optional

class CSVOpenReviewAuthors:
    def __init__(self, csv_path: str = "authors.csv"):
        self.csv_path = csv_path
        self.openreview_crawler = OpenReviewCrawler()
        
        # 如果CSV文件不存在，创建空的DataFrame
        if not os.path.exists(csv_path):
            self.create_author_table()
    
    def create_author_table(self):
        columns = ['venue', 'author_openreview_id', 'author_full_name', 'email', 
                   'affiliation', 'homepage', 'dblp']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(self.csv_path, index=False)
        print(f"Created empty CSV file at {self.csv_path}")
    
    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            return df
    
    def _save_data(self, df: pd.DataFrame):
        df.to_csv(self.csv_path, index=False)
    
    def insert_author(self, venue: str, author_openreview_id: str, 
                     author_full_name: str, email: str, affiliation: str, 
                     homepage: str, dblp: str) -> Optional[tuple]:
        df = self._load_data()
        
        # 检查是否已存在（基于venue和author_openreview_id的组合键）
        exists = ((df['venue'] == venue) & 
                 (df['author_openreview_id'] == author_openreview_id)).any()
        
        if exists:
            return None
        
        # 创建新行
        new_row = pd.DataFrame([{
            'venue': self._clean_string(venue),
            'author_openreview_id': self._clean_string(author_openreview_id),
            'author_full_name': self._clean_string(author_full_name),
            'email': self._clean_string(email),
            'affiliation': self._clean_string(affiliation),
            'homepage': self._clean_string(homepage),
            'dblp': self._clean_string(dblp)
        }])
        
        # 添加到DataFrame并保存
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        
        return (venue, author_openreview_id)
    
    def delete_author_by_id(self, author_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['author_openreview_id'] == author_openreview_id
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No author found with author_openreview_id {author_openreview_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"Author with author_openreview_id {author_openreview_id} deleted successfully.")
        return deleted_rows
    
    def delete_authors_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['venue'] == venue
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No authors found in venue {venue}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"All authors in venue {venue} deleted successfully.")
        return deleted_rows
    
    def update_author(self, venue: str, author_openreview_id: str, 
                     author_full_name: str, email: str, affiliation: str, 
                     homepage: str, dblp: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要更新的行
        mask = ((df['venue'] == venue) & 
               (df['author_openreview_id'] == author_openreview_id))
        
        if not mask.any():
            print(f"No author found with author_openreview_id {author_openreview_id}.")
            return None
        
        # 保存原始记录
        original_record = df[mask].copy()
        
        # 更新记录
        df.loc[mask, 'author_full_name'] = self._clean_string(author_full_name)
        df.loc[mask, 'email'] = self._clean_string(email)
        df.loc[mask, 'affiliation'] = self._clean_string(affiliation)
        df.loc[mask, 'homepage'] = self._clean_string(homepage)
        df.loc[mask, 'dblp'] = self._clean_string(dblp)
        
        self._save_data(df)
        
        print(f"Author with author_openreview_id {author_openreview_id} updated successfully.")
        return original_record
    
    def get_author_by_id(self, author_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['author_openreview_id'] == author_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            print(f"No author found with author_openreview_id {author_openreview_id}.")
            return None
        
        return result
    
    def get_authors_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['venue'] == venue
        result = df[mask].copy()
        
        if result.empty:
            print(f"No authors found in venue {venue}.")
            return None
        
        return result
    
    def get_all_authors(self, is_all_features: bool = False) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        # 按author_openreview_id排序
        df = df.sort_values('author_openreview_id')
        
        if is_all_features:
            return df[['venue', 'author_openreview_id', 'author_full_name', 
                      'email', 'affiliation', 'homepage', 'dblp']].copy()
        else:
            return df[['venue', 'author_openreview_id', 'author_full_name']].copy()
    
    def check_author_exists(self, author_openreview_id: str) -> bool:
        df = self._load_data()
        return (df['author_openreview_id'] == author_openreview_id).any()
    
    def construct_authors_table_from_api(self, venue: str):
        # 从API爬取数据
        print("Crawling author data from OpenReview API...")
        author_data = self.openreview_crawler.crawl_author_data_from_api(venue)
        
        # 插入数据
        if len(author_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(author_data):
                self.insert_author(**data)
        else:
            print("No new author data to insert.")
    
    def construct_authors_table_from_json(self, json_file: str):
        print(f"Reading authors data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            author_data = json.load(f)
        
        if len(author_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(author_data):
                self.insert_author(**data)
        else:
            print("No new author data to insert.")
    
    def construct_authors_table_from_csv(self, csv_file: str):
        print(f"Reading authors data from {csv_file}...")
        import_df = pd.read_csv(csv_file)
        author_data = import_df.to_dict(orient='records')
        
        if len(author_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(author_data):
                self.insert_author(**data)
        else:
            print("No new author data to insert.")
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s