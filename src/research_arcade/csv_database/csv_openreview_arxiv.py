from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
from typing import Optional

class CSVOpenReviewArxiv:
    def __init__(self, csv_dir: str) -> None:
        self.csv_path = csv_dir + "openreview_arxiv.csv"
        self.openreview_crawler = OpenReviewCrawler()
        
        # 如果CSV文件不存在，创建空的DataFrame
        if not os.path.exists(self.csv_path):
            self.create_openreview_arxiv_table()
    
    def create_openreview_arxiv_table(self) -> None:
        columns = ['venue', 'paper_openreview_id', 'arxiv_id', 'title']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(self.csv_path, index=False)
        print(f"Created empty CSV file at {self.csv_path}")
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            return df
        else:
            return None
    
    def _save_data(self, df: pd.DataFrame) -> None:
        df.to_csv(self.csv_path, index=False)
    
    def insert_openreview_arxiv(self, venue: str, paper_openreview_id: str, 
                               arxiv_id: str, title: str) -> Optional[tuple]:
        df = self._load_data()
        
        # 检查是否已存在（基于venue和paper_openreview_id的组合键）
        exists = ((df['venue'] == venue) & 
                 (df['paper_openreview_id'] == paper_openreview_id)).any()
        
        if exists:
            return None
        
        # 创建新行
        new_row = pd.DataFrame([{
            'venue': self._clean_string(venue),
            'paper_openreview_id': self._clean_string(paper_openreview_id),
            'arxiv_id': self._clean_string(arxiv_id),
            'title': self._clean_string(title)
        }])
        
        # 添加到DataFrame并保存
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        
        return (venue, paper_openreview_id)
    
    def delete_openreview_arxiv_by_id(self, paper_openreview_id: str, arxiv_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = (df['paper_openreview_id'] == paper_openreview_id) & (df['arxiv_id'] == arxiv_id)
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No records found in 'openreview_arxiv' with paper_openreview_id = {paper_openreview_id} and arxiv_id = {arxiv_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"Deleted {len(deleted_rows)} records from 'openreview_arxiv' with paper_openreview_id = {paper_openreview_id} and arxiv_id = {arxiv_id}.")
        return deleted_rows
    
    def delete_openreview_arxiv_by_openreview_id(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['paper_openreview_id'] == paper_openreview_id
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No records found in 'openreview_arxiv' with paper_openreview_id = {paper_openreview_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"Deleted {len(deleted_rows)} records from 'openreview_arxiv' with paper_openreview_id = {paper_openreview_id}.")
        return deleted_rows
    
    def delete_openreview_arxiv_by_arxiv_id(self, arxiv_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['arxiv_id'] == arxiv_id
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No records found in 'openreview_arxiv' with arxiv_id = {arxiv_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"Deleted {len(deleted_rows)} records from 'arxiv_id' with arxiv_id = {arxiv_id}.")
        return deleted_rows
    
    def delete_openreview_arxiv_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['venue'] == venue
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No records found in 'openreview_arxiv' for venue = {venue}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"Deleted {len(deleted_rows)} records from 'openreview_arxiv' where venue = {venue}.")
        return deleted_rows
    
    def get_openreview_neighboring_arxivs(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['paper_openreview_id'] == paper_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_arxiv_neighboring_openreviews(self, arxiv_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['arxiv_id'] == arxiv_id
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_openreview_arxiv_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['venue'] == venue
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_all_openreview_arxiv(self) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()
    
    def check_openreview_arxiv_exists(self, venue: str, paper_openreview_id: str) -> bool:
        df = self._load_data()
        exists = ((df['venue'] == venue) & 
                 (df['paper_openreview_id'] == paper_openreview_id)).any()
        return exists
    
    def construct_openreview_arxiv_table_from_api(self, venue: str) -> bool:
        # 从API爬取数据
        print(f"Crawling openreview arxiv data for venue: {venue}...")
        openreview_arxiv_data = self.openreview_crawler.crawl_openreview_arxiv_data_from_api(venue)
        
        # 插入数据
        if len(openreview_arxiv_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(openreview_arxiv_data):
                self.insert_openreview_arxiv(**data)
            return True
        else:
            print("No new openreview arxiv data to insert.")
            return False
    
    def construct_openreview_arxiv_table_from_csv(self, csv_file: str) -> bool:
        if not os.path.exists(csv_file):
            return False
        else:
            print(f"Reading openreview arxiv data from {csv_file}...")
            import_df = pd.read_csv(csv_file)
            openreview_arxiv_data = import_df.to_dict(orient='records')
            
            if len(openreview_arxiv_data) > 0:
                print("Inserting data into CSV file...")
                for data in tqdm(openreview_arxiv_data):
                    self.insert_openreview_arxiv(**data)
                return True
            else:
                print("No new openreview arxiv data to insert.")
                return False
    
    def construct_openreview_arxiv_table_from_json(self, json_file: str) -> bool:
        if not os.path.exists(json_file):
            return False
        else:
            print(f"Reading openreview arxiv data from {json_file}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                openreview_arxiv_data = json.load(f)
            
            if len(openreview_arxiv_data) > 0:
                print("Inserting data into CSV file...")
                for data in tqdm(openreview_arxiv_data):
                    self.insert_openreview_arxiv(**data)
                return True
            else:
                print("No new openreview arxiv data to insert.")
                return False
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s