from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
from typing import Optional

class CSVOpenReviewPapersRevisions:
    def __init__(self, csv_dir: str) -> None:
        self.csv_path = csv_dir + "openreview_papers_revisions.csv"
        self.openreview_crawler = OpenReviewCrawler()
        
        # 如果CSV文件不存在，创建空的DataFrame
        if not os.path.exists(self.csv_path):
            self.create_papers_revisions_table()
    
    def create_papers_revisions_table(self) -> None:
        columns = ['venue', 'paper_openreview_id', 'revision_openreview_id', 
                   'title', 'time']
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
        """将DataFrame保存到CSV文件"""
        df.to_csv(self.csv_path, index=False)
    
    def insert_paper_revisions(self, venue: str, paper_openreview_id: str, 
                              revision_openreview_id: str, title: str, 
                              time: str) -> Optional[tuple]:
        df = self._load_data()
        
        # 检查是否已存在（基于三个字段的组合键）
        exists = ((df['venue'] == venue) & 
                 (df['paper_openreview_id'] == paper_openreview_id) &
                 (df['revision_openreview_id'] == revision_openreview_id)).any()
        
        if exists:
            return None
        
        # 创建新行
        new_row = pd.DataFrame([{
            'venue': self._clean_string(venue),
            'paper_openreview_id': self._clean_string(paper_openreview_id),
            'revision_openreview_id': self._clean_string(revision_openreview_id),
            'title': self._clean_string(title),
            'time': self._clean_string(time)
        }])
        
        # 添加到DataFrame并保存
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        
        return (venue, paper_openreview_id, revision_openreview_id)
    
    def delete_paper_revision_by_id(self, paper_openreview_id: str, 
                                   revision_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = ((df['paper_openreview_id'] == paper_openreview_id) & 
                (df['revision_openreview_id'] == revision_openreview_id))
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No connection found between paper {paper_openreview_id} and revision {revision_openreview_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"The connection between paper {paper_openreview_id} and revision {revision_openreview_id} is deleted successfully.")
        return deleted_rows
    
    def delete_paper_revision_by_paper_id(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['paper_openreview_id'] == paper_openreview_id
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No connection found for paper {paper_openreview_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"The connection for paper {paper_openreview_id} is deleted successfully.")
        return deleted_rows
    
    def delete_paper_revision_by_revision_id(self, revision_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['revision_openreview_id'] == revision_openreview_id
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No connection found for revision {revision_openreview_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"The connection for revision {revision_openreview_id} is deleted successfully.")
        return deleted_rows
    
    def delete_papers_revisions_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
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
    
    def get_papers_revisions_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['venue'] == venue
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_all_papers_revisions(self) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        # 按paper_openreview_id排序
        df = df.sort_values('paper_openreview_id')
        return df.copy()
    
    def check_paper_revision_exists(self, paper_openreview_id: str, 
                                   revision_openreview_id: str) -> bool:
        df = self._load_data()
        exists = ((df['paper_openreview_id'] == paper_openreview_id) & 
                 (df['revision_openreview_id'] == revision_openreview_id)).any()
        return exists
    
    def get_paper_neighboring_revisions(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['paper_openreview_id'] == paper_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_revision_neighboring_papers(self, revision_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['revision_openreview_id'] == revision_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def construct_papers_revisions_table_from_api(self, venue: str) -> bool:
        # 从API爬取数据
        print(f"Crawling paper-revision data from OpenReview API for venue: {venue}...")
        paper_revision_data = self.openreview_crawler.crawl_papers_revisions_data_from_api(venue)
        
        if len(paper_revision_data) > 0:
            print(f"Inserting paper-revision data into CSV file for venue: {venue}...")
            for data in tqdm(paper_revision_data):
                self.insert_paper_revisions(**data)
            return True
        else:
            print(f"No paper-revision data found for venue: {venue}.")
            return False
    
    def construct_papers_revisions_table_from_csv(self, csv_file: str) -> bool:
        if not os.path.exists(csv_file):
            return False
        else:
            print(f"Reading paper-revision data from {csv_file}...")
            import_df = pd.read_csv(csv_file)
            revision_data = import_df.to_dict(orient='records')
            
            if len(revision_data) > 0:
                print(f"Inserting paper-revision data from {csv_file} into CSV file...")
                for data in tqdm(revision_data):
                    self.insert_paper_revisions(**data)
                return True
            else:
                print(f"No paper-revision data found in {csv_file}.")
                return False
    
    def construct_papers_revisions_table_from_json(self, json_file: str) -> bool:
        if not os.path.exists(json_file):
            return False
        else:
            print(f"Reading revisions data from {json_file}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                revision_data = json.load(f)
            
            if len(revision_data) > 0:
                print(f"Inserting paper-revision data from {json_file} into CSV file...")
                for data in tqdm(revision_data):
                    self.insert_paper_revisions(**data)
                return True
            else:
                print(f"No paper-revision data found in {json_file}.")
                return False
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s