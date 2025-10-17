from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
from typing import Optional

class CSVOpenReviewPapers:
    def __init__(self, csv_dir: str = "./"):
        self.csv_path = csv_dir + "openreview_papers.csv"
        self.openreview_crawler = OpenReviewCrawler()
        
        # 如果CSV文件不存在，创建空的DataFrame
        if not os.path.exists(self.csv_path):
            self.create_papers_table()
    
    def create_papers_table(self):
        columns = ['venue', 'paper_openreview_id', 'title', 'abstract', 
                   'paper_decision', 'paper_pdf_link']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(self.csv_path, index=False)
        print(f"Created empty CSV file at {self.csv_path}")
    
    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            return df
    
    def _save_data(self, df: pd.DataFrame):
        df.to_csv(self.csv_path, index=False)
    
    def insert_paper(self, venue: str, paper_openreview_id: str, title: str, 
                    abstract: str, paper_decision: str, 
                    paper_pdf_link: str) -> Optional[tuple]:
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
            'title': self._clean_string(title),
            'abstract': self._clean_string(abstract),
            'paper_decision': self._clean_string(paper_decision),
            'paper_pdf_link': self._clean_string(paper_pdf_link)
        }])
        
        # 添加到DataFrame并保存
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        
        return (venue, paper_openreview_id)
    
    def delete_paper_by_id(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['paper_openreview_id'] == paper_openreview_id
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No paper found with paper_openreview_id {paper_openreview_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"Paper with paper_openreview_id {paper_openreview_id} deleted successfully.")
        return deleted_rows
    
    def delete_papers_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['venue'] == venue
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No papers found in venue {venue}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"All papers in venue {venue} deleted successfully.")
        return deleted_rows
    
    def update_paper(self, venue: str, paper_openreview_id: str, title: str, 
                    abstract: str, paper_decision: str, 
                    paper_pdf_link: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要更新的行
        mask = ((df['venue'] == venue) & 
               (df['paper_openreview_id'] == paper_openreview_id))
        
        if not mask.any():
            print(f"No paper found with paper_openreview_id {paper_openreview_id}.")
            return None
        
        # 保存原始记录
        original_record = df[mask].copy()
        
        # 更新记录
        df.loc[mask, 'title'] = self._clean_string(title)
        df.loc[mask, 'abstract'] = self._clean_string(abstract)
        df.loc[mask, 'paper_decision'] = self._clean_string(paper_decision)
        df.loc[mask, 'paper_pdf_link'] = self._clean_string(paper_pdf_link)
        
        self._save_data(df)
        
        print(f"Paper with paper_openreview_id {paper_openreview_id} updated successfully.")
        return original_record
    
    def get_paper_by_id(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        """根据paper_openreview_id获取论文"""
        df = self._load_data()
        
        mask = df['paper_openreview_id'] == paper_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            print(f"No paper found with paper_openreview_id {paper_openreview_id}.")
            return None
        
        return result
    
    def get_papers_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        """根据venue获取所有论文"""
        df = self._load_data()
        
        mask = df['venue'] == venue
        result = df[mask].copy()
        
        if result.empty:
            print(f"No papers found in venue {venue}.")
            return None
        
        return result
    
    def get_all_papers(self, is_all_features: bool = False) -> Optional[pd.DataFrame]:
        """获取所有论文"""
        df = self._load_data()
        
        if df.empty:
            return None
        
        if is_all_features:
            return df[['venue', 'paper_openreview_id', 'title', 'abstract', 
                      'paper_decision', 'paper_pdf_link']].copy()
        else:
            return df[['venue', 'paper_openreview_id', 'title']].copy()
    
    def check_paper_exists(self, paper_openreview_id: str) -> bool:
        """检查论文是否存在"""
        df = self._load_data()
        return (df['paper_openreview_id'] == paper_openreview_id).any()
    
    def construct_papers_table_from_api(self, venue: str):
        # 从API爬取数据
        print("Crawling paper data from OpenReview API...")
        paper_data = self.openreview_crawler.crawl_paper_data_from_api(venue)
        
        # 插入数据
        if len(paper_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(paper_data):
                self.insert_paper(**data)
        else:
            print("No new paper data to insert.")
    
    def construct_papers_table_from_csv(self, csv_file: str):
        print(f"Reading paper data from {csv_file}...")
        import_df = pd.read_csv(csv_file)
        paper_data = import_df.to_dict(orient='records')
        
        if len(paper_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(paper_data):
                self.insert_paper(**data)
        else:
            print("No new paper data to insert.")
    
    def construct_papers_table_from_json(self, json_file: str):
        print(f"Reading paper data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        if len(paper_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(paper_data):
                self.insert_paper(**data)
        else:
            print("No new paper data to insert.")
    
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s