from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
from typing import Optional

class CSVOpenReviewParagraphs:
    def __init__(self, csv_dir: str = "./"):
        self.csv_path = csv_dir + "openreview_paragraphs.csv"
        self.openreview_crawler = OpenReviewCrawler()
        
        # 如果CSV文件不存在，创建空的DataFrame
        if not os.path.exists(self.csv_path):
            self.create_paragraphs_table()
    
    def create_paragraphs_table(self):
        columns = ['venue', 'paper_openreview_id', 'paragraph_idx', 'section', 'content']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(self.csv_path, index=False)
        print(f"Created empty CSV file at {self.csv_path}")
    
    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            return df
    
    def _save_data(self, df: pd.DataFrame):
        df.to_csv(self.csv_path, index=False)
    
    def insert_paragraph(self, venue: str, paper_openreview_id: str, paragraph_idx: int, section: str, content: str) -> Optional[tuple]:
        df = self._load_data()
        
        # 检查是否已存在（基于venue和paper_openreview_id的组合键）
        exists = ((df['venue'] == venue) & 
                 (df['paper_openreview_id'] == paper_openreview_id) &
                 (df['paragraph_idx'] == paragraph_idx)).any()
        
        if exists:
            return None
        
        # 创建新行
        new_row = pd.DataFrame([{
            'venue': self._clean_string(venue),
            'paper_openreview_id': self._clean_string(paper_openreview_id),
            'paragraph_idx': paragraph_idx,
            'section': self._clean_string(section),
            'content': self._clean_string(content)
        }])
        
        # 添加到DataFrame并保存
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        
        return (venue, paper_openreview_id, paragraph_idx)     
    
    def get_all_paragraphs(self, is_all_features: bool = False) -> Optional[pd.DataFrame]:
        """获取所有论文"""
        df = self._load_data()
        
        if df.empty:
            return None
        
        if is_all_features:
            return df[['venue', 'paper_openreview_id', 'paragraph_idx', 
                      'section', 'content']].copy()
        else:
            return df[['venue', 'paper_openreview_id', 'paragraph_idx', 'section']].copy()
    
    def get_paragraphs_by_paper_id(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['paper_openreview_id'] == paper_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            print(f"No paragraph found with paper_openreview_id {paper_openreview_id}.")
            return None
        
        return result
    
    def get_paragraphs_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        """根据venue获取所有论文"""
        df = self._load_data()
        
        mask = df['venue'] == venue
        result = df[mask].copy()
        
        if result.empty:
            print(f"No paragraph found in venue {venue}.")
            return None
        
        return result
    
    def delete_paragraphs_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['venue'] == venue
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No paragraphs found in venue {venue}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"All paragraphs in venue {venue} deleted successfully.")
        return deleted_rows
    
    def delete_paragraphs_by_paper_id(self, paper_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['paper_openreview_id'] == paper_openreview_id
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No paragraph found with paper_openreview_id {paper_openreview_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"Paragraphs with paper_openreview_id {paper_openreview_id} deleted successfully.")
        return deleted_rows
    
    def construct_paragraphs_table_from_api(self, venue: str, pdf_dir: str, filter_list: list, log_file: str, 
                                            is_paper = True, is_revision = True, is_pdf_delete: bool = True):
        # 从API爬取数据
        print("Crawling paragraph data from OpenReview API...")
        paragraph_data = self.openreview_crawler.crawl_paragraph_data_from_api(venue, pdf_dir, filter_list, log_file,
                                                                           is_paper, is_revision, is_pdf_delete)
        
        # 插入数据
        if len(paragraph_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(paragraph_data):
                self.insert_paragraph(**data)
        else:
            print("No new paragraph data to insert.")
    
    def construct_paragraphs_table_from_csv(self, csv_file: str):
        if not os.path.exists(csv_file):
            print(f"File {csv_file} not exists")
            return False
        
        print(f"Reading paragraph data from {csv_file}...")
        import_df = pd.read_csv(csv_file)
        paper_data = import_df.to_dict(orient='records')
        
        if len(paper_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(paper_data):
                self.insert_paragraph(**data)
            return True
        else:
            print("No new paragraph data to insert.")
            return False
    
    def construct_paragraphs_table_from_json(self, json_file: str):
        if not os.path.exists(json_file):
            print(f"File {json_file} not exists")
            return False
        
        print(f"Reading paper data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        if len(paper_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(paper_data):
                self.insert_paragraph(**data)
            return True
        else:
            print("No new paper data to insert.")
            return False
    
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s