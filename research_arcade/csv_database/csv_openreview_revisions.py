from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
import ast
from typing import Optional, Union

class CSVOpenReviewRevisions:
    def __init__(self, csv_dir: str) -> None:
        self.csv_path = csv_path = os.path.join(csv_dir, 'openreview_revisions.csv')
        self.openreview_crawler = OpenReviewCrawler()
        
        # 如果CSV文件不存在，创建空的DataFrame
        if not os.path.exists(self.csv_path):
            self.create_revisions_table()
    
    def create_revisions_table(self) -> None:
        columns = ['venue', 'original_openreview_id', 'revision_openreview_id', 
                   'content', 'time']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(self.csv_path, index=False)
        print(f"Created empty CSV file at {self.csv_path}")
    
    def _load_data(self) -> Optional[pd.DataFrame]:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            # 将content列从JSON字符串转换为字典
            if not df.empty:
                df['content'] = df['content'].apply(
                    lambda x: ast.literal_eval(str(x)) if pd.notna(x) and x != '' else []
                )
            return df
        else:
            return None
    
    def _save_data(self, df: pd.DataFrame) -> None:
        # 创建副本以避免修改原始数据
        df_to_save = df.copy()
        # 将content列从字典转换为JSON字符串
        if 'content' in df_to_save.columns:
            df_to_save['content'] = df_to_save['content'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )
        df_to_save.to_csv(self.csv_path, index=False)
    
    def sample_revisions(self, sample_size: int) -> Optional[pd.DataFrame]:
        """Sample a number of revisions randomly. Returns a DataFrame with sampled revisions or None if empty."""
        df = self._load_data()
        
        if df.empty:
            return None
        
        sampled_df = df.sample(n=sample_size)
        return sampled_df.reset_index(drop=True)
    
    def insert_revision(self, venue: str, original_openreview_id: str, 
                       revision_openreview_id: str, content: dict, 
                       time: str) -> Optional[tuple]:
        df = self._load_data()
        
        # 检查是否已存在（基于venue和revision_openreview_id的组合键）
        exists = ((df['venue'] == venue) & 
                 (df['revision_openreview_id'] == revision_openreview_id)).any()
        
        if exists:
            return None
        
        # 清理content
        cleaned_content = self._clean_json_content(content)
        
        # 创建新行
        new_row = pd.DataFrame([{
            'venue': self._clean_string(venue),
            'original_openreview_id': self._clean_string(original_openreview_id),
            'revision_openreview_id': self._clean_string(revision_openreview_id),
            'content': cleaned_content,
            'time': self._clean_string(time)
        }])
        
        # 添加到DataFrame并保存
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        
        return (venue, revision_openreview_id)
    
    def delete_revision_by_id(self, revision_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['revision_openreview_id'] == revision_openreview_id
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No revision found with revision_openreview_id {revision_openreview_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"Revision with revision_openreview_id {revision_openreview_id} deleted successfully.")
        return deleted_rows
    
    def delete_revisions_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['venue'] == venue
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No revisions found in venue {venue}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"All revisions in venue {venue} deleted successfully.")
        return deleted_rows
    
    def update_revision(self, venue: str, original_openreview_id: str, 
                       revision_openreview_id: str, content: dict, 
                       time: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要更新的行
        mask = ((df['venue'] == venue) & 
               (df['revision_openreview_id'] == revision_openreview_id))
        
        if not mask.any():
            print(f"No revision found with revision_openreview_id {revision_openreview_id}.")
            return None
        
        # 保存原始记录
        original_record = df[mask].copy()
        
        # 更新记录
        cleaned_content = self._clean_json_content(content)
        df.loc[mask, 'original_openreview_id'] = self._clean_string(original_openreview_id)
        df.loc[mask, 'content'] = str(cleaned_content)
        df.loc[mask, 'time'] = self._clean_string(time)
        
        self._save_data(df)
        
        print(f"Revision with revision_openreview_id {revision_openreview_id} updated successfully.")
        return original_record
    
    def get_revision_by_id(self, revision_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['revision_openreview_id'] == revision_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            print(f"No revision found with revision_openreview_id {revision_openreview_id}.")
            return None
        
        return result
    
    def get_revisions_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['venue'] == venue
        result = df[mask].copy()
        
        if result.empty:
            print(f"No revisions found in venue {venue}.")
            return None
        
        return result
    
    def get_all_revisions(self, is_all_features: bool = False) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        if is_all_features:
            return df[['venue', 'original_openreview_id', 'revision_openreview_id', 
                      'content', 'time']].copy()
        else:
            return df[['venue', 'original_openreview_id', 'revision_openreview_id', 
                      'time']].copy()
    
    def check_revision_exists(self, revision_openreview_id: str) -> bool:
        df = self._load_data()
        return (df['revision_openreview_id'] == revision_openreview_id).any()
    
    def construct_revisions_table_from_api(self, venue: str, filter_list: list, 
                                          pdf_dir: str, log_file: str) -> bool:
        # 从API爬取数据
        print("Crawling revision data from OpenReview API...")
        revision_data = self.openreview_crawler.crawl_revision_data_from_api(
            venue, filter_list, pdf_dir, log_file
        )
        
        # 插入数据
        if len(revision_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(revision_data):
                self.insert_revision(**data)
        else:
            print("No new revision data to insert.")
    
    def construct_revisions_table_from_csv(self, csv_file: str) -> bool:
        if not os.path.exists(csv_file):
            return False
        else:
            print(f"Reading revisions data from {csv_file}...")
            import_df = pd.read_csv(csv_file)
            revision_data = import_df.to_dict(orient='records')
            
            if len(revision_data) > 0:
                print("Inserting data into CSV file...")
                for data in tqdm(revision_data):
                    # 确保content是字典类型
                    if isinstance(data.get('content'), str):
                        data['content'] = ast.literal_eval(data['content'])
                    self.insert_revision(**data)
                return True
            else:
                print("No new revision data to insert.")
                return False
    
    def construct_revisions_table_from_json(self, json_file: str) -> bool:
        if not os.path.exists(json_file):
            return False
        else:
            print(f"Reading revisions data from {json_file}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                revision_data = json.load(f)
            
            if len(revision_data) > 0:
                print("Inserting data into CSV file...")
                for data in tqdm(revision_data):
                    self.insert_revision(**data)
                return True
            else:
                print("No new revision data to insert.")
                return False
    
    def _clean_json_content(self, content: Union[str, dict, list, int, float, bool, None]) -> Union[str, dict, list, int, float, bool, None]:
        if isinstance(content, str):
            return ''.join(char for char in content if char.isprintable())  # 返回 str
        elif isinstance(content, dict):
            return {key: self._clean_json_content(value) for key, value in content.items()}  # 返回 dict
        elif isinstance(content, list):
            return [self._clean_json_content(item) for item in content]  # 返回 list
        else:
            return content  # 返回原始类型
        
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return s.replace('\x00', '')
        return s