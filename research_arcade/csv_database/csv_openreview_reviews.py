from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
from pathlib import Path
from typing import Optional

class CSVOpenReviewReviews:
    def __init__(self, csv_path: str = "reviews.csv"):
        self.csv_path = csv_path
        self.crawler = OpenReviewCrawler()
        
        if not os.path.exists(csv_path):
            self.create_reviews_table()
        
    def create_reviews_table(self):
        columns = ['venue', 'review_openreview_id', 'replyto_openreview_id', 
                   'writer', 'title', 'content', 'time']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(self.csv_path, index=False)
        print(f"Created empty CSV file at {self.csv_path}")
    
    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            # 将content列从JSON字符串转换为字典
            if not df.empty:
                df['content'] = df['content'].apply(
                    lambda x: json.loads(x) if pd.notna(x) and x != '' else {}
                )
            return df
    
    def _save_data(self, df: pd.DataFrame):
        # 创建副本以避免修改原始数据
        df_to_save = df.copy()
        
        # 将content列从字典转换为JSON字符串
        if 'content' in df_to_save.columns:
            df_to_save['content'] = df_to_save['content'].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )

        df_to_save.to_csv(self.csv_path, index=False)
    
    def insert_review(self, venue: str, review_openreview_id: str, 
                     replyto_openreview_id: str, title: str, writer: str, 
                     content: dict, time: str) -> Optional[tuple]:
        df = self._load_data()
        
        # 检查是否已存在（基于venue和review_openreview_id的组合键）
        exists = ((df['venue'] == venue) & 
                 (df['review_openreview_id'] == review_openreview_id)).any()
        
        if exists:
            return None
        
        # 清理content
        cleaned_content = self._clean_json_content(content)
        
        # 创建新行
        new_row = pd.DataFrame([{
            'venue': self._clean_string(venue),
            'review_openreview_id': self._clean_string(review_openreview_id),
            'replyto_openreview_id': self._clean_string(replyto_openreview_id),
            'writer': self._clean_string(writer),
            'title': self._clean_string(title),
            'content': cleaned_content,
            'time': self._clean_string(time)
        }])
        
        # 添加到DataFrame并保存
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        
        return (venue, review_openreview_id)
    
    def delete_review_by_id(self, review_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['review_openreview_id'] == review_openreview_id
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No review found with review_openreview_id {review_openreview_id}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"Review with review_openreview_id {review_openreview_id} deleted successfully.")
        return deleted_rows
    
    def delete_reviews_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要删除的行
        mask = df['venue'] == venue
        deleted_rows = df[mask].copy()
        
        if deleted_rows.empty:
            print(f"No reviews found in venue {venue}.")
            return None
        
        # 删除行
        df = df[~mask]
        self._save_data(df)
        
        print(f"All reviews in venue {venue} deleted successfully.")
        return deleted_rows
    
    def update_review(self, venue: str, review_openreview_id: str, 
                     replyto_openreview_id: str, writer: str, title: str, 
                     content: dict, time: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        # 查找要更新的行
        mask = ((df['venue'] == venue) & 
               (df['review_openreview_id'] == review_openreview_id))
        
        if not mask.any():
            print(f"No review found with review_openreview_id {review_openreview_id}.")
            return None
        
        # 保存原始记录
        original_record = df[mask].copy()
        
        # 更新记录
        cleaned_content = self._clean_json_content(content)
        df.loc[mask, 'replyto_openreview_id'] = self._clean_string(replyto_openreview_id)
        df.loc[mask, 'writer'] = self._clean_string(writer)
        df.loc[mask, 'title'] = self._clean_string(title)
        df.loc[mask, 'content'] = cleaned_content
        df.loc[mask, 'time'] = self._clean_string(time)
        
        self._save_data(df)
        
        print(f"Review with review_openreview_id {review_openreview_id} updated successfully.")
        return original_record
    
    def get_review_by_id(self, review_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['review_openreview_id'] == review_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            print(f"No review found with review_openreview_id {review_openreview_id}.")
            return None
        
        return result
    
    def get_reviews_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['venue'] == venue
        result = df[mask].copy()
        
        if result.empty:
            print(f"No reviews found in venue {venue}.")
            return None
        
        return result
    
    def get_all_reviews(self, is_all_features: bool = False) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        if is_all_features:
            return df[['venue', 'review_openreview_id', 'replyto_openreview_id', 
                      'writer', 'title', 'content', 'time']].copy()
        else:
            return df[['venue', 'review_openreview_id', 'replyto_openreview_id', 
                      'title', 'time']].copy()
    
    def check_review_exists(self, review_openreview_id: str) -> bool:
        df = self._load_data()
        return (df['review_openreview_id'] == review_openreview_id).any()
    
    def construct_reviews_table_from_api(self, venue: str):
        # 从API爬取数据
        print("Crawling review data from OpenReview API...")
        review_data = self.crawler.crawl_review_data_from_api(venue)
        
        # 插入数据
        if len(review_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(review_data):
                self.insert_review(**data)
        else:
            print("No new review data to insert.")
    
    def construct_reviews_table_from_csv(self, csv_path: str):
        print(f"Reading review data from {csv_path}...")
        import_df = pd.read_csv(csv_path)
        review_data = import_df.to_dict(orient='records')
        
        if len(review_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(review_data):
                # 确保content是字典类型
                if isinstance(data.get('content'), str):
                    data['content'] = json.loads(data['content'])
                self.insert_review(**data)
        else:
            print("No new review data to insert.")
    
    def construct_reviews_table_from_json(self, json_path: str):
        print(f"Reading review data from {json_path}...")
        with open(json_path, 'r') as f:
            review_data = json.load(f)
        
        if len(review_data) > 0:
            print("Inserting data into CSV file...")
            for data in tqdm(review_data):
                self.insert_review(**data)
        else:
            print("No new review data to insert.")
    
    def _clean_json_content(self, content):
        if isinstance(content, str):
            return ''.join(char for char in content if char.isprintable())
        elif isinstance(content, dict):
            return {key: self._clean_json_content(value) 
                   for key, value in content.items()}
        elif isinstance(content, list):
            return [self._clean_json_content(item) for item in content]
        else:
            return content
        
     def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return re.sub(r'[\x00-\x1F\x7F]', '', s)
        return s