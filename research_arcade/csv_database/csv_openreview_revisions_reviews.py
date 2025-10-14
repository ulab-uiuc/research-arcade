from ..openreview_utils.openreview_crawler import OpenReviewCrawler
from tqdm import tqdm
import pandas as pd
import json
import os
from typing import Optional

class CSVOpenReviewRevisionsReviews:
    def __init__(self, csv_path: str = "revisions_reviews.csv", 
                 revisions_csv: Optional[str] = "revisions.csv", 
                 reviews_csv: Optional[str] = "reviews.csv"):
        self.csv_path = csv_path
        self.revisions_csv = revisions_csv
        self.reviews_csv = reviews_csv
        self.openreview_crawler = OpenReviewCrawler()
        
        # 如果CSV文件不存在，创建空的DataFrame
        if not os.path.exists(csv_path):
            self.create_revisions_reviews_table()
    
    def create_revisions_reviews_table(self):
        columns = ['venue', 'revision_openreview_id', 'review_openreview_id']
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(self.csv_path, index=False)
        print(f"Created empty CSV file at {self.csv_path}")
    
    def _load_data(self) -> pd.DataFrame:
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            return df
    
    def _save_data(self, df: pd.DataFrame):"
        df.to_csv(self.csv_path, index=False)
    
    def check_revision_exists(self, revision_openreview_id: str) -> bool:
        if self.revisions_csv is None or not os.path.exists(self.revisions_csv):
            # 如果没有指定修订表，跳过验证
            return False
        
        revisions_df = pd.read_csv(self.revisions_csv)
        return (revisions_df['revision_openreview_id'] == revision_openreview_id).any()
    
    def check_review_exists(self, review_openreview_id: str) -> bool:
        if self.reviews_csv is None or not os.path.exists(self.reviews_csv):
            # 如果没有指定评审表，跳过验证
            return False
        
        reviews_df = pd.read_csv(self.reviews_csv)
        return (reviews_df['review_openreview_id'] == review_openreview_id).any()
    
    def insert_revision_reviews(self, venue: str, revision_openreview_id: str, 
                               review_openreview_id: str) -> Optional[tuple]:
        # 验证修订和评审是否存在
        if not self.check_revision_exists(revision_openreview_id):
            print(f"The revision {revision_openreview_id} does not exist in the database.")
            return None
        
        if not self.check_review_exists(review_openreview_id):
            print(f"The review {review_openreview_id} does not exist in the database.")
            return None
        
        df = self._load_data()
        
        # 检查是否已存在（基于三个字段的组合键）
        exists = ((df['venue'] == venue) & 
                 (df['revision_openreview_id'] == revision_openreview_id) &
                 (df['review_openreview_id'] == review_openreview_id)).any()
        
        if exists:
            return None
        
        # 创建新行
        new_row = pd.DataFrame([{
            'venue': self._clean_string(venue),
            'revision_openreview_id': self._clean_string(revision_openreview_id),
            'review_openreview_id': self._clean_string(review_openreview_id)
        }])
        
        # 添加到DataFrame并保存
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        
        return (venue, revision_openreview_id, review_openreview_id)
    
    def get_revision_review_by_id(self, revision_openreview_id: str, 
                                  review_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = ((df['revision_openreview_id'] == revision_openreview_id) & 
                (df['review_openreview_id'] == review_openreview_id))
        result = df[mask].copy()
        
        if result.empty:
            print(f"The revision {revision_openreview_id} and the review {review_openreview_id} are not connected in this database.")
            return None
        
        return result
    
    def get_revisions_reviews_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['venue'] == venue
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_all_revisions_reviews(self) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        if df.empty:
            return None
        
        return df.copy()
    
    def delete_revisions_reviews_by_venue(self, venue: str) -> Optional[pd.DataFrame]:
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
    
    def check_revision_review_exists(self, revision_openreview_id: str, 
                                    review_openreview_id: str) -> bool:
        df = self._load_data()
        exists = ((df['revision_openreview_id'] == revision_openreview_id) & 
                 (df['review_openreview_id'] == review_openreview_id)).any()
        return exists
    
    def get_revision_neighboring_reviews(self, revision_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['revision_openreview_id'] == revision_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def get_review_neighboring_revisions(self, review_openreview_id: str) -> Optional[pd.DataFrame]:
        df = self._load_data()
        
        mask = df['review_openreview_id'] == review_openreview_id
        result = df[mask].copy()
        
        if result.empty:
            return None
        
        return result
    
    def construct_revisions_reviews_table(self, papers_reviews_df: pd.DataFrame, 
                                         papers_revisions_df: pd.DataFrame):        
        # 获取唯一的论文ID
        unique_paper_ids = papers_revisions_df['paper_openreview_id'].unique()
        
        print(f"Constructing revisions-reviews connections for {len(unique_paper_ids)} papers...")
        
        for paper_id in tqdm(unique_paper_ids):
            # 获取该论文的所有修订（按时间排序）
            paper_revision_edges = papers_revisions_df[
                papers_revisions_df['paper_openreview_id'] == paper_id
            ].sort_values(by='time', ascending=True)
            
            # 获取该论文的所有评审（按时间排序）
            paper_review_edges = papers_reviews_df[
                papers_reviews_df['paper_openreview_id'] == paper_id
            ].sort_values(by='time', ascending=True)
            
            if paper_review_edges.empty:
                continue
            
            # 为每个修订关联在其之前的所有评审
            start_idx = 0
            for _, revision in paper_revision_edges.iterrows():
                revision_time = revision['time']
                revision_id = revision['revision_openreview_id']
                venue = revision['venue']
                
                # 遍历所有在当前修订时间之前的评审
                for idx, review in paper_review_edges.iloc[start_idx:].iterrows():
                    review_time = review['time']
                    
                    # 如果评审时间晚于修订时间，停止
                    if review_time > revision_time:
                        break
                    
                    review_id = review['review_openreview_id']
                    
                    # 插入修订-评审关联
                    self.insert_revision_reviews(venue, revision_id, review_id)
                    
                    start_idx += 1
        
        print("Revisions-reviews table construction completed.")
    
    def construct_revisions_reviews_table_from_csv(self, csv_file: str):
        print(f"Reading revisions-reviews data from {csv_file}...")
        import_df = pd.read_csv(csv_file)
        revisions_reviews_data = import_df.to_dict(orient='records')
        
        if len(revisions_reviews_data) > 0:
            print(f"Inserting revisions-reviews data from {csv_file}...")
            for data in tqdm(revisions_reviews_data):
                self.insert_revision_reviews(**data)
        else:
            print(f"No revisions-reviews data found in {csv_file}.")
    
    def construct_revisions_reviews_table_from_json(self, json_file: str):
        print(f"Reading revisions-reviews data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            revisions_reviews_data = json.load(f)
        
        if len(revisions_reviews_data) > 0:
            print(f"Inserting revisions-reviews data from {json_file}...")
            for data in tqdm(revisions_reviews_data):
                self.insert_revision_reviews(**data)
        else:
            print(f"No revisions-reviews data found in {json_file}.")
            
    def _clean_string(self, s: str) -> str:
        if isinstance(s, str):
            return re.sub(r'[\x00-\x1F\x7F]', '', s)
        return s