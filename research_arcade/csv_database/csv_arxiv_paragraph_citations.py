import pandas as pd
import os
from pathlib import Path
from typing import Optional
import json
from rapidfuzz import fuzz, process
from collections import defaultdict
from ..arxiv_utils.utils import get_paragraph_num, arxiv_ids_hashing
from semanticscholar import SemanticScholar
from dotenv import load_dotenv
from ..arxiv_utils.citation_processing import citation_matching_csv
load_dotenv()

class CSVArxivParagraphCitation:
    def __init__(self, csv_dir: str):
        self.csv_dir = csv_dir
        csv_path = f"{csv_dir}/arxiv_paragraph_citations.csv"
        self.csv_path = csv_path
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(csv_path):
            self.create_paragraph_references_table()

    def create_paragraph_references_table(self):
        df = pd.DataFrame(columns=[
            'id', 'paragraph_id', 'paper_section',
            'citing_arxiv_id', 'bib_key', 'cited_arxiv_id', 'paragraph_global_id'
        ])
        df.to_csv(self.csv_path, index=False)
        print(f"Created paragraph_references CSV at {self.csv_path}")

    def _load_data(self): 
        dtype_map = {
            "citing_arxiv_id": str,
            "cited_arxiv_id": str,
            "bib_key": str
        }
        return pd.read_csv(self.csv_path, dtype=dtype_map) if os.path.exists(self.csv_path) else pd.DataFrame()    


    def _load_data2(self):

        df = pd.read_csv(
            self.csv_path,
            dtype={
                "cited_arxiv_id": str
            },
            keep_default_na=False
        )

        return df if os.path.exists(self.csv_path) else pd.DataFrame()

    def _save_data(self, df):
        df.to_csv(self.csv_path, index=False)

    def insert_paragraph_reference(
        self,
        paragraph_id,
        paper_section,
        citing_arxiv_id,
        bib_key,
        cited_arxiv_id=None,
        paragraph_global_id=None
    ):
        df = self._load_data()
        new_id = df['id'].max() + 1 if not df.empty else 1

        new_row = pd.DataFrame([{
            'id': new_id,
            'paragraph_id': paragraph_id,
            'paper_section': paper_section,
            'citing_arxiv_id': citing_arxiv_id,
            'bib_key': bib_key,
            'cited_arxiv_id': cited_arxiv_id,
            'paragraph_global_id': paragraph_global_id
        }])

        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)
        return new_id

    def get_all_paragraph_references(self):
        df = self._load_data()
        return None if df.empty else df.copy()

    def get_reference_neighboring_paragraphs(self, reference_id: int) -> Optional[pd.DataFrame]:
        df = self._load_data()

        if df.empty:
            return None

        result = df[df['id'] == reference_id].copy()
        return None if result.empty else result.reset_index(drop=True)

    def delete_paragraph_citation_by_paragraph_id(self, paragraph_id: int) -> int:
        df = self._load_data()

        if df.empty:
            return 0

        mask = df['paragraph_id'] == paragraph_id
        count = mask.sum()

        if count == 0:
            return 0

        df = df[~mask]
        self._save_data(df)
        return count

    def construct_citations_table_from_api(self, arxiv_ids, dest_dir):
        prefix = arxiv_ids_hashing(arxiv_ids=arxiv_ids)
        # Build the paragraphs
        paragraph_path = f"{dest_dir}/output/paragraphs/{prefix}/text_nodes.jsonl"

        with open(paragraph_path) as f:
            data = [json.loads(line) for line in f]

        section_min_paragraph = {}
        # print("len(data)")
        # print(len(data))
        # First pass
        for paragraph in data:
            paragraph_id = paragraph.get("id")
            paper_arxiv_id = paragraph.get("paper_id")
            paper_section = paragraph.get("section")

            if not paragraph_id or not paper_arxiv_id or not paper_section:
                continue

            id_number = get_paragraph_num(paragraph_id)
            key = (paper_arxiv_id, paper_section)

            if key not in section_min_paragraph:
                section_min_paragraph[key] = int(id_number)
            else:
                section_min_paragraph[key] = min(section_min_paragraph[key], int(id_number))

        # Second pass
        for paragraph in data:
            paragraph_id = paragraph.get("id")
            paper_arxiv_id = paragraph.get("paper_id")
            paper_section = paragraph.get("section")

            if not paragraph_id or not paper_arxiv_id or not paper_section:
                continue
            
            key = (paper_arxiv_id, paper_section)
            if key not in section_min_paragraph:
                continue
            
            id_number = get_paragraph_num(paragraph_id)
            id_zero_based = id_number - section_min_paragraph[key]

            cite_keys = paragraph.get("cites") or []
            for bib_key in cite_keys:
                self.insert_paragraph_reference(
                    paragraph_id=id_zero_based,
                    paper_section=paper_section,
                    citing_arxiv_id=paper_arxiv_id,
                    bib_key=bib_key
                )
        
        # update paragraph level citation with paper level citation





    def arxiv_match_bib_key_to_bib_title(self, arxiv_citations):
        df = self._load_data()

        if df.empty:
            return 0

        # Ensure bib_title column exists
        if "bib_title" not in df.columns:
            df["bib_title"] = None

        updated = 0

        for idx, row in df.iterrows():
            if pd.isna(row["bib_title"]) or row["bib_title"] == "":
                bib_title = arxiv_citations.get_bib_title_by_citing_arxiv_id_bib_key(
                    row["citing_arxiv_id"],
                    row["bib_key"]
                )

                if bib_title:
                    df.loc[idx, "bib_title"] = bib_title
                    updated += 1

        self._save_data(df)
        return updated

    def match_paragraph_citations_to_paragraph_global_id(self, arxiv_paragraphs):
        df = self._load_data()

        if df.empty:
            return 0

        updated = 0

        for idx, row in df.iterrows():
            if pd.isna(row['paragraph_global_id']):
                paragraph_global_id = arxiv_paragraphs.get_id_by_arxiv_id_section_paragraph_id(
                    row['citing_arxiv_id'],
                    row['paper_section'],
                    row['paragraph_id']
                )

                if paragraph_global_id is not None:
                    df.loc[idx, 'paragraph_global_id'] = paragraph_global_id
                    updated += 1

        self._save_data(df)
        return updated

    def _normalize_title(self, title: str) -> str:
        if not title:
            return ""
        return " ".join(title.lower().strip().split())



    def bib_title_matching(self, similarity_threshold: int = 95):

        df = self._load_data()

        if df.empty:
            print("No data to process.")
            return 0

        if "bib_title" not in df.columns:
            raise ValueError("Missing required column: bib_title")

        # Ensure output column exists
        if "cited_arxiv_id" not in df.columns:
            df["cited_arxiv_id"] = None

        citing_arxiv_ids = df["citing_arxiv_id"].dropna().unique().tolist()

        print("Number of citing arxiv ids to process")
        print(len(citing_arxiv_ids))
        sch = SemanticScholar(api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"))

        ref_map = defaultdict(list)

        for arxiv_id in citing_arxiv_ids:
            try:
                paper = sch.get_paper(
                    f"arXiv:{arxiv_id}",
                    fields=[
                        "title",
                        "references",
                        "references.title",
                        "references.externalIds"
                    ]
                )

                if not paper or not paper.references:
                    continue

                for ref in paper.references:
                    if ref.externalIds and "ArXiv" in ref.externalIds:
                        cited_id = ref.externalIds["ArXiv"]
                        cited_title = ref.title or ""

                        if cited_id and cited_title:
                            ref_map[str(arxiv_id)].append(
                                (str(cited_id), cited_title)
                            )

            except Exception as e:
                print(f"[Warning] Failed Semantic Scholar query for {arxiv_id}: {e}")
                continue
        # print("ref_map")
        # print(ref_map)

        total = 0
        matched = 0
        no_match = 0

        for idx, row in df.iterrows():
            bib_title = row.get("bib_title")
            citing_id = str(row.get("citing_arxiv_id"))

            if not bib_title or citing_id not in ref_map:
                no_match += 1
                continue

            candidates = ref_map[citing_id]
            candidate_titles = [self._normalize_title(t[1]) for t in candidates]

            norm_query = self._normalize_title(bib_title)

            best = process.extractOne(
                norm_query,
                candidate_titles,
                scorer=fuzz.ratio,
                score_cutoff=similarity_threshold
            )

            total += 1

            if best:
                _, score, best_idx = best
                cited_arxiv_id = candidates[best_idx][0]

                df.loc[idx, "cited_arxiv_id"] = cited_arxiv_id
                matched += 1
            else:
                no_match += 1


        self._save_data(df)


        print("\n" + "=" * 50)
        print("Semantic Scholar Citation Matching Stats")
        print(f"Total processed: {total}")
        print(f"Matched: {matched}")
        print(f"No match: {no_match}")
        print("=" * 50)

        return matched


    def save_paragraph_with_reference(self, arxiv_paragraphs, output_csv_path: str):
        df = self._load_data2()
        
        if df.empty:
            print("No data to process.")
            return 0

        output_rows = []

        for _, row in df.iterrows():
            cited_arxiv_id = row['cited_arxiv_id']
            paragraph_global_id = row['paragraph_global_id']
            
            if pd.isna(cited_arxiv_id) or pd.isna(paragraph_global_id):
                continue
            
            para_obj = arxiv_paragraphs.get_paragraph_by_id(paragraph_global_id)

            if para_obj is None or para_obj.empty:
                continue

            paragraph_text = para_obj["content"].iloc[0]


            if not paragraph_text or not isinstance(paragraph_text, str):
                continue

            output_rows.append({
                "citing_arxiv_id": row['citing_arxiv_id'],
                "paragraph_global_id": paragraph_global_id,
                "paragraph_text": paragraph_text,
                "cited_arxiv_id": cited_arxiv_id,
                "bib_key": row['bib_key'],
                "bib_title": row['bib_title'],
                "paper_section": row['paper_section'],
                "paragraph_id": row['paragraph_id']
            })

        if not output_rows:
            print("No valid rows to save.")
            return 0

        out_df = pd.DataFrame(output_rows)
        out_df.to_csv(output_csv_path, index=False)

        print(f"Saved {len(out_df)} paragraph-reference rows to {output_csv_path}")
        return len(out_df)
    

    def update_paragraph_global_id(self, arxiv_ids):
        df1 = self._load_data()

        if df1.empty:
            return 0

        paragraph_csv = os.path.join(self.csv_dir, "arxiv_paragraphs.csv")

        if not os.path.exists(paragraph_csv):
            print(f"Paragraph CSV not found: {paragraph_csv}")
            return 0

        df2 = pd.read_csv(paragraph_csv)

        # Build lookup: (paper_arxiv_id, paper_section, paragraph_in_paper_id) -> id
        paragraph_lookup = {}
        for _, row in df2.iterrows():
            key = (str(row['paper_arxiv_id']), str(row['paper_section']), str(row['paragraph_in_paper_id']))
            paragraph_id = row.get('id')
            if pd.notna(paragraph_id):
                paragraph_lookup[key] = paragraph_id

        updated = 0

        for idx, row in df1.iterrows():
            arxiv_id = str(row['citing_arxiv_id'])
            
            if arxiv_id in arxiv_ids and (pd.isna(row['paragraph_global_id']) or row['paragraph_global_id'] == ''):
                # Match df1 columns to df2 columns
                lookup_key = (arxiv_id, str(row['paper_section']), str(row['paragraph_id']))
                
                if lookup_key in paragraph_lookup:
                    df1.loc[idx, 'paragraph_global_id'] = paragraph_lookup[lookup_key]
                    updated += 1

        self._save_data(df1)
        print(f"Updated {updated} paragraph_global_id entries")
        return updated

    def update_cited_paper_arxiv_ids(self, arxiv_ids):
        df1 = self._load_data()
        
        if df1.empty:
            return

        citation_csv = os.path.join(self.csv_dir, "arxiv_citations.csv")
        
        if not os.path.exists(citation_csv):
            print(f"Citation CSV not found: {citation_csv}")
            return

        df2 = pd.read_csv(citation_csv, dtype={"cited_arxiv_id": str}, keep_default_na=False)

        # Create a lookup dictionary from df2: (citing_arxiv_id, bib_key) -> cited_arxiv_id
        citation_lookup = {}
        for _, row in df2.iterrows():
            key = (str(row['citing_arxiv_id']), str(row['bib_key']))
            cited_id = row.get('cited_arxiv_id')
            if cited_id and cited_id != '':
                citation_lookup[key] = cited_id

        updated = 0

        for idx, row in df1.iterrows():
            citing_id = str(row['citing_arxiv_id'])
            
            # Check if this row matches our arxiv_ids filter and has empty cited_arxiv_id
            if citing_id in arxiv_ids and (pd.isna(row['cited_arxiv_id']) or row['cited_arxiv_id'] == ''):
                lookup_key = (citing_id, str(row['bib_key']))
                
                if lookup_key in citation_lookup:
                    df1.loc[idx, 'cited_arxiv_id'] = citation_lookup[lookup_key]
                    updated += 1

        self._save_data(df1)
