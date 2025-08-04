"""
A csv version of the paper graph db
"""

import json
from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict, Any

SEQUENCE_FILENAME = "sequences.json"


def _ensure_list_field(df: pd.DataFrame, col: str):
    if col in df.columns:
        # Normalize stored JSON-string lists into Python lists
        def _parse(v):
            if pd.isna(v):
                return []
            if isinstance(v, list):
                return v
            try:
                return json.loads(v)
            except Exception:
                return []
        df[col] = df[col].apply(_parse)
    else:
        df[col] = [[] for _ in range(len(df))]  # ensure column exists


class CSVGraph:
    """
    Lightweight CSV-backed graph storage for papers/authors/citations/etc.
    Mimics subset of the original Database class behavior without Postgres.
    """

    TABLES = [
        "papers", "sections", "paragraphs", "authors", "categories", "institutions",
        "figures", "tables", "paper_authors", "paper_category", "citations",
        "paper_figures", "paper_tables", "author_affiliation",
        "paragraph_references", "paragraph_citations"
    ]

    def __init__(self, csv_dir: str):
        self.csv_dir = Path(csv_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._load_sequences()
        self._load_all()

    # ---- sequence bookkeeping ----
    def _load_sequences(self):
        seq_path = self.csv_dir / SEQUENCE_FILENAME
        if seq_path.exists():
            with open(seq_path, "r") as f:
                self._sequences = json.load(f)
        else:
            self._sequences = {}

    def _save_sequences(self):
        seq_path = self.csv_dir / SEQUENCE_FILENAME
        with open(seq_path, "w") as f:
            json.dump(self._sequences, f)

    def _next_id(self, table_name: str) -> int:
        if table_name not in self._sequences:
            self._sequences[table_name] = 0
        self._sequences[table_name] += 1
        self._save_sequences()
        return self._sequences[table_name]

    # ---- loading / saving ----
    def _load(self, name: str) -> pd.DataFrame:
        path = self.csv_dir / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            if name == "citations":
                _ensure_list_field(df, "citing_sections")
                _ensure_list_field(df, "citing_paragraphs")
            # For boolean-like or list-like fields, user code can normalize as needed.
            return df.fillna(pd.NA)
        else:
            return pd.DataFrame()

    def _load_all(self):
        self.papers = self._load("papers")
        self.sections = self._load("sections")
        self.paragraphs = self._load("paragraphs")
        self.authors = self._load("authors")
        self.categories = self._load("categories")
        self.institutions = self._load("institutions")
        self.figures = self._load("figures")
        self.tables = self._load("tables")
        self.paper_authors = self._load("paper_authors")
        self.paper_category = self._load("paper_category")
        self.citations = self._load("citations")
        self.paper_figures = self._load("paper_figures")
        self.paper_tables = self._load("paper_tables")
        self.author_affiliation = self._load("author_affiliation")
        self.paragraph_references = self._load("paragraph_references")
        self.paragraph_citations = self._load("paragraph_citations")

        if not self.citations.empty:
            _ensure_list_field(self.citations, "citing_sections")
            _ensure_list_field(self.citations, "citing_paragraphs")

    def _save(self, name: str, df: pd.DataFrame):
        path = self.csv_dir / f"{name}.csv"
        if name == "citations" and not df.empty:
            df = df.copy()
            for col in ("citing_sections", "citing_paragraphs"):
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: json.dumps(v) if not pd.isna(v) else "[]")
        df.to_csv(path, index=False)

    def save_all(self):
        for name in self.TABLES:
            df = getattr(self, name, None)
            if isinstance(df, pd.DataFrame):
                self._save(name, df)

    # ---- insert / upsert equivalents ----

    def insert_paper(self, arxiv_id: str, base_arxiv_id: Optional[str], version: Optional[str],
                    title: str, abstract: Optional[str] = None,
                    submit_date: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        if self.papers is None or self.papers.empty:
            self.papers = pd.DataFrame(columns=[
                "id", "arxiv_id", "base_arxiv_id", "version", "title",
                "abstract", "submit_date", "metadata"
            ])
        conflict = self.papers[self.papers["arxiv_id"] == arxiv_id]
        if not conflict.empty:
            return conflict.iloc[0]["id"]
        new_id = self._next_id("papers")
        row = {
            "id": new_id,
            "arxiv_id": arxiv_id,
            "base_arxiv_id": base_arxiv_id or "",
            "version": version or "",
            "title": title,
            "abstract": abstract or "",
            "submit_date": submit_date or "",
            "metadata": json.dumps(metadata) if metadata is not None else "{}"
        }
        self.papers = pd.concat([self.papers, pd.DataFrame([row])], ignore_index=True)
        self._save("papers", self.papers)
        return new_id

    def insert_section(self, content: str, title: str, is_appendix: bool, paper_arxiv_id: str):
        if self.sections is None or self.sections.empty:
            self.sections = pd.DataFrame(columns=[
                "id", "content", "title", "appendix", "paper_arxiv_id"
            ])
        conflict = self.sections[
            (self.sections["title"] == title) &
            (self.sections["paper_arxiv_id"] == paper_arxiv_id)
        ]
        if not conflict.empty:
            return conflict.iloc[0]["id"]
        new_id = self._next_id("sections")
        row = {
            "id": new_id,
            "content": content,
            "title": title,
            "appendix": bool(is_appendix),
            "paper_arxiv_id": paper_arxiv_id
        }
        self.sections = pd.concat([self.sections, pd.DataFrame([row])], ignore_index=True)
        self._save("sections", self.sections)
        return new_id

    def insert_paragraph(self, paragraph_id: str, content: str, paper_arxiv_id: str, paper_section: str):
        if self.paragraphs is None or self.paragraphs.empty:
            self.paragraphs = pd.DataFrame(columns=[
                "id", "paragraph_id", "content", "paper_arxiv_id", "paper_section"
            ])
        conflict = self.paragraphs[
            (self.paragraphs["paragraph_id"] == paragraph_id) &
            (self.paragraphs["paper_arxiv_id"] == paper_arxiv_id) &
            (self.paragraphs["paper_section"] == paper_section)
        ]
        if not conflict.empty:
            return conflict.iloc[0]["id"]
        new_id = self._next_id("paragraphs")
        row = {
            "id": new_id,
            "paragraph_id": paragraph_id,
            "content": content,
            "paper_arxiv_id": paper_arxiv_id,
            "paper_section": paper_section
        }
        self.paragraphs = pd.concat([self.paragraphs, pd.DataFrame([row])], ignore_index=True)
        self._save("paragraphs", self.paragraphs)
        return new_id

    def insert_author(self, semantic_scholar_id: str, name: str, homepage: Optional[str] = None):
        if self.authors is None or self.authors.empty:
            self.authors = pd.DataFrame(columns=["id", "semantic_scholar_id", "name", "homepage"])
        conflict = self.authors[self.authors["semantic_scholar_id"] == semantic_scholar_id]
        if not conflict.empty:
            return conflict.iloc[0]["id"]
        new_id = self._next_id("authors")
        row = {
            "id": new_id,
            "semantic_scholar_id": semantic_scholar_id,
            "name": name,
            "homepage": homepage or ""
        }
        self.authors = pd.concat([self.authors, pd.DataFrame([row])], ignore_index=True)
        self._save("authors", self.authors)
        return new_id

    def insert_category(self, name: str, description: Optional[str] = None):
        if self.categories is None or self.categories.empty:
            self.categories = pd.DataFrame(columns=["id", "name", "description"])
        conflict = self.categories[self.categories["name"] == name]
        if not conflict.empty:
            return conflict.iloc[0]["id"]
        new_id = self._next_id("categories")
        row = {
            "id": new_id,
            "name": name,
            "description": description or ""
        }
        self.categories = pd.concat([self.categories, pd.DataFrame([row])], ignore_index=True)
        self._save("categories", self.categories)
        return new_id

    def insert_institution(self, name: str, location: Optional[str] = None):
        if self.institutions is None or self.institutions.empty:
            self.institutions = pd.DataFrame(columns=["id", "name", "location"])
        # Deduplication: allow same name but you could extend with more sophisticated logic
        existing = self.institutions[self.institutions["name"] == name]
        if not existing.empty:
            return existing.iloc[0]["id"]
        new_id = self._next_id("institutions")
        row = {
            "id": new_id,
            "name": name,
            "location": location or ""
        }
        self.institutions = pd.concat([self.institutions, pd.DataFrame([row])], ignore_index=True)
        self._save("institutions", self.institutions)
        return new_id

    def insert_figure(self, paper_arxiv_id: str, path: Optional[str] = None,
                      caption: Optional[str] = None, label: Optional[str] = None, name: Optional[str] = None):
        if self.figures is None or self.figures.empty:
            self.figures = pd.DataFrame(columns=["id", "paper_arxiv_id", "path", "caption", "label", "name"])
        # No uniqueness enforced; you could dedupe on (paper_arxiv_id,label) if desired
        new_id = self._next_id("figures")
        row = {
            "id": new_id,
            "paper_arxiv_id": paper_arxiv_id,
            "path": path or "",
            "caption": caption or "",
            "label": label or "",
            "name": name or ""
        }
        self.figures = pd.concat([self.figures, pd.DataFrame([row])], ignore_index=True)
        self._save("figures", self.figures)
        return new_id

    def insert_table(self, paper_arxiv_id: str, path: Optional[str] = None,
                     caption: Optional[str] = None, label: Optional[str] = None, table_text: Optional[str] = None):
        if self.tables is None or self.tables.empty:
            self.tables = pd.DataFrame(columns=["id", "paper_arxiv_id", "path", "caption", "label", "table_text"])
        new_id = self._next_id("tables")
        row = {
            "id": new_id,
            "paper_arxiv_id": paper_arxiv_id,
            "path": path or "",
            "caption": caption or "",
            "label": label or "",
            "table_text": table_text or ""
        }
        self.tables = pd.concat([self.tables, pd.DataFrame([row])], ignore_index=True)
        self._save("tables", self.tables)
        return new_id

    def insert_paper_author(self, paper_arxiv_id: str, author_id: str, author_sequence: int):
        if self.paper_authors is None or self.paper_authors.empty:
            self.paper_authors = pd.DataFrame(columns=["paper_arxiv_id", "author_id", "author_sequence"])
        conflict = self.paper_authors[
            (self.paper_authors["paper_arxiv_id"] == paper_arxiv_id) &
            (self.paper_authors["author_id"] == author_id)
        ]
        if not conflict.empty:
            return False
        self.paper_authors = pd.concat([self.paper_authors, pd.DataFrame([{
            "paper_arxiv_id": paper_arxiv_id,
            "author_id": author_id,
            "author_sequence": author_sequence
        }])], ignore_index=True)
        self._save("paper_authors", self.paper_authors)
        return True

    def insert_paper_category(self, paper_arxiv_id: str, category_id: str):
        if self.paper_category is None or self.paper_category.empty:
            self.paper_category = pd.DataFrame(columns=["paper_arxiv_id", "category_id"])
        conflict = self.paper_category[
            (self.paper_category["paper_arxiv_id"] == paper_arxiv_id) &
            (self.paper_category["category_id"] == category_id)
        ]
        if not conflict.empty:
            return False
        self.paper_category = pd.concat([self.paper_category, pd.DataFrame([{
            "paper_arxiv_id": paper_arxiv_id,
            "category_id": category_id
        }])], ignore_index=True)
        self._save("paper_category", self.paper_category)
        return True

    def insert_citation(self, citing_arxiv_id: str, cited_arxiv_id: Optional[str],
                        bib_title: Optional[str], bib_key: Optional[str],
                        author_cited_paper: Optional[str], citing_sections: Optional[List[str]] = None):
        if self.citations is None or self.citations.empty:
            self.citations = pd.DataFrame(columns=[
                "id", "citing_arxiv_id", "cited_arxiv_id", "bib_title", "bib_key",
                "author_cited_paper", "citing_sections", "citing_paragraphs"
            ])
        if citing_arxiv_id == cited_arxiv_id:
            return False
        conflict = self.citations[
            (self.citations["citing_arxiv_id"] == citing_arxiv_id) &
            (self.citations["cited_arxiv_id"] == (cited_arxiv_id or ""))
        ]
        if not conflict.empty:
            return False
        new_id = self._next_id("citations")
        row = {
            "id": new_id,
            "citing_arxiv_id": citing_arxiv_id,
            "cited_arxiv_id": cited_arxiv_id or "",
            "bib_title": bib_title or "",
            "bib_key": bib_key or "",
            "author_cited_paper": author_cited_paper or "",
            "citing_sections": citing_sections or [],
            "citing_paragraphs": []
        }
        self.citations = pd.concat([self.citations, pd.DataFrame([row])], ignore_index=True)
        _ensure_list_field(self.citations, "citing_sections")
        _ensure_list_field(self.citations, "citing_paragraphs")
        self._save("citations", self.citations)
        return True

    def append_citing_paragraph(self, citing_arxiv_id: str, bib_key: str, paragraph_id: str):
        if self.citations is None or self.citations.empty:
            return
        mask = (
            (self.citations["citing_arxiv_id"] == citing_arxiv_id) &
            (self.citations["bib_key"] == bib_key)
        )
        idx = self.citations[mask].index
        if len(idx) == 0:
            return
        i = idx[0]
        current = self.citations.at[i, "citing_paragraphs"]
        if isinstance(current, str):
            try:
                current = json.loads(current)
            except:
                current = []
        if paragraph_id not in current:
            current.append(paragraph_id)
            self.citations.at[i, "citing_paragraphs"] = current
            self._save("citations", self.citations)

    def insert_paper_figure(self, paper_arxiv_id: str, figure_id: str):
        if self.paper_figures is None or self.paper_figures.empty:
            self.paper_figures = pd.DataFrame(columns=["paper_arxiv_id", "figure_id"])
        conflict = self.paper_figures[
            (self.paper_figures["paper_arxiv_id"] == paper_arxiv_id) &
            (self.paper_figures["figure_id"] == figure_id)
        ]
        if not conflict.empty:
            return False
        self.paper_figures = pd.concat([self.paper_figures, pd.DataFrame([{
            "paper_arxiv_id": paper_arxiv_id,
            "figure_id": figure_id
        }])], ignore_index=True)
        self._save("paper_figures", self.paper_figures)
        return True

    def insert_paper_table(self, paper_arxiv_id: str, table_id: str):
        if self.paper_tables is None or self.paper_tables.empty:
            self.paper_tables = pd.DataFrame(columns=["paper_arxiv_id", "table_id"])
        conflict = self.paper_tables[
            (self.paper_tables["paper_arxiv_id"] == paper_arxiv_id) &
            (self.paper_tables["table_id"] == table_id)
        ]
        if not conflict.empty:
            return False
        self.paper_tables = pd.concat([self.paper_tables, pd.DataFrame([{
            "paper_arxiv_id": paper_arxiv_id,
            "table_id": table_id
        }])], ignore_index=True)
        self._save("paper_tables", self.paper_tables)
        return True

    def insert_paragraph_citations(self, paragraph_id: str, paper_section: str, citing_arxiv_id: str, bib_key: str):
        if self.paragraph_citations is None or self.paragraph_citations.empty:
            self.paragraph_citations = pd.DataFrame(columns=["id", "paragraph_id", "paper_section", "citing_arxiv_id", "bib_key"])
        conflict = self.paragraph_citations[
            (self.paragraph_citations["paragraph_id"] == paragraph_id) &
            (self.paragraph_citations["paper_section"] == paper_section) &
            (self.paragraph_citations["citing_arxiv_id"] == citing_arxiv_id) &
            (self.paragraph_citations["bib_key"] == bib_key)
        ]
        if not conflict.empty:
            return conflict.iloc[0]["id"]
        new_id = self._next_id("paragraph_citations")
        row = {
            "id": new_id,
            "paragraph_id": paragraph_id,
            "paper_section": paper_section,
            "citing_arxiv_id": citing_arxiv_id,
            "bib_key": bib_key
        }
        self.paragraph_citations = pd.concat([self.paragraph_citations, pd.DataFrame([row])], ignore_index=True)
        self._save("paragraph_citations", self.paragraph_citations)
        return new_id

    def insert_paragraph_reference(self, paragraph_id: str, paper_section: str, paper_arxiv_id: str, reference_label: str, reference_type: Optional[str] = None):
        if self.paragraph_references is None or self.paragraph_references.empty:
            self.paragraph_references = pd.DataFrame(columns=["id", "paragraph_id", "paper_section", "paper_arxiv_id", "reference_label", "reference_type"])
        conflict = self.paragraph_references[
            (self.paragraph_references["paragraph_id"] == paragraph_id) &
            (self.paragraph_references["paper_section"] == paper_section) &
            (self.paragraph_references["paper_arxiv_id"] == paper_arxiv_id) &
            (self.paragraph_references["reference_label"] == reference_label)
        ]
        if not conflict.empty:
            return conflict.iloc[0]["id"]
        new_id = self._next_id("paragraph_references")
        row = {
            "id": new_id,
            "paragraph_id": paragraph_id,
            "paper_section": paper_section,
            "paper_arxiv_id": paper_arxiv_id,
            "reference_label": reference_label,
            "reference_type": reference_type or ""
        }
        self.paragraph_references = pd.concat([self.paragraph_references, pd.DataFrame([row])], ignore_index=True)
        self._save("paragraph_references", self.paragraph_references)
        return new_id

    def insert_author_affiliation(self, author_id: str, institution_id: str):
        if self.author_affiliation is None or self.author_affiliation.empty:
            self.author_affiliation = pd.DataFrame(columns=["author_id", "institution_id"])
        conflict = self.author_affiliation[
            (self.author_affiliation["author_id"] == author_id) &
            (self.author_affiliation["institution_id"] == institution_id)
        ]
        if not conflict.empty:
            return False
        self.author_affiliation = pd.concat([self.author_affiliation, pd.DataFrame([{
            "author_id": author_id,
            "institution_id": institution_id
        }])], ignore_index=True)
        self._save("author_affiliation", self.author_affiliation)
        return True

    # ---- lookups / denorm ----

    def get_paper_with_authors_and_citations(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        paper_df = self.papers[self.papers["arxiv_id"] == arxiv_id]
        if paper_df.empty:
            return None
        paper = paper_df.iloc[0].to_dict()

        pa = self.paper_authors[self.paper_authors["paper_arxiv_id"] == arxiv_id]
        pa = pa.sort_values("author_sequence") if not pa.empty else pa
        author_ids = pa["author_id"].tolist() if not pa.empty else []
        authors = self.authors[self.authors["semantic_scholar_id"].isin(author_ids)] if not self.authors.empty else pd.DataFrame()
        authors_list = authors.to_dict(orient="records") if not authors.empty else []

        citations = self.citations[self.citations["citing_arxiv_id"] == arxiv_id] if not self.citations.empty else pd.DataFrame()
        if not citations.empty:
            _ensure_list_field(citations, "citing_sections")
            _ensure_list_field(citations, "citing_paragraphs")
        citations_list = citations.to_dict(orient="records") if not citations.empty else []

        return {
            "paper": paper,
            "authors": authors_list,
            "citations": citations_list
        }

    def build_denorm_paper_csv(self, output_path: str):
        rows = []
        for _, paper in (self.papers or pd.DataFrame()).iterrows():
            arxiv_id = paper["arxiv_id"]
            info = self.get_paper_with_authors_and_citations(arxiv_id)
            if not info:
                continue
            author_names = [a.get("name", "") for a in info["authors"]]
            cited = [
                c.get("cited_arxiv_id", "")
                for c in info["citations"]
                if c.get("cited_arxiv_id")
            ]
            row = {
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": ";".join(author_names),
                "citations": ";".join(filter(None, cited))
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    # ---- existence checks ----

    def check_exist_figure(self, bib_key: str) -> bool:
        entry = f"\\label{{{bib_key}}}"
        if self.figures.empty:
            return False
        return not self.figures[self.figures["label"] == entry].empty

    def check_exist_table(self, bib_key: str) -> bool:
        entry = f"\\label{{{bib_key}}}"
        if self.tables.empty:
            return False
        return not self.tables[self.tables["label"] == entry].empty

    def check_exist(self, paper_arxiv_id: str) -> bool:
        if self.papers.empty:
            return False
        return not self.papers[self.papers["arxiv_id"] == paper_arxiv_id].empty

    def paper_authors_exist(self, paper_arxiv_id: str) -> bool:
        if self.paper_authors.empty:
            return False
        return not self.paper_authors[self.paper_authors["paper_arxiv_id"] == paper_arxiv_id].empty
