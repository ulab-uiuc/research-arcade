"""
Given a figure or a table, predict which paragraph it is inserted.

Three approaches:
1. LLM-based method
    1.1. oken-based graph-llm on raw document
    1.2. oken-based graph-llm on our dataset with graph structure
    1.3. embedding-based graph-llm on our dataset with graph structure
2. GNN based method: Graph-based (GNN): Paper Graph → Embedding → Aggregation
    2.1. Simple GNN
    2.2. Heterogeneous GNN
3. RAG based method: paragraph embedding → top-k aggregations
    3.1. RAG

Prior to all, we need to fetch the data needed
The real model methods are stored in another python file
"""
from tasks.utils import paragraph_ref_to_global_ref
import 

def _fetch_paper_sections(cur, paper_arxiv_id):
    statement = """
    SELECT title FROM sections WHERE paper_arxiv_id = %s
    """
    
    cur.execute(statement, (paper_arxiv_id,))

    return cur.fetchall()

def _fetch_paragraph_key_ids(cur, paper_arxiv_id, paper_section):
    statement = """"
    SELECT id FROM paragraphs WHERE paper_arxiv_id = %s AND paper_section = %s
    """

    cur.execute(statement, (paper_arxiv_id, section_name))

    return cur.fetchall()


def _fetch_context_for_paragraph(cur, arxiv_id, paragraph_id, paper_section):

    cur.execute(
        """
        SELECT paper_arxiv_id, paragraph_id, content
        FROM paragraphs
        WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s 
        """,
        (arxiv_id, paper_section, paragraph_id),
    )
    # The three inputs ensures the uniqueness of the row
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Arxiv id {arxiv_id} + Paragraph id {paragraph_id} not found.")
    return row  # (paper_arxiv_id, paper_section, paragraph_id_local)

def _fetch_global_refs_for_paragraph(cur, paper_arxiv_id: str, paragraph_id: int, paper_section: str , ref_type: str) -> List[int]:
    """
    Uses your helper mapping. If you already have paragraph_ref_to_global_ref, call it;
    otherwise select from a junction table like paragraph_references(paragraph_id, reference_type, global_id).
    """

    # cur.execute(
    #     """
    #     SELECT id
    #     FROM paragraph_references
    #     WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s AND reference_type = %s;
    #     """,
    #     (paper_arxiv_id, paper_section, paragraph_id, ref_type),
    # )

    # rows = cur.fetchall()

    # for row in rows:
    #     paragraph_id = row[0]
    global_vals = paragraph_ref_to_global_ref(arxiv_id = paper_arxiv_id, paper_section=paper_section,paragraph_id=paragraph_id, ref_type=ref_type)
    
    # print(f"global_vals: {global_vals}")
    
    return global_vals


def _fetch_figures(cur, figure_ids: List[int]) -> List[Dict]:
    
    if not figure_ids:
        return []
    cur.execute(
        """
        SELECT id, label, caption, path
        FROM figures
        WHERE id = ANY(%s)
        """,
        (figure_ids,),
    )
    rows = cur.fetchall()
    figures = []
    for rid, label, caption, path in rows:
        figures.append({
            "id": rid,
            "label": label,
            "caption": caption or "",
            "path": path or ""
        })
    return figures


def _fetch_tables(cur, table_ids: List[int]) -> List[Dict]:
    if not table_ids:
        return []
    cur.execute(
        """
        SELECT id, label, table_text
        FROM tables
        WHERE id = ANY(%s)
        """,
        (table_ids,),
    )
    rows = cur.fetchall()
    tables = []
    for rid, label, ttext in rows:
        tables.append({
            "id": rid,
            "label": label,
            "text": ttext or ""
        })
    return tables


def _fetch_paper_title_abstract(cur, paper_arxiv_id: str) -> Dict[str, str]:

    mapping = {"title": "", "abstract": ""}


    if not paper_arxiv_id:
        return mapping
    # Since arxiv_id is unique, we can assume that at most one result is fetched 
    cur.execute("""
        SELECT title, abstract FROM papers WHERE arxiv_id = %s
    """)

    return cur.fetchone




def _data_fetching(paper_arxiv_ids: Iterable[str], data_path: str, *, PASSWORD: str,
                   collect_fig_labels_fn=None) -> None:
    """
    Given paper arXiv IDs, write one JSON object per paper to a .jsonl file.
    Each object has:
      - paper_title, paper_arxiv_id, paper_abstract
      - sections: [{section_name, paragraphs: [...]}, ...]
    Paragraphs include content, figures/tables and normalized labels, plus citations with abstracts.
    """

    if collect_fig_labels_fn is None:
        collect_fig_labels_fn = _default_collect_fig_labels

    Path(data_path).parent.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="cl195",
        password=PASSWORD,
        port="5433",
    )
    conn.autocommit = True
    cur = conn.cursor()

    with open(data_path, "w", encoding="utf-8") as fp:
        for paper_arxiv_id in paper_arxiv_ids:
            # Title & abstract
            meta = _fetch_paper_title_abstract(cur, paper_arxiv_id)
            paper_title = meta["title"]
            paper_abstract = meta["abstract"]

            section_records = []
            section_names = _fetch_paper_sections(cur, paper_arxiv_id)

            for section_name in section_names:
                paragraph_key_ids, paragraph_ids_local = _fetch_paragraph_key_ids(
                    cur=cur,
                    paper_arxiv_id=paper_arxiv_id,
                    paper_section=section_name,
                )

                paragraph_records = []
                for pk_id, para_local in zip(paragraph_key_ids, paragraph_ids_local):
                    # paragraph text
                    paragraph_content = _fetch_context_for_paragraph(
                        cur=cur,
                        arxiv_id=paper_arxiv_id,
                        paragraph_id=para_local,
                        paper_section=section_name,
                    )

                    # Figures/tables
                    fig_ids = _fetch_global_refs_for_paragraph(
                        cur=cur,
                        paper_arxiv_id=paper_arxiv_id,
                        paper_section=section_name,
                        paragraph_id=para_local,
                        ref_type="figure",
                    )
                    figures = _fetch_figures(cur, fig_ids)
                    fig_labels_raw = [f["label"] for f in figures]
                    fig_labels = [collect_fig_labels_fn(lbl) for lbl in fig_labels_raw]

                    tab_ids = _fetch_global_refs_for_paragraph(
                        cur=cur,
                        paper_arxiv_id=paper_arxiv_id,
                        paper_section=section_name,
                        paragraph_id=para_local,
                        ref_type="table",
                    )
                    tables = _fetch_tables(cur, tab_ids)
                    table_labels_raw = [t["label"] for t in tables]
                    table_labels = [collect_fig_labels_fn(lbl) for lbl in table_labels_raw]

                    # Citations
                    cited_bib_keys = _fetch_cited_bib_keys(
                        cur=cur,
                        citing_arxiv_id=paper_arxiv_id,
                        paper_section=section_name,
                        paragraph_id=para_local,
                    )
                    title_id_pairs = _fetch_titles(cur, paper_arxiv_id, cited_bib_keys)
                    cited_ids = [cid for (_bk, _title, cid) in title_id_pairs]
                    cited_abstracts = _fetch_abstracts(cur, cited_ids)

                    cited_triples = []
                    for (bib_key, bib_title, _arx), abs_txt in zip(title_id_pairs, cited_abstracts):
                        cited_triples.append((bib_key, bib_title, abs_txt or ""))

                    # Assemble paragraph record
                    paragraph_records.append({
                        "paragraph_key_id": int(pk_id),
                        "paper_arxiv_id": paper_arxiv_id,
                        "paragraph_id_local": int(para_local),
                        "paper_section": section_name,
                        "paragraph_content": paragraph_content,
                        "num_char": len(paragraph_content),
                        "title": paper_title,
                        "abstract": paper_abstract,
                        "figures": figures,                # [{id,label,caption,path}, ...]
                        "tables": tables,                  # [{id,label,text}, ...]
                        "fig_labels": fig_labels,          # ["fig:...", ...]
                        "table_labels": table_labels,      # ["tab:...", ...]
                        "cited_triples": cited_triples,    # [(bib_key, title, abstract), ...]
                    })

                # (theyrre already ordered by paragraph_id_local from the query)
                section_records.append({
                    "section_name": section_name,
                    "paragraphs": paragraph_records,
                })

            paper_record = {
                "paper_title": paper_title,
                "paper_arxiv_id": paper_arxiv_id,
                "paper_abstract": paper_abstract,
                "sections": section_records,
            }

            fp.write(json.dumps(paper_record, ensure_ascii=False) + "\n")

    cur.close()
    conn.close()
