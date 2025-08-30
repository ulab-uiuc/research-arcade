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
"""


def _data_fetching(paper_arxiv_ids, data_path):

    """
    Given the paper arxiv ids, save data into jsonl file, which includes:

    1. Sections of a paper 
    2. Paragraphs in each section
    3. The figures and tables of paragraphs
    4. Citations of the paragraphs
    """

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

            # First fetch all the sections
            section_names = _fetch_paper_sections(cur, paper_arxiv_id)


            for section_name in section_names:
                # Then fetch all the paragraph ids given any section name
                paragraph_key_ids = _fetch_paragraph_key_ids(cur=cur, paper_arxiv_id=paper_arxiv_id, paper_section=section_name)

                # For each paragraph, fetch its content, figures and tables as specified

                for paragraph_key_id in paragraph_key_ids:
                    




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


