import sys
import os
import json
from typing import List, Tuple
import psycopg2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def _fetch_adjacent_paragraphs(cur, paper_arxiv_id: str, paper_section: str, pivot_local_id: int, k: int) -> Tuple[List[str], List[str]]:
    """
    Returns (prev, next) paragraphs as lists of text.
    prev: earlier paragraphs in descending-to-ascending order (we'll re-order to natural ascending).
    next: following paragraphs in natural ascending order.
    """
    prev_ids = list(range(pivot_local_id - k, pivot_local_id))
    next_ids = list(range(pivot_local_id + 1, pivot_local_id + 1 + k))

    prev_texts = []
    for pid in prev_ids:
        if pid <= 0:
            continue
        cur.execute(
            """
            SELECT content FROM paragraphs
            WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s
            """,
            (paper_arxiv_id, paper_section, pid),
        )
        row = cur.fetchone()
        if row and row[0]:
            prev_texts.append(row[0])

    next_texts = []
    for nid in next_ids:
        cur.execute(
            """
            SELECT content FROM paragraphs
            WHERE paper_arxiv_id = %s AND paper_section = %s AND paragraph_id = %s
            """,
            (paper_arxiv_id, paper_section, nid),
        )
        row = cur.fetchone()
        if row and row[0]:
            next_texts.append(row[0])

    # Ensure prev is in natural ascending order (older to newer)
    return prev_texts, next_texts

def main():


    data_path = "./data/paragraph_generation/paragraph_generation_validation_preprocessed.jsonl"
    dest_file = "./data/paragraph_generation/tasks/paragraph_generation_validatio_exp3.json"

    results = []


    conn = psycopg2.connect(
        host="localhost",
        port="5433",
        dbname="postgres",
        user="cl195"
    )
    conn.autocommit = True
    cur = conn.cursor()


    with open(data_path, 'r') as data_file:
        for line in data_file:
            line = line.strip()
            
            if line:
                json_line = json.loads(line)
                meta = json_line['meta']
                target_paragraph = json_line['original_content']
                # We can now build prompt!
                title = meta['title']
                abstract = meta['abstract']
                paper_section = meta['paper_section']
                num_char = len(target_paragraph)
                prompt = f"""You are reconstructing one missing LaTeX paragraph in a research paper.

                Paper title: {title}
                Section name of the paragraph: {paper_section}

                # Task
                Write exactly one LaTeX-formatted paragraph.

                # HARD REQUIREMENTS
                - Maintain objective, concise academic tone; ensure logical continuity with context.
                - Output exactly one paragraph.
                """.strip()

                # prev_paras = meta['prev_paras']
                # next_paras = meta['next_paras']

                # We may need to re-fetch adj paragraphs
                # adj_paras = prev_paras + next_paras

                paragraph_id_local = meta['paragraph_id_local']
                paper_arxiv_id = json_line['paper_arxiv_id']
                # paper_section

                # Set a k=5
                prev_paras, next_paras = _fetch_adjacent_paragraphs(cur=cur, paper_arxiv_id=paper_arxiv_id, paper_section=paper_section, pivot_local_id=paragraph_id_local, k=5)

                image_list = json_line['image_tag_list']

                figure_keys = json_line['fig_labels']
                figures = meta['figures']
                figure_captions = []
                for figure in figures:
                    figure_captions.append(figure['caption'])
                

                figure_key = ', '.join(figure_keys)
                figure_caption = ', '.join(figure_captions)

                n_figure = min(len(figure_keys), len(image_list))
                image_description_list = []
                for i in range(n_figure):
                    image_description_list.append(f"Figure key: {figure_keys[i]}, figure caption: {figure_captions[i]}, Image description: {image_list[i]}")

                image_descriptions = ', '.join([' '.join(map(str, item)) for item in image_list])

                
                

                tables = meta['tables']
                table_contents = []
                table_labels = []
                table_list = []
                for table in tables:
                    table_contents.append(table['text'])
                    table_labels.append(table['label'])
                    table_list.append(f"Table content: {table['text']}, table label: {table['label']}")


                table_key = ', '.join(table_labels)
                table_content = ', '.join(table_contents)
                

                bib_keys = json_line['bib_keys']
                bib_key = ', '.join(bib_keys)

                
                
                result = {
                    "previous_paragraphs": prev_paras,
                    "next_paragraphs": next_paras,
                    "image_description_list": image_description_list,
                    "table_list": table_list,
                    "bib_keys": bib_key,
                    "target_paragraph": target_paragraph,
                    "prompt": prompt,
                    "title": title,
                    "paper_section": paper_section
                }
                
                results.append(result)

    # Write results to destination file
    with open(dest_file, 'w') as dest:
        json.dump(results, dest, indent=2)


if __name__ == "__main__": 
    main()