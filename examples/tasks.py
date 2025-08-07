import argparse
import pandas as pd
from pathlib import Path
from typing import Dict
from transformers import pipeline
from PIL import Image
from pdf2image import convert_from_path
import openai
from openai import OpenAI


class LLM_task:
    def __init__(
        self,
        image_model: str = "Salesforce/blip-image-captioning-base",
        api_key: str = "",
        text_model: str = "gpt-3.5-turbo"
    ):
        # Image captioner
        self.captioner = pipeline("image-to-text", model=image_model)
        # OpenAI config
        # openai.api_key    = api_key
        self.text_model   = text_model
        self.client = OpenAI(api_key=api_key)


    def caption_generation_table(
        self,
        paragraph_ref_path: str,
        table_path: str,
        paragraph_path: str,
        output_csv: str,
        max_results: int = 10
    ):
        df_pf = pd.read_csv(paragraph_ref_path, dtype=str).head(max_results)
        df_t  = pd.read_csv(table_path,         dtype=str)
        df_p  = pd.read_csv(paragraph_path,     dtype=str)

        records = []
        for _, row in df_pf.iterrows():
            if row['reference_type'] != "table":
                continue

            ref_id        = row['id']
            raw_label     = row['reference_label']
            paragraph_id  = row['paragraph_id']
            paper_id      = row['paper_arxiv_id']

            # Find the table
            latex_label = f"\\label{{{raw_label}}}"
            match_t = df_t[df_t['label'] == latex_label]

            if match_t.empty:
                caption = "[No matching table found]"
            else:
                table_content = match_t.iloc[0]['table_text']

                # Find the paragraph
                match_p = df_p[
                    (df_p['paragraph_id']   == paragraph_id) &
                    (df_p['paper_arxiv_id'] == paper_id)
                ]
                if match_p.empty:
                    caption = "[No matching paragraph found]"
                else:
                    paragraph_content = match_p.iloc[0]['content']

                    # Build prompt
                    prompt = (
                        "Below is a paragraph from a paper and the text contents of its associated table.\n\n"
                        f"Paragraph:\n{paragraph_content}\n\n"
                        f"Table:\n{table_content}\n\n"
                        "Generate a concise, descriptive caption for this table as it would appear in figure captions."
                    )

                    # Call OpenAI via the new client
                    # resp = self.client.chat.completions.create(
                    #     model=self.text_model,
                    #     messages=[{"role":"user","content":prompt}],
                    #     temperature=0.0,
                    #     max_tokens=256
                    # )
                    # caption = resp.choices[0].message.content.strip()
                    print(prompt)
                    caption = "TBD"

            records.append({
                "paper_arxiv_id":        paper_id,
                "paragraph_id":          paragraph_id,
                "paragraph_ref_id":      ref_id,
                "table_label":           raw_label,
                "generated_table_caption": caption
            })

        # Write CSV with the new columns
        out_df = pd.DataFrame.from_records(
            records,
            columns=[
                "paper_arxiv_id",
                "paragraph_id",
                "paragraph_ref_id",
                "table_label",
                "generated_table_caption"
            ]
        )
        out_df.to_csv(output_csv, index=False, sep='\t')
        print(f"Saved table captions to {output_csv}")


                


    def caption_generation(self, csv_path: str, max_results: int = 10) -> Dict[str, str]:
        # 1. Read your CSV (comma-delimited with quoted captions)
        df = pd.read_csv(csv_path, dtype=str).head(max_results)

        captions: Dict[str, str] = {}
        for _, row in df.iterrows():
            row_id = row['id']
            arxiv_id = row['paper_arxiv_id']
            img_dir = Path(f"download/output/figures/{arxiv_id}")

            if not img_dir.is_dir():
                captions[row_id] = f"[Error: directory not found {img_dir}]"
                continue

            # 2. Iterate all image files in that folder
            for img_path in img_dir.iterdir():
                if not img_path.is_file():
                    continue

                key = f"{row_id}:{img_path.name}"
                try:
                    # Load and caption
                    img = self._load_image(str(img_path))
                    result = self.captioner(img)[0]
                    captions[key] = result["generated_text"].strip()
                except Exception as e:
                    captions[key] = f"[Error captioning {img_path.name}: {e}]"

        return captions
    
    def _load_image(self, file_path: str, dpi: int = 200) -> Image.Image:
        """
        Given a path to an image or PDF, return a PIL Image.
        - If `file_path` ends with .pdf, convert its first page to an image.
        - Otherwise, open with Pillow.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            # Convert only the first page of the PDF to an image
            # You need poppler installed on your system for pdf2image to work.
            pages = convert_from_path(
                str(path),
                dpi=dpi,
                first_page=1,
                last_page=1,
            )
            return pages[0]

        else:
            # Standard image formats
            return Image.open(path).convert("RGB")

def main():
    parser = argparse.ArgumentParser(description="Generate captions for all figures in given directories")
    parser.add_argument("csv_input", help="Path to the input CSV file")
    parser.add_argument("csv_output", help="Path to write the output CSV (cols: id, file_name, generated_caption)")
    parser.add_argument("--max_results", type=int, default=10, help="Max number of CSV rows to process")
    args = parser.parse_args()

    task = LLM_task()
    captions = task.caption_generation(args.csv_input, args.max_results)

    # Build DataFrame from captions dict
    records = []
    for key, caption in captions.items():
        row_id, file_name = key.split(":", 1)
        records.append({
            "id": row_id,
            "file_name": file_name,
            "generated_caption": caption
        })

    out_df = pd.DataFrame.from_records(records, columns=["id", "file_name", "generated_caption"])
    out_df.to_csv(args.csv_output, index=False)
    print(f"Saved {len(out_df)} captions to {args.csv_output}")

def main2():
    parser = argparse.ArgumentParser(
        description="Generate table captions via OpenAI chat API"
    )
    parser.add_argument("paragraph_ref_path", help="CSV of paragraph refs")
    parser.add_argument("table_path",         help="CSV of tables")
    parser.add_argument("paragraph_path",     help="CSV of paragraphs")
    parser.add_argument("output_csv",         help="Where to write output")
    parser.add_argument(
        "--max_results", type=int, default=10,
        help="Max number of paragraph-ref rows to process"
    )
    parser.add_argument(
        "--api_key", required=True,
        help="Your OpenAI API key"
    )
    parser.add_argument(
        "--text_model", default="gpt-3.5-turbo",
        help="OpenAI chat model (e.g. gpt-3.5-turbo, gpt-4)"
    )
    args = parser.parse_args()

    task = LLM_task(
        api_key    = args.api_key,
        text_model = args.text_model
    )
    task.caption_generation_table(
        paragraph_ref_path = args.paragraph_ref_path,
        table_path         = args.table_path,
        paragraph_path     = args.paragraph_path,
        output_csv         = args.output_csv,
        max_results        = args.max_results
    )

if __name__ == "__main__":
    main2()
