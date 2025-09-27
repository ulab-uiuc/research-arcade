import os
import sys
import json
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.efficient_prompt_embedder import EfficientTokenEmbedder
from tasks.external_embedder import ExternalEmbedder

def parse_args():
    parser = argparse.ArgumentParser(description="Embed paragraph generation data")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input JSON file (raw data)."
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        required=True,
        help="Path to save the output JSON file with embeddings."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path
    dest_path = args.dest_path

    ete = EfficientTokenEmbedder()
    ee = ExternalEmbedder()

    with open(data_path, 'r', encoding='utf-8') as data_file, open(dest_path, 'w', encoding='utf-8') as dest_file:
        json_arr = json.load(data_file)  # ✅ load JSON array
        n_para = len(json_arr)
        iiiii = 0
        for json_object in json_arr:
            prev_paras = json_object.get("previous_paragraphs", [])
            next_paras = json_object.get("next_paragraphs", [])
            image_description_list = json_object.get("image_description_list", [])
            table_list = json_object.get("table_list", [])
            bib_key = json_object.get("bib_keys", "")
            target_paragraph = json_object.get("target_paragraph", "")
            prompt = json_object.get("prompt", "")
            title = json_object.get("title", "")
            paper_section = json_object.get("paper_section", "")

            # Build tokens (convert to list for JSON dumping)
            combined_token_ids, attention_mask = ete.construct_full_prompt_tokens(
                title=title,
                section=paper_section,
                max_length=2048
            )
            combined_token_ids = combined_token_ids.tolist()
            attention_mask = attention_mask.tolist()
            
            # External embeddings
            prev_paras_embeddings = ee.get_embeddings(prev_paras).squeeze(0).tolist() if prev_paras else []
            next_paras_embeddings = ee.get_embeddings(next_paras).squeeze(0).tolist() if next_paras else []
            image_description_list_embeddings = ee.get_embeddings(image_description_list).squeeze(0).tolist() if image_description_list else []
            table_list_embeddings = ee.get_embeddings(table_list).squeeze(0).tolist() if table_list else []
            bib_key_embedding = ee.get_embeddings([bib_key]).squeeze(0).tolist() if bib_key else []

            # Merge into new object
            result = {
                **json_object,
                "combined_token_ids": combined_token_ids,
                "attention_mask": attention_mask,
                "prev_paras_embeddings": prev_paras_embeddings,
                "next_paras_embeddings": next_paras_embeddings,
                "image_description_list_embeddings": image_description_list_embeddings,
                "table_list_embeddings": table_list_embeddings,
                "bib_key_embedding": bib_key_embedding
            }
            
            # Write one object per line (JSONL style)
            json.dump(result, dest_file, ensure_ascii=False)
            dest_file.write("\n")
            iiiii += 1
            print(f"saved {iiiii} lines out of {n_para}")
            

if __name__ == "__main__":
    main()
