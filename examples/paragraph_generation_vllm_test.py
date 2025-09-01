import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.paragraph_generation_local_vllm import Args, llm_generate, _data_extraction_vlm, _data_extraction_non_vlm
from tasks.generated_paragraph_evaluation import answer_evaluation
import csv

# Configuration - ensure these match
VLM = False  # Set to True if using VLM model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Use appropriate model for VLM setting
RESULT_PATH = "./task_result/paragraph_generation_result3.csv"

def main():
    paragraph_ids = ["21829759", "21831811", "21854471", "56348", "2848716", "21846899"]
    jsonl_file_path = "./jsonl/paragraph_generation2.jsonl"

    # Load data once, outside the loop
    print(f"Loading data with VLM={VLM}, MODEL={MODEL_NAME}")
    if VLM:
        paragraph_datas = _data_extraction_vlm(jsonl_file_path=jsonl_file_path, model_name=MODEL_NAME)
    else:
        paragraph_datas = _data_extraction_non_vlm(jsonl_file_path=jsonl_file_path, model_name=MODEL_NAME)

    print(f"Loaded {len(paragraph_datas)} paragraph data entries")
    
    # Ensure we have the right number of entries
    if len(paragraph_datas) != len(paragraph_ids):
        print(f"Warning: Number of data entries ({len(paragraph_datas)}) doesn't match paragraph IDs ({len(paragraph_ids)})")

    # Process the data
    original_contents = []
    generated_paragraphs = []
    prompts = []
    
    if VLM:
        image_embeddings_list = []
        for paragraph_data in paragraph_datas:
            # DON'T reload data here - just use what we already have
            original_content = paragraph_data.get("original_content", "")
            original_contents.append(original_content)
            
            prompt = paragraph_data.get("prompt", "")
            image_embeddings = paragraph_data.get("image_embeddings", [])
            
            prompts.append(prompt)
            image_embeddings_list.append(image_embeddings)
        
        print(f"Prepared {len(prompts)} prompts with {len(image_embeddings_list)} image embeddings")
        
        # Generate paragraphs with VLM
        generated_paragraphs = llm_generate(
            prompts=prompts, 
            model_name=MODEL_NAME, 
            is_vlm=VLM, 
            image_embeddings=image_embeddings_list,
            tensor_parallel_size=2
        )
    
    else:
        image_tag_lists = []
        image_projection_lists = []
        for paragraph_data in paragraph_datas:
            # DON'T reload data here - just use what we already have
            original_content = paragraph_data.get("original_content", "")
            original_contents.append(original_content)
            
            prompt = paragraph_data.get("prompt", "")
            image_tag_list = paragraph_data.get("image_tag_list", [])
            image_projection_list = paragraph_data.get("image_projections", [])
            
            prompts.append(prompt)
            image_tag_lists.append(image_tag_list)
            image_projection_lists.append(image_projection_list)
        
        print(f"Prepared {len(prompts)} prompts")
        print(f"Image tag lists: {len(image_tag_lists)}")
        print(f"Image projection lists: {len(image_projection_lists)}")
        
        # Generate paragraphs without VLM
        generated_paragraphs = llm_generate(
            prompts=prompts, 
            model_name=MODEL_NAME, 
            is_vlm=VLM,
            image_labels=image_tag_lists, 
            image_projections=image_projection_lists
        )
    
    # Validate results
    if len(generated_paragraphs) != len(original_contents):
        print(f"Warning: Generated {len(generated_paragraphs)} paragraphs but have {len(original_contents)} original contents")
        # Adjust to match sizes if needed
        min_len = min(len(generated_paragraphs), len(original_contents), len(paragraph_ids))
        generated_paragraphs = generated_paragraphs[:min_len]
        original_contents = original_contents[:min_len]
        paragraph_ids = paragraph_ids[:min_len]
        prompts = prompts[:min_len]
    
    # Evaluate results
    print("Evaluating generated paragraphs...")
    evals = answer_evaluation(generated_paragraphs, original_contents)
    
    # Save results to CSV
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    fieldnames = [
        "paragraph_id",
        "prompt",
        "original_content",
        "generated_paragraph",
    ]
    
    # Add dynamic eval keys
    eval_keys = set()
    for e in evals:
        if isinstance(e, dict):
            eval_keys.update(e.keys())
    fieldnames.extend(sorted(eval_keys))
    
    with open(RESULT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pid, pr, orig, gen, ev in zip(
            paragraph_ids, prompts, original_contents, generated_paragraphs, evals
        ):
            row = {
                "paragraph_id": pid,
                "prompt": pr,
                "original_content": orig,
                "generated_paragraph": gen,
            }
            if isinstance(ev, dict):
                row.update(ev)
            writer.writerow(row)
    
    print(f"Saved results to {RESULT_PATH}")

if __name__ == '__main__':
    main()