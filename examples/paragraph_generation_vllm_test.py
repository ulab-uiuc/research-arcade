import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from tasks.paragraph_generation_local_vllm import Args, llm_generate, _data_extraction_vlm, _data_extraction_non_vlm

from tasks.generated_paragraph_evaluation import answer_evaluation

import csv

VLM = False
MODEL_NAME = "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"
MODEL_NAME = "Qwen/Qwen3-8B"
RESULT_PATH = "./task_result/paragraph_generation_result2.csv"


def main():
    
    paragraph_ids = ["21829759", "21831811", "21854471", "56348", "2848716", "21846899"]

    jsonl_file_path = "./jsonl/paragraph_generation2.jsonl"

    # First load the data

    if VLM:
        paragraph_datas = _data_extraction_vlm(jsonl_file_path=jsonl_file_path, model_name=MODEL_NAME)
    else:
        paragraph_datas = _data_extraction_non_vlm(jsonl_file_path=jsonl_file_path, model_name=MODEL_NAME)



    # Then pass the data to LLM for tasks
    original_contents = []
    generated_paragraphs = []
    prompts = []
    if VLM:
        image_embeddings_list = []
        for paragraph_data in paragraph_datas:
            paragraph_datas = _data_extraction_vlm(jsonl_file_path)
            # This serves for ground truth
            original_content = paragraph_data["original_content"]
            original_contents.append(original_content)

            # We also need to build figure blocks
            # We need caption and labels
            prompt = paragraph_data["prompt"]
            image_embeddings = paragraph_data["image_embeddings"]

            prompts.append(prompt)
            image_embeddings_list.append(image_embeddings)

            # Send the prompt to the vllm for evaluation
        print(f"Prompt list: {prompts}")
        print(f"Number of prompts: {len(prompts)}")
        print(f"Image embedding list: {image_embeddings}")
        print(f"Number of Image embeddings: {len(image_embeddings)}")
        
        # TODO REMOVE IT
        # sys.exit()

        generated_paragraphs = llm_generate(prompts = prompts, model_name = MODEL_NAME, is_vlm = VLM, image_embeddings = image_embeddings_list)

        # Send the answer list for evaluation
        # Before that, we first check length consistency

        if len(generated_paragraphs) != len(original_contents):
            print("Number of generated paragraphs does not match that of original contents")
            print(f"Number of generated paragraphs: {generated_paragraphs}")
            print(f"Number of original contents: {original_contents}")

            return
    else:
        image_tag_lists = []
        for paragraph_data in paragraph_datas:
            paragraph_datas = _data_extraction_vlm(jsonl_file_path)
            # This serves for ground truth
            original_content = paragraph_data["original_content"]
            original_contents.append(original_content)

            # We also need to build figure blocks
            # We need caption and labels
            prompt = paragraph_data["prompt"]
            
            # TODO ensure that the paragraph_data each line includes the 

            image_tag_list = paragraph_data["image_tag_list"]

            prompts.append(prompt)
            image_tag_lists.append(image_tag_list)

            # Send the prompt to the vllm for evaluation
        print(f"Prompt list: {prompts}")
        print(f"Number of prompts: {len(prompts)}")
        print(f"Image tag lists: {image_tag_lists}")
        print(f"Number of Image tag lists: {len(image_tag_lists)}")
        

        generated_paragraphs = llm_generate(prompts = prompts, model_name = MODEL_NAME, is_vlm = VLM, image_labels = image_tag_lists)

        # Send the answer list for evaluation
        # Before that, we first check length consistency

        if len(generated_paragraphs) != len(original_contents):
            print("Number of generated paragraphs does not match that of original contents")
            print(f"Number of generated paragraphs: {generated_paragraphs}")
            print(f"Number of original contents: {original_contents}")



        # Save all the data into a csv file

    evals = answer_evaluation(generated_paragraphs, original_contents)

    # 5) Save results CSV
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    fieldnames = [
        "paragraph_id",
        "prompt",
        "original_content",
        "generated_paragraph",
        # Dynamic eval keys added below
    ]
    # union of all eval keys
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