import sys
import os
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from tasks.paragraph_generation_local_vllm import _data_extraction_non_vlm_save

# Assume that the output format is alpaca
COLUMNS = {
    "prompt": "instruction",
    "query": "input", 
    "response": "output"
}

def remove_lines_starting_with(text, line_starts):
    """Remove lines that start with any of the given prefixes"""
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        should_remove = False
        for start in line_starts:
            if line.strip().startswith(start):
                should_remove = True
                break
        if not should_remove:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)

def main():
    parser = argparse.ArgumentParser(description="Preprocess JSONL for non-VLM models.")
    parser.add_argument(
        "--jsonl_path",
        type=str,
        required=True,
        help="Input JSONL file path."
    )
    parser.add_argument(
        "--save_path", 
        type=str,
        required=True,
        help="Output preprocessed JSONL file path."
    )
    # Option 1: Use action='store_true'
    parser.add_argument(
        "--include_figure",
        action='store_true',
        help="Whether to include figure in the prompt."
    )
    parser.add_argument(
        "--include_table",
        action='store_true',
        help="Whether to include table in the prompt."
    )
    
    args = parser.parse_args()
    
    jsonl_path = args.jsonl_path
    save_path = args.save_path
    include_figure = args.include_figure
    include_table = args.include_table
    
    written = 0
    output_data = []
    
    with open(jsonl_path, 'r') as jsonl_file:
        for line in jsonl_file:
            json_line = json.loads(line.strip())
            # print(json_line)
            # sys.exit()
            if include_figure and include_table:
                # We can just use the original prompt as input
                prompt = json_line.get("prompt", "")
                figure_tags = json_line.get("image_tag_list", [])
                # print(prompt)
                # keys = json_line.keys()
                # if tables != []:
                meta = json_line.get("meta")
                tables = meta.get("tables", [])
                # meta_keys = metas.keys()
                # print(keys)
                # print(meta_keys)
                # break

                table_contents = [table["text"] for table in tables]
                
                # Turn the tags into a string
                figure_tags_str = ", ".join(str(figure_tag) for figure_tag in figure_tags)

                table_contents_str = ", ".join(str(table_content) for table_content in table_contents)

                
                augmented_prompt = f"{prompt}\n The image descriptions corresponding the image labels are provided below in the same order: {figure_tags_str}\n The table contents corresponding to the table labels are provided below in the same order: {table_contents_str}."
                
                original_content = json_line.get("original_content", "")
                
                # Save to the output path
                # We don't really need the input. Leave it blank
                data = {
                    "instruction": augmented_prompt,
                    "input": "",
                    "output": original_content
                }
                
                output_data.append(data)
                written += 1


            elif not include_figure and include_table:
                # Here, we may need to rebuild the prompt
                old_prompt = json_line.get("prompt", "")
                # Find the line starting with "Figure block (optional)" if exists, then delete the exact line
                line_starts_to_remove = ["Figure block (optional)", "fig_labels ="]
                # Remove what is on the same line with them
                prompt = remove_lines_starting_with(old_prompt, line_starts_to_remove)

                tables = json_line.get("tables")

                table_contents = [table["text"] for table in tables]
                table_contents_str = ",".join(str(table_content) for table_content in table_contents)

                augmented_prompt = f"{prompt}\nThe table contents corresponding to the table labels are provided below in the same order: {table_contents_str}."

                original_content = json_line.get("original_content", "")

                data = {
                    "instruction": augmented_prompt,
                    "input": "",
                    "output": original_content
                }
                
                output_data.append(data)
                written += 1

            elif include_figure and not include_table:
                # Here, we may need to rebuild the prompt
                old_prompt = json_line.get("prompt", "")

                line_starts_to_remove = ["Table block (optional)", "table_labels ="]
                # Remove what is on the same line with them
                prompt = remove_lines_starting_with(old_prompt, line_starts_to_remove)

                figure_tags = json_line.get("image_tag_list", [])

                # Turn the tags into a string
                figure_tags_str = ", ".join(str(figure_tag) for figure_tag in figure_tags)

                augmented_prompt = f"{prompt}\n The image descriptions corresponding the image labels are provided below in the same order: {figure_tags_str}\n"
                
                original_content = json_line.get("original_content", "")
                
                # Save to the output path
                # We don't really need the input. Leave it blank
                data = {
                    "instruction": augmented_prompt,
                    "input": "",
                    "output": original_content
                }
                
                output_data.append(data)
                written += 1

            else:  # not include_figure and not include_table
                # Here, we may need to rebuild the prompt
                old_prompt = json_line.get("prompt", "")

                line_starts_to_remove = ["Figure block (optional)", "fig_labels =", "Table block (optional)", "table_labels ="]

                # Remove what is on the same line with both of them
                prompt = remove_lines_starting_with(old_prompt, line_starts_to_remove)

                augmented_prompt = prompt
                original_content = json_line.get("original_content", "")
                
                # Save to the output path
                # We don't really need the input. Leave it blank
                data = {
                    "instruction": augmented_prompt,
                    "input": "",
                    "output": original_content
                }
                
                output_data.append(data)
                written += 1
    
    # Write the array of JSON objects to file
    with open(save_path, 'w') as save_file:
        for data in output_data:
            save_file.write(json.dumps(data) + '\n')  # One JSON per line
    
    print(f"The number of json objects written to the jsonl file is {written}")

if __name__ == "__main__":
    main()