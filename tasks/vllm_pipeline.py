"""
This program defines a pipeline for LLM to process prompt and return the answer for evaluation
"""

import os
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import torch

# 1. 定义文本格式化函数
def format_input(reviews, original_paragraph):
    PROMPT_FORMAT = f'''
        REVIEWS: {reviews}

        ORIGINAL PARAGRAPH: {original_paragraph}

        INSTRUCTIONS:
        - Please revise the paragraph according to the provided reviews.
        - Output only the revised paragraph, enclosed between [START] and [END], without any extra explanation or analysis.

        REVISED PARAGRAPH: [START]{{your revised paragraph here}}[END]
    '''
    return PROMPT_FORMAT

# 2. 定义提取修订段落的函数
def extract_revised_paragraph(response_text):
    # 使用正则表达式匹配[START]和[END]之间的内容
    match = re.search(r'\[START\](.*?)\[END\]', response_text, re.DOTALL)
    if match:
        return match.group(1).strip()  # 返回匹配的内容并去掉多余的空白
    else:
        return response_text

# 3. 加载数据集
def load_dataset(dataset_path):
    with open(dataset_path, 'r') as file:
        return json.load(file)

# 4. 初始化模型和tokenizer
def initialize_model_and_tokenizer(model_id):
    model = LLM(model_id)  # 加载模型
    # model = LLM(model_id, pipeline_parallel_size=4)
    tokenizer = AutoTokenizer.from_pretrained(model_id)  # 加载Tokenizer
    return model, tokenizer

# 5. 生成和处理prompt
def create_prompts(dataset, tokenizer):
    prompt_list = []
    result = {}

    for key, value in tqdm(zip(dataset.keys(), dataset.values()), total=len(dataset), desc="Processing dataset"):
        result[key] = []
        for modified_content in value['modified_content']:
            original_paragraph = modified_content["original_paragraph"]
            num_review = 0

            for reviews in value['review']:
                if len(reviews) >= 100:  # 只处理大于等于100个字符的评论
                    num_review += 1

                    instruction = format_input(reviews, original_paragraph)
                    message = [{"role": "user", "content": instruction}]
                    prompt = tokenizer.apply_chat_template(message, tokenize=False)  # 应用模板
                    prompt_list.append(prompt)

            # 为每个内容准备结果结构
            data = {
                "modified_paragraph": modified_content["modified_paragraph"],
                "predicted_paragraphs": [None] * num_review  # 修正了拼写错误
            }
            result[key].append(data)

    return prompt_list, result

# 6. 生成文本并将其填充到结果中
def generate_and_fill_results(model, prompt_list, result, sampling_params, dataset):
    # 使用模型生成文本
    response_list = model.generate(prompt_list, sampling_params)
    
    # 填充生成的文本到结果字典中
    idx = 0  # 用于追踪生成文本的索引
    for key, value in tqdm(zip(dataset.keys(), dataset.values()), total=len(dataset), desc="Filling results"):
        for idx_data, modified_content in enumerate(value['modified_content']):
            num_review = 0
            # 填充 predicted_paragraphs
            for reviews in value['review']:
                if len(reviews) >= 100:
                    generated_text = response_list[idx].outputs[0].text
                    result[key][idx_data]["predicted_paragraphs"][num_review] = extract_revised_paragraph(generated_text)
                    num_review += 1
                    idx += 1 

    return result

# 7. 保存结果到文件
def save_results(result, output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建目录

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"File saved to {output_file}")

# 主程序逻辑
def main():
    # 路径和文件设置
    DATASET_PATH = "/data/jingjunx/evaluation_tasks/generate_modified_paragraph/llm/generate_modified_paragraph_test_dataset_100.json"
    OUTPUT_DIR = "/data/jingjunx/evaluation_tasks/generate_modified_paragraph/llm/Qwen3-8B"
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "generate_modified_paragraph_test_completion_100.json")

    # 加载数据集
    dataset = load_dataset(DATASET_PATH)

    # 初始化模型和tokenizer
    model_id = "Qwen/Qwen3-8B"
    model, tokenizer = initialize_model_and_tokenizer(model_id)

    # 创建prompts
    prompt_list, result = create_prompts(dataset, tokenizer)
    
    print(f"Total prompts created: {len(prompt_list)}")

    # 设置采样参数 - 修正了类名
    my_sampling_params = SamplingParams(
        temperature=0,
        max_tokens=4096,
    )

    # 生成文本并填充到结果中
    result = generate_and_fill_results(model, prompt_list, result, my_sampling_params, dataset)

    # 保存结果到文件
    save_results(result, OUTPUT_FILE)
    
    # 手动释放 GPU 空间
    torch.cuda.empty_cache()
    print("GPU memory cleared")



def main():
    


# 执行主程序
if __name__ == "__main__":
    main()