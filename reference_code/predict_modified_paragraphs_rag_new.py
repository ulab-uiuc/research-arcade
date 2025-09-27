import pandas as pd
from datetime import datetime
# from contriever.src.contriever import Contriever
# from transformers import AutoTokenizer
from transformers import LongformerModel, AutoTokenizer
import torch
from tqdm import tqdm
import json
import copy
import re
import os
import numpy as np

def top_k_score(y_true, y_pred, k=5):
    """
    计算Top-k Accuracy，Recall和F1-Score，其中y_true表示用户喜欢的物品的索引。
    
    :param y_true: 真实标签，是一个包含用户喜欢的物品的索引列表或数组。
    :param y_pred: 预测得分，是一个数组，表示所有物品的预测得分。
    :param k: Top-k的k值。
    :return: Top-k准确率、Recall和F1-Score。
    """
    # 获取预测得分排名前k的物品索引
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    
    # 计算推荐的Top-k物品中，哪些是用户真实喜欢的物品
    relevant_items_in_top_k = sum((idx + 1) in y_true for idx in top_k_indices)
    
    # 计算Recall
    recall = relevant_items_in_top_k / len(y_true) if len(y_true) > 0 else 0
    
    # 计算Precision (推荐物品中有多少是用户喜欢的)
    precision = relevant_items_in_top_k / k if k > 0 else 0
    
    # 计算F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


def format_floats(obj, precision=6):
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, (np.float64, np.float32)):
        return round(float(obj), precision)
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, list):
        return [format_floats(item, precision) for item in obj]
    if isinstance(obj, dict):
        return {k: format_floats(v, precision) for k, v in obj.items()}
    return obj

def mean_pooling(last_hidden_state, attention_mask):
    # last_hidden_state: [B, T, H]
    # attention_mask:    [B, T]
    mask = attention_mask.unsqueeze(-1)          # [B, T, 1]
    summed = (last_hidden_state * mask).sum(1)   # [B, H]
    counted = mask.sum(1).clamp(min=1e-9)        # [B, 1]
    return summed / counted

# dataset
with open("/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/rag/test_dataset.json", "r") as f:
    dataset = json.load(f)
    
# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Contriever.from_pretrained("facebook/contriever", ignore_mismatched_sizes=True).to(device)
# tokenizer = AutoTokenizer.from_pretrained("facebook/contriever") #Load the associated tokenizer:
model = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(device)
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

# testing
top_k = 5
result = {}
total_precision = []
total_recall = []
total_f1_score = []
for original_id, data in tqdm(zip(dataset.keys(), dataset.values()), total=len(dataset)):
    reviews = data['reviews']
    original_paragraphs = data['original_paragraphs']
    modified_paragraph_idx = data['modified_paragraph_idx']
    
    # get input embedding
    embed_input_list = reviews + original_paragraphs
    # inputs = tokenizer(embed_input_list, padding=True, truncation=True, return_tensors="pt").to(device)
    inputs = tokenizer(embed_input_list, padding=True, truncation=True, max_length=4096, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        sentence_embeddings = mean_pooling(token_embeddings, inputs["attention_mask"])
    
    reviews_embed_list = sentence_embeddings[:len(reviews)]
    paragraphs_embed_list = sentence_embeddings[len(reviews):]
    
    # get similarity score
    similarity_matrix = torch.matmul(
            torch.nn.functional.normalize(reviews_embed_list, p=2, dim=1),  # L2归一化
            torch.nn.functional.normalize(paragraphs_embed_list, p=2, dim=1).T   # 转置
        )  # shape: [n_reviews, n_paras], [1, n_paras]

    paragraphs_total_similarity = similarity_matrix.mean(dim=0).cpu().numpy()

    # get top k accuracy
    precision, recall, f1_score = top_k_score(modified_paragraph_idx, paragraphs_total_similarity, top_k)
    total_precision.append(precision)
    total_recall.append(recall)
    total_f1_score.append(f1_score)
    
    result[original_id] = {
        "modified_paragraph_idx": modified_paragraph_idx,
        "predicted_paragraph_idx": [x + 1 for x in np.argsort(paragraphs_total_similarity)[::-1][:top_k]],
        "top_k_precision": precision,
        "top_k_recall": recall,
        "top_k_f1_score": f1_score
    }
    # print(f"Top {top_k} accuracy: {top_k_accuracy_score}")
    # break

result = format_floats(result, precision=4)
average_precision = sum(total_precision)/len(total_precision)
average_recall = sum(total_recall)/len(total_recall)
average_f1_score = sum(total_f1_score)/len(total_f1_score)
result["Average Precision"] = average_precision
result["Average Recall"] = average_recall
result["Average F1-Score"] = average_f1_score
print(f"Average Precision: {average_precision}")
print(f"Average Recall: {average_recall}")
print(f"Average F1-Score: {average_f1_score}")

# save result
output_dir = "/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/rag"
# output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, "test_completion_longformer.json")
# output_file = os.path.join(output_dir, "predicted_modified_paragraphs_test_completion.json")


with open(output_file, 'w') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
    
print("File save to "+output_file)