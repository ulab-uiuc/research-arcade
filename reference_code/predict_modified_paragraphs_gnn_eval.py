import torch
import torch.nn as nn
from torch_geometric.nn import HANConv
from torch_geometric.data import HeteroData
import torch.nn.functional as F
import json
import numpy as np
import os

# 加载图数据和标签 - 简化版本
def load_data(TEST_REVIEW_GRAPH_PATH, TEST_MODIFIED_PARAGRAPHS_PATH, TEST_ORIGINAL_PARAGRAPHS_PATH):
    # 加载你的数据
    review_graph_test_dataset = torch.load(TEST_REVIEW_GRAPH_PATH, weights_only=False)
    original_paragraphs_test_dataset = torch.load(TEST_ORIGINAL_PARAGRAPHS_PATH, weights_only=False)
    with open(TEST_MODIFIED_PARAGRAPHS_PATH, 'r') as f:
        modified_paragraphs_test_labels = json.load(f)
        
    assert len(review_graph_test_dataset) == len(original_paragraphs_test_dataset)
    assert len(review_graph_test_dataset) == len(modified_paragraphs_test_labels)
    
    return review_graph_test_dataset, modified_paragraphs_test_labels, original_paragraphs_test_dataset

# 定义 HANConv 网络 - 输出矩阵而不是单个向量
class GraphEmbeddingModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata):
        super(GraphEmbeddingModel, self).__init__()
        # 定义 HANConv 层
        self.conv = HANConv(in_channels, hidden_channels, heads=8, dropout=0.6, metadata=metadata)
        # FC层用于将每个节点特征映射到输出维度
        self.fc = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x_dict, edge_index_dict):
        # 计算 GNN 的输出
        out = self.conv(x_dict, edge_index_dict)
        
        # 选择所有节点的特征（reviewer + author）
        # 拼接所有节点类型的输出
        all_node_features = torch.cat([out['reviewer'], out['author']], dim=0)  # [total_nodes, hidden_channels]
        
        # 通过FC层映射到输出维度
        output_matrix = self.fc(all_node_features)  # [total_nodes, out_channels]
        
        return output_matrix

# evaluation metric
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
    if isinstance(obj, dict):
        return {k: format_floats(v, precision) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_floats(item, precision) for item in obj]
    elif isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, np.int64):  # 处理 numpy.int64
        return int(obj)  # 转换为 Python 原生 int
    else:
        return obj

# Load data
TEST_REVIEW_GRAPH_PATH="/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/gnn/test_review_graph_dataset_longformer.pt"
TEST_MODIFIED_PARAGRAPHS_PATH="/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/gnn/test_modified_paragraphs_labels_longformer.json"
TEST_ORIGINAL_PARAGRAPHS_PATH="/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/gnn/test_original_paragraphs_dataset_longformer.pt"

review_graph_test_dataset, \
modified_paragraphs_test_labels, \
original_paragraphs_test_dataset \
= load_data(TEST_REVIEW_GRAPH_PATH, TEST_MODIFIED_PARAGRAPHS_PATH, TEST_ORIGINAL_PARAGRAPHS_PATH)

# Load model
MODEL_PATH="/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/gnn/HANConv_infoNCEloss_longformer.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphEmbeddingModel(in_channels=768, hidden_channels=128, out_channels=768, metadata=review_graph_test_dataset[0].metadata())
# model = GraphEmbeddingModel(in_channels=768, hidden_channels=256, out_channels=768, metadata=review_graph_test_dataset[0].metadata(), num_layers=3) # 2
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)

# Evaluation
model.eval()
total_precision = []
total_recall = []
total_f1 = []
top_k = 5
result = []
for review_graph, modified_paragraphs_label, original_paragraphs in zip(review_graph_test_dataset, modified_paragraphs_test_labels, original_paragraphs_test_dataset):
    review_graph = review_graph.to(device)
    original_paragraphs = original_paragraphs.to(device)
    modified_paragraphs_label = modified_paragraphs_label
    
    if len(original_paragraphs) < max(modified_paragraphs_label):
        continue

    # 前向传播
    reviews_embedding = model(review_graph.x_dict, review_graph.edge_index_dict)
    
    # 计算余弦相似度矩阵
    similarity_matrix = torch.matmul(
        torch.nn.functional.normalize(reviews_embedding, p=2, dim=1),  # L2归一化
        torch.nn.functional.normalize(original_paragraphs, p=2, dim=1).T   # 转置
    )
    # similarity_matrix = torch.matmul(
    #     reviews_embedding,  # L2归一化
    #     original_paragraphs.T   # 转置
    # )

    # 计算沿着第0轴的均值
    paragraphs_total_similarity = torch.mean(similarity_matrix, dim=0)

    # 如果你最终需要 numpy 数组，可以在这里转换
    paragraphs_total_similarity = paragraphs_total_similarity.cpu().detach().numpy()
    
    precision, recall, f1_score = top_k_score(modified_paragraphs_label, paragraphs_total_similarity)
    
    total_precision.append(precision)
    total_recall.append(recall)
    total_f1.append(f1_score)

    result.append({
        "modified_paragraph_idx": modified_paragraphs_label,
        "predicted_paragraph_idx": [x + 1 for x in np.argsort(paragraphs_total_similarity)[::-1][:top_k]],
        "top_k_precision": precision,
        "top_k_recall": recall,
        "top_k_f1_score": f1_score
    })
    # break

result = format_floats(result, precision=4)
average_precision = sum(total_precision) / len(total_precision)
average_recall = sum(total_recall) / len(total_recall)
average_f1_score = sum(total_f1) / len(total_f1)
result.append({"Average Precision": average_precision})
result.append({"Average Recall": average_recall})
result.append({"Average F1-Score": average_f1_score})

# save result
output_dir = "/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/gnn"
# output_dir = "./"
output_path = os.path.join(output_dir, "test_completion_infoNCEloss_longformer.json")
# output_path = os.path.join(output_dir, "predicted_modified_paragraphs_test_completion_gnn.json")
with open(output_path, 'w') as f:
    json.dump(result, f, indent=4)
    
print(f"File saved to {output_path}")

print(f"Precision: {sum(total_precision) / len(total_precision)}")
print(f"Recall: {sum(total_recall) / len(total_recall)}")
print(f"F1-Score: {sum(total_f1) / len(total_f1)}")