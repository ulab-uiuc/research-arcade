import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from torch_geometric.nn import HANConv
from torch_geometric.data import HeteroData
import logging
import json
import matplotlib.pyplot as plt
import os
from info_nce import InfoNCE, info_nce

def info_nce_loss(query, positive_key, negative_keys):
    '''
    query: batch_size, embedding_size
    positive_key: batch_size, embedding_size
    negative_key: num_negative, embedding_size
    '''
    loss = InfoNCE(negative_mode='unpaired')
    output = loss(query, positive_key, negative_keys)
    
    return output

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
        
# 加载图数据和标签 - 简化版本
def load_data(TRAIN_REVIEW_GRAPH_PATH, TRAIN_ORIGINAL_PARAGRAPHS_PATH, TRAIN_MODIFIED_PARAGRAPHS_PATH):
    # 加载你的数据
    review_graph_dataset = torch.load(TRAIN_REVIEW_GRAPH_PATH, weights_only=False)
    original_paragraphs_dataset = torch.load(TRAIN_ORIGINAL_PARAGRAPHS_PATH, weights_only=False)
    with open(TRAIN_MODIFIED_PARAGRAPHS_PATH, 'r') as f:
        modified_paragraphs_labels = json.load(f)
    
    assert len(review_graph_dataset) == len(original_paragraphs_dataset)
    assert len(review_graph_dataset) == len(modified_paragraphs_labels)

    # 简单划分训练集和验证集
    split_idx = int(len(review_graph_dataset) * 0.9)
    
    review_graph_train_dataset = review_graph_dataset[:split_idx]
    original_paragraphs_train_dataset = original_paragraphs_dataset[:split_idx]
    modified_paragraphs_train_labels = modified_paragraphs_labels[:split_idx]
    
    review_graph_val_dataset = review_graph_dataset[split_idx:]
    original_paragraphs_val_dataset = original_paragraphs_dataset[split_idx:]
    modified_paragraphs_val_labels = modified_paragraphs_labels[split_idx:]
    
    return review_graph_train_dataset, original_paragraphs_train_dataset, modified_paragraphs_train_labels, review_graph_val_dataset, original_paragraphs_val_dataset, modified_paragraphs_val_labels

# 配置日志
logging.basicConfig(filename='train_log_infoNCEloss_longformer.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
TRAIN_REVIEW_GRAPH_PATH="/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/gnn/train_review_graph_dataset_longformer.pt"
TRAIN_ORIGINAL_PARAGRAPHS_PATH="/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/gnn/train_original_paragraphs_dataset_longformer.pt"
TRAIN_MODIFIED_PARAGRAPHS_PATH="/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/gnn/train_modified_paragraphs_labels_longformer.json"

# 加载数据
review_graph_train_dataset, \
original_paragraphs_train_dataset, \
modified_paragraphs_train_labels, \
review_graph_val_dataset, \
original_paragraphs_val_dataset, \
modified_paragraphs_val_labels \
= load_data(TRAIN_REVIEW_GRAPH_PATH, TRAIN_ORIGINAL_PARAGRAPHS_PATH, TRAIN_MODIFIED_PARAGRAPHS_PATH)

print(f"Training samples: {len(review_graph_train_dataset)}")
print(f"Validation samples: {len(review_graph_val_dataset)}")

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphEmbeddingModel(in_channels=768, hidden_channels=128, out_channels=768, metadata=review_graph_train_dataset[0].metadata()) # 0 1
# model = GraphEmbeddingModel(in_channels=768, hidden_channels=256, out_channels=768, metadata=review_graph_train_dataset[0].metadata(), num_layers=3) # 2
model.to(device)

# 定义优化器
# optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=0.001) # 0
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)  # 1
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.005)  # 2

# 设置训练参数
epochs = 100

# 梯度累积参数
accumulation_steps = 16
# top_k = 5

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs*len(review_graph_train_dataset)/accumulation_steps), eta_min=0.0001) # 0
# total_steps = int(epochs * len(review_graph_train_dataset) / accumulation_steps) + 200 
# scheduler = OneCycleLR(
#     optimizer, 
#     max_lr=0.001,
#     total_steps=total_steps,
#     pct_start=0.1,  # 前10%步数用于warmup
#     anneal_strategy='cos'
# ) # 1
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7) # 2

# 训练过程记录
total_train_loss = []
total_val_loss = []

# 训练主循环
for epoch in range(epochs):
    # Training
    model.train()
    epoch_train_losses = []
    accumulated_loss = 0.0
    
    # 在每个epoch开始时清零梯度
    optimizer.zero_grad()
    
    for idx, (review_graph, original_paragraphs, modified_paragraphs_label) in enumerate(zip(review_graph_train_dataset, original_paragraphs_train_dataset, modified_paragraphs_train_labels)):
        # 移动数据到设备
        review_graph = review_graph.to(device)
        original_paragraphs = original_paragraphs.to(device)
        modified_paragraphs_label = modified_paragraphs_label
        
        if len(original_paragraphs) < max(modified_paragraphs_label):
            continue
        
        # 前向传播 - 单个样本
        output = model(review_graph.x_dict, review_graph.edge_index_dict)
        positive_keys = torch.stack([original_paragraphs[i-1] for i in modified_paragraphs_label]).to(device)
        negative_keys = torch.stack([original_paragraphs[i] for i in range(len(original_paragraphs)) if (i+1) not in modified_paragraphs_label]).to(device)
        
        loss = 0
        for query in output:
            queries = torch.stack([query for _ in range(len(positive_keys))])
            loss += info_nce_loss(queries, positive_keys, negative_keys) / len(output)
        
        loss = loss / accumulation_steps
        
        # 反向传播（累积梯度）
        loss.backward()
        
        # 累积损失用于记录
        accumulated_loss += loss.item()
        
        # 每accumulation_steps步或者最后一个样本时更新参数
        if (idx + 1) % accumulation_steps == 0 or (idx + 1) == len(review_graph_train_dataset):
            # 梯度更新
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # 记录累积的损失
            epoch_train_losses.append(accumulated_loss)
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs}, Step {(idx+1)//accumulation_steps + 1}, "
                  f"Sample {idx+1}/{len(review_graph_train_dataset)}, Accumulated Loss: {accumulated_loss:.4f}")
            
            # 重置累积损失
            accumulated_loss = 0.0
        
        # 每100个样本打印一次进度（可选）
        elif (idx + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Sample {idx+1}/{len(review_graph_train_dataset)}, "
                  f"Current Loss: {loss.item() * accumulation_steps:.4f}")  # 恢复原始loss大小显示

    # Validation
    model.eval()
    epoch_val_losses = []
    
    with torch.no_grad():
        for idx, (review_graph, original_paragraphs, modified_paragraphs_label) in enumerate(zip(review_graph_val_dataset, original_paragraphs_val_dataset, modified_paragraphs_val_labels)):
            review_graph = review_graph.to(device)
            original_paragraphs = original_paragraphs.to(device)
            modified_paragraphs_label = modified_paragraphs_label
            
            if len(original_paragraphs) < max(modified_paragraphs_label):
                continue
            
            # 前向传播
            output = model(review_graph.x_dict, review_graph.edge_index_dict)
            positive_keys = torch.stack([original_paragraphs[i-1] for i in modified_paragraphs_label]).to(device)
            negative_keys = torch.stack([original_paragraphs[i] for i in range(len(original_paragraphs)) if (i+1) not in modified_paragraphs_label]).to(device)
            
            loss = 0
            for query in output:
                queries = torch.stack([query for _ in range(len(positive_keys))])
                loss += info_nce_loss(queries, positive_keys, negative_keys) / len(output)
            
            epoch_val_losses.append(loss.item())
    
    # 计算平均损失
    avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
    
    total_train_loss.append(avg_train_loss)
    total_val_loss.append(avg_val_loss)
    
    # 记录日志
    logging.info(f'Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')
    
    # 打印结果
    print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')

# 保存模型
output_dir = "/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/gnn"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

torch.save(model.state_dict(), os.path.join(output_dir, 'HANConv_infoNCEloss_longformer.pth'))
print(f"Model saved to {os.path.join(output_dir, 'HANConv_infoNCEloss_longformer.pth')}")

# 绘制训练曲线
plt.figure(figsize=(10, 6))
plt.plot(total_train_loss, label='Train Loss', color='blue')
plt.plot(total_val_loss, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)
plt.savefig('training_loss_curve_HANConv_infoNCEloss_longformer.png', dpi=300, bbox_inches='tight')
print("Training curve saved to training_loss_curve_HANConv_infoNCEloss_longformer.png")