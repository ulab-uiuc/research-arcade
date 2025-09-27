import torch
import json
# from contriever.src.contriever import Contriever
# from transformers import AutoTokenizer
from transformers import LongformerModel, AutoTokenizer
from tqdm import tqdm
from torch_geometric.data import HeteroData
from collections import defaultdict
import os

def mean_pooling(last_hidden_state, attention_mask):
    # last_hidden_state: [B, T, H]
    # attention_mask:    [B, T]
    mask = attention_mask.unsqueeze(-1)          # [B, T, 1]
    summed = (last_hidden_state * mask).sum(1)   # [B, H]
    counted = mask.sum(1).clamp(min=1e-9)        # [B, 1]
    return summed / counted

def review_graph_transformation(reviews, tokenizer, embedding_model, device):
    # 创建 HeteroData 实例
    data = HeteroData()
    
    # 初始化数据结构
    reviewer_node_raw_features = []
    author_node_raw_features = []
    edge_index = defaultdict(list)
    
    # 创建节点特征字典和边索引
    reviewer_node_id_map = {}
    author_node_id_map = {}
    
    # 遍历所有 review 来构建图
    for review_list in reviews:
        for review in review_list:
            review_id = review["review_id"]
            replyto_id = review["replyto_id"]
            writer = review["writer"]
            text = review["text"]
            
            # 为每个评论创建一个唯一的节点
            if writer == "reviewer":
                reviewer_node_id_map[review_id] = len(reviewer_node_id_map)
                reviewer_node_raw_features.append(text)
                # reviewer -> reviewer
                if replyto_id in reviewer_node_id_map:
                    edge_index['reviewer', 'reviewer-to-reviewer', 'reviewer'].append([reviewer_node_id_map[replyto_id], reviewer_node_id_map[review_id]])
                # author -> reviewer
                elif replyto_id in author_node_id_map:
                    edge_index['author', 'author-to-reviewer', 'reviewer'].append([author_node_id_map[replyto_id], reviewer_node_id_map[review_id]])
            elif writer == "author":
                author_node_id_map[review_id] = len(author_node_id_map)
                author_node_raw_features.append(text)
                # reviewer -> author
                if replyto_id in reviewer_node_id_map:
                    edge_index['reviewer', 'reviewer-to-author', 'author'].append([reviewer_node_id_map[replyto_id], author_node_id_map[review_id]])
                # author -> author
                elif replyto_id in author_node_id_map:
                    edge_index['author', 'author-to-author', 'author'].append([author_node_id_map[replyto_id], author_node_id_map[review_id]])
    
    # format edge
    for key, value in edge_index.items():
        data[key].edge_index = torch.tensor(value, dtype=torch.long).t().contiguous()

    expected_edge_types = [
        ('reviewer', 'reviewer-to-reviewer', 'reviewer'),
        ('reviewer', 'reviewer-to-author', 'author'),
        ('author', 'author-to-reviewer', 'reviewer'),
        ('author', 'author-to-author', 'author')
    ]
    
    for edge_type in expected_edge_types:
        if edge_type not in data.edge_types:
            data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    assert len(data.edge_types) == 4
    # get embedding of node features
    embed_input_list = reviewer_node_raw_features + author_node_raw_features
    # inputs = tokenizer(embed_input_list, padding=True, truncation=True, return_tensors="pt").to(device)
    inputs = tokenizer(embed_input_list, padding=True, truncation=True, max_length=4096, return_tensors="pt").to(device)
    with torch.no_grad():
        # embeddings = embedding_model(**inputs)
        outputs = embedding_model(**inputs)
        token_embeddings = outputs.last_hidden_state
        sentence_embeddings = mean_pooling(token_embeddings, inputs["attention_mask"])
    
    reviewer_node_features = sentence_embeddings[:len(reviewer_node_raw_features)].cpu()
    author_node_features = sentence_embeddings[len(reviewer_node_raw_features):].cpu()
        
    # 为每种类型的节点添加特征
    data['reviewer'].x = reviewer_node_features
    data['author'].x = author_node_features
    
    return data

def original_paragraphs_transformation(original_paragraphs, tokenizer, embedding_model, device):
    #
    # inputs = tokenizer(original_paragraphs, padding=True, truncation=True, return_tensors="pt").to(device)
    inputs = tokenizer(original_paragraphs, padding=True, truncation=True, max_length=4096, return_tensors="pt").to(device)
    with torch.no_grad():
        # original_paragraphs_embeddings = embedding_model(**inputs).cpu()
        outputs = embedding_model(**inputs)
        token_embeddings = outputs.last_hidden_state
        original_paragraphs_embeddings = mean_pooling(token_embeddings, inputs["attention_mask"])
    
    return original_paragraphs_embeddings

# mode = "test"
mode = "train"
# Load dataset
with open(f"/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/gnn/{mode}_dataset.json", 'r') as file:
    dataset = json.load(file)
    
# Embedding model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# embedding_model = Contriever.from_pretrained("facebook/contriever", ignore_mismatched_sizes=True).to(device)
# tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
embedding_model = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(device)
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

# Preprocess dataset
review_graph_dataset = []
original_paragraphs_dataset = []
modified_paragraphs_labels = []
for key, value in tqdm(zip(dataset.keys(), dataset.values()), total=len(dataset), desc="Processing dataset"):
    try:
        review_graph = review_graph_transformation(value["reviews"], tokenizer, embedding_model, device)
        original_paragraphs_embeddings = original_paragraphs_transformation(value["original_paragraphs"], tokenizer, embedding_model, device)
    except:
        continue
    review_graph_dataset.append(review_graph) 
    original_paragraphs_dataset.append(original_paragraphs_embeddings)
    modified_paragraphs_labels.append(value["modified_paragraph_idx"])
    # break

assert len(review_graph_dataset) == len(original_paragraphs_dataset)
assert len(review_graph_dataset) == len(modified_paragraphs_labels)

output_dir = "/data/jingjunx/evaluation_tasks/predict_modified_paragraphs/gnn"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, f"{mode}_review_graph_dataset_longformer.pt")
torch.save(review_graph_dataset, output_file)
print("File save to "+output_file)

output_file = os.path.join(output_dir, f"{mode}_original_paragraphs_dataset_longformer.pt")
torch.save(original_paragraphs_dataset, output_file)
print("File save to "+output_file)

output_file = os.path.join(output_dir, f"{mode}_modified_paragraphs_labels_longformer.json")
with open(output_file, 'w') as f:
    json.dump(modified_paragraphs_labels, f)
print("File save to "+output_file)