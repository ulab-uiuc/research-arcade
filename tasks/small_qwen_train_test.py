import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import math
import json
import numpy as np
from typing import Dict, List, Optional
import random
from tqdm import tqdm
import wandb
from sklearn.cluster import KMeans
# 强制使用GPU
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login(key="6ddb8013e0510de0df50ceb98eba190007f91dee")

def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class MultiModalQwenEmbedding(nn.Module):
    def __init__(self, base_model_name: str, embedding_dim: int,
                 use_lora: bool = True, lora_config: Optional[LoraConfig] = None):
        super().__init__()
        self.device = device
        self.use_lora = use_lora

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # 加载基础模型
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=device
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = self.base_model.config.hidden_size
        self.embedding_projector = nn.Linear(embedding_dim, self.hidden_size).cuda()

        # 移除输出projector，直接使用Qwen3的embedding维度(1024)
        # self.output_projector = nn.Linear(self.hidden_size, item_embedding_dim).cuda()

        self.embedding_token = "<|embedding|>"
        special_tokens = {"additional_special_tokens": [self.embedding_token]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # 应用LoRA
        if use_lora:
            if lora_config is None:
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=16,  # LoRA rank
                    lora_alpha=32,  # LoRA alpha
                    lora_dropout=0.1,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"
                    ],  # 适用于Qwen模型的target modules
                    bias="none",
                )

            print("🔧 Applying LoRA to base model...")
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()

    def forward(self, input_ids, attention_mask=None, external_embeddings=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        if external_embeddings is not None:
            external_embeddings = external_embeddings.cuda()

        text_embeds = self.base_model.get_input_embeddings()(input_ids)

        if external_embeddings is not None:
            B, N, D = external_embeddings.shape
            external_embeddings_proj = self.embedding_projector(
                external_embeddings.view(B * N, D)
            ).view(B, N, -1)

            text_embeds = torch.cat([external_embeddings_proj, text_embeds], dim=1)

            if attention_mask is not None:
                prefix_mask = torch.ones((B, N), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.base_model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask
        )

        last_hidden_state = outputs.last_hidden_state
        # pooled_output = torch.mean(last_hidden_state, dim=1)

        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        masked_sum = (last_hidden_state * mask).sum(dim=1)  # [B, H]
        lengths = mask.sum(dim=1).clamp_min(1e-6)  # [B, 1]
        pooled_output = masked_sum / lengths
        # pooled_output = last_token_pool(last_hidden_state, attention_mask)

        # 直接返回Qwen3的embedding，不经过额外的projection
        return pooled_output

    def save_pretrained(self, save_directory):
        """保存LoRA模型和projector"""
        import os
        os.makedirs(save_directory, exist_ok=True)

        # 保存tokenizer
        self.tokenizer.save_pretrained(save_directory)

        # 如果使用LoRA，保存LoRA权重
        if self.use_lora:
            self.base_model.save_pretrained(save_directory)
        else:
            # 如果没有使用LoRA，保存整个模型
            torch.save(self.base_model.state_dict(), os.path.join(save_directory, "base_model.bin"))

        # 只保存embedding_projector权重，移除output_projector
        projector_state = {
            'embedding_projector': self.embedding_projector.state_dict(),
        }
        torch.save(projector_state, os.path.join(save_directory, "projector_weights.bin"))

        # 保存模型配置
        config = {
            "hidden_size": self.hidden_size,
            "embedding_dim": self.embedding_projector.in_features,
            "use_lora": self.use_lora,
        }

        with open(os.path.join(save_directory, "model_config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {save_directory}")

    @classmethod
    def load_pretrained(cls, model_directory, base_model_name, lora_config: Optional[LoraConfig] = None):
        """加载预训练的LoRA模型"""
        import os

        # 加载配置
        config_path = os.path.join(model_directory, "model_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        use_lora = config.get("use_lora", True)

        # 创建模型实例（不应用LoRA）
        model = cls(
            base_model_name=base_model_name,
            embedding_dim=config["embedding_dim"],
            use_lora=False,  # 先创建不含LoRA的模型
            lora_config=None
        )

        # 如果原模型使用了LoRA，加载LoRA权重
        if use_lora:
            model.base_model = PeftModel.from_pretrained(
                model.base_model,
                model_directory
            )
            model.use_lora = True
        else:
            # 加载完整的基础模型权重
            base_model_path = os.path.join(model_directory, "base_model.bin")
            model.base_model.load_state_dict(torch.load(base_model_path, map_location="cuda:0"))

        # 只加载embedding_projector权重
        projector_path = os.path.join(model_directory, "projector_weights.bin")
        projector_state = torch.load(projector_path, map_location="cuda:0")

        model.embedding_projector.load_state_dict(projector_state['embedding_projector'])

        print(f"Model loaded from {model_directory}")
        return model

    def get_trainable_parameters(self):
        """获取可训练参数数量"""
        trainable_params = 0
        all_param = 0

        for param in self.parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || "
              f"Trainable%: {100 * trainable_params / all_param:.2f}%")

        return trainable_params, all_param


class MultiModalDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, max_negatives: int = 10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_negatives = max_negatives
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['problem']
        embedding = np.array(item['global_info_embedding'], dtype=np.float32)
        positive_item_embedding = np.array(item['postive_embedding'], dtype=np.float32)
        negative_item_embeddings = np.array(item['negetive_embedding'], dtype=np.float32)

        if len(negative_item_embeddings) > self.max_negatives:
            indices = np.random.choice(len(negative_item_embeddings), self.max_negatives, replace=False)
            negative_item_embeddings = negative_item_embeddings[indices]

        num_negatives = len(negative_item_embeddings)
        input_text = f"<|embedding|> {text}"

        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'external_embeddings': torch.tensor(embedding, dtype=torch.float32),
            'positive_item_embedding': torch.tensor(positive_item_embedding, dtype=torch.float32),
            'negative_item_embeddings': torch.tensor(negative_item_embeddings, dtype=torch.float32),
            'num_negatives': num_negatives
        }


class ValidationDataset(Dataset):
    """验证集数据集，用于MRR计算"""

    def __init__(self, data: List[Dict], name_row_dict, embedding_data, tokenizer, max_length: int = 512):
        self.data = data
        self.data = {i: v for i, v in enumerate(self.data.values())}
        self.name_row_dict = name_row_dict
        self.name_row_dict_reverse = {v: k for k, v in self.name_row_dict.items()}

        # 预转换为numpy array和tensor，避免重复转换
        self.embedding_data = np.array(embedding_data, dtype=np.float32)
        self.embedding_data_tensor = torch.from_numpy(self.embedding_data).float()

        self.index_list = list(range(len(embedding_data)))

        # 预计算并转换为tensor
        cluster_centers = np.array(self.get_kmeans_cluster_centers(self.embedding_data), dtype=np.float32)
        self.embed = cluster_centers
        self.embed_tensor = torch.from_numpy(cluster_centers).float()

        # 预计算name lookup
        self.name_lookup = np.array([self.name_row_dict_reverse.get(i, f"item_{i}")
                                     for i in range(len(self.embedding_data))])

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def build_val_sample(self, seq, window_size=50):
        """
        Build the validation sample from the sequence.
        - Last item of seq -> test label
        - Second last item -> validation label
        - Validation sample: window_size features before validation label + 1 validation label
        """
        if len(seq) <= window_size + 1:
            raise ValueError("Sequence length must be greater than window_size + 1")

        # 验证 label 的索引
        val_label_idx = len(seq) - 2

        # 提取验证窗口（window_size features + 1 val label）
        start = val_label_idx - window_size
        sub_seq = seq[start: val_label_idx + 1]

        return sub_seq

    def build_test_sample(self, seq, window_size=20):
        """构建测试样本"""
        if len(seq) <= window_size:
            raise ValueError("Sequence length must be greater than window_size")

        # label 取最后一个元素
        test_label_idx = len(seq) - 1
        start = test_label_idx - window_size
        sub_seq = seq[start: test_label_idx + 1]
        return sub_seq

    def get_kmeans_cluster_centers(self, tem_all_embedding, n_clusters=20, random_state=42):
        """
        Perform KMeans clustering on the given embeddings and return cluster centers.

        Args:
            tem_all_embedding (list or np.ndarray): List/array of embeddings, shape (N, D).
            n_clusters (int): Number of clusters.
            random_state (int): Random seed for reproducibility.

        Returns:
            list: A list of cluster center embeddings (each a list of floats).
        """
        # 转换成 numpy 数组
        embeddings = np.array(tem_all_embedding, dtype=np.float32)

        # 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(embeddings)
        labels = kmeans.labels_

        # 计算每个 cluster 的均值向量
        cluster_centers = []
        for k in range(n_clusters):
            cluster_embeddings = embeddings[labels == k]
            cluster_mean = cluster_embeddings.mean(axis=0)
            cluster_centers.append(cluster_mean.tolist())

        return cluster_centers

    def __getitem__(self, idx):
        item = self.data[idx]
        sequence = self.build_test_sample(item)
        positive_idx = sequence[-1]

        # 优化: 直接切片，避免复杂计算
        recent_20_idx = sequence[-21:-1] if len(sequence) >= 21 else sequence[:-1]

        # 使用预计算的name lookup
        recent_20_idx_name = self.name_lookup[recent_20_idx].tolist()
        text = "These are the most recent 20 user interaction records, listed from oldest to newest:" + str(
            recent_20_idx_name)

        # 直接使用预转换的tensor
        gt_embedding = self.embedding_data_tensor[positive_idx]

        # 高效的负样本选择：使用torch操作
        mask = torch.ones(len(self.embedding_data_tensor), dtype=torch.bool)
        mask[positive_idx] = False
        negative_item_embeddings = self.embedding_data_tensor[mask]

        input_text = f"<|embedding|> {text}"
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'external_embeddings': self.embed_tensor,  # 直接使用预转换的tensor
            'positive_item_embedding': gt_embedding,  # 已经是tensor
            'negative_item_embeddings': negative_item_embeddings,  # 已经是tensor
        }


class MultiModalDataCollator:
    def __init__(self, tokenizer, max_negatives: int = 10):
        self.tokenizer = tokenizer
        self.max_negatives = max_negatives

    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        external_embeddings = torch.stack([item['external_embeddings'] for item in batch])
        positive_item_embeddings = torch.stack([item['positive_item_embedding'] for item in batch])

        batch_size = len(batch)
        embedding_dim = batch[0]['negative_item_embeddings'].shape[-1]

        padded_negatives = torch.zeros(batch_size, self.max_negatives, embedding_dim)
        neg_masks = torch.zeros(batch_size, self.max_negatives, dtype=torch.bool)

        for i, item in enumerate(batch):
            neg_embs = item['negative_item_embeddings']
            num_negs = min(len(neg_embs), self.max_negatives)
            padded_negatives[i, :num_negs] = neg_embs[:num_negs]
            neg_masks[i, :num_negs] = True

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'external_embeddings': external_embeddings,
            'positive_item_embeddings': positive_item_embeddings,
            'negative_item_embeddings': padded_negatives,
            'negative_masks': neg_masks
        }


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, user_embeddings, positive_item_embeddings, negative_item_embeddings, negative_masks=None):
        batch_size = user_embeddings.size(0)
        user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
        positive_item_embeddings = F.normalize(positive_item_embeddings, p=2, dim=-1)
        negative_item_embeddings = F.normalize(negative_item_embeddings, p=2, dim=-1)

        # 使用内积计算相似度
        pos_sim = torch.sum(user_embeddings * positive_item_embeddings, dim=-1) / self.temperature

        neg_sim = torch.bmm(
            user_embeddings.unsqueeze(1),
            negative_item_embeddings.transpose(-2, -1)
        ).squeeze(1) / self.temperature

        if negative_masks is not None:
            neg_sim = neg_sim.masked_fill(~negative_masks, -1e9)

        losses = []
        for i in range(batch_size):
            if negative_masks is not None:
                valid_neg_sim = neg_sim[i][negative_masks[i]]
            else:
                valid_neg_sim = neg_sim[i]

            all_sim = torch.cat([pos_sim[i:i + 1], valid_neg_sim])
            loss = -pos_sim[i] + torch.logsumexp(all_sim, dim=0)
            losses.append(loss)

        return torch.stack(losses).mean()


class MRREvaluator:
    def __init__(self, model, tokenizer, validation_dataset: ValidationDataset):
        self.model = model
        self.tokenizer = tokenizer
        self.validation_dataset = validation_dataset

    def evaluate_mrr(self, batch_size: int = 256) -> float:
        self.model.eval()
        mrr_scores = []

        val_dataloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._validation_collate_fn
        )

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating MRR"):
                batch_mrr = self._compute_batch_mrr(batch)
                mrr_scores.extend(batch_mrr)

        return np.mean(mrr_scores)

    def _validation_collate_fn(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        external_embeddings = torch.stack([item['external_embeddings'] for item in batch])
        positive_item_embeddings = torch.stack([item['positive_item_embedding'] for item in batch])
        negative_item_embeddings = [item['negative_item_embeddings'] for item in batch]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'external_embeddings': external_embeddings,
            'positive_item_embeddings': positive_item_embeddings,
            'negative_item_embeddings': negative_item_embeddings,
        }

    def _compute_batch_mrr(self, batch) -> List[float]:
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        external_embeddings = batch['external_embeddings'].cuda()
        positive_item_embeddings = batch['positive_item_embeddings'].cuda()

        user_embeddings = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            external_embeddings=external_embeddings
        )

        user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
        positive_item_embeddings = F.normalize(positive_item_embeddings, p=2, dim=-1)

        batch_mrr = []
        for i in range(len(user_embeddings)):
            user_emb = user_embeddings[i]
            pos_emb = positive_item_embeddings[i]
            neg_embs = F.normalize(batch['negative_item_embeddings'][i].cuda(), p=2, dim=-1)

            all_items = torch.cat([pos_emb.unsqueeze(0), neg_embs], dim=0)
            # 使用内积计算相似度
            similarities = torch.matmul(user_emb, all_items.t())
            sorted_indices = torch.argsort(similarities, descending=True)
            pos_rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
            mrr = 1.0 / pos_rank
            batch_mrr.append(mrr)

        return batch_mrr


class BestMRRCallback(TrainerCallback):
    def __init__(self, mrr_evaluator: MRREvaluator, save_dir: str = "./best_model",
                 eval_steps: int = 10, save_strategy: str = "best_only", use_wandb: bool = True):
        """
        Args:
            mrr_evaluator: MRR评估器
            save_dir: 模型保存目录
            eval_steps: 评估间隔步数
            save_strategy: 保存策略
                - "best_only": 只保存最佳MRR的模型（默认）
                - "always": 每次评估都保存模型
                - "both": 既保存最佳模型，也保存最新模型
            use_wandb: 是否使用wandb记录MRR
        """
        self.mrr_evaluator = mrr_evaluator
        self.save_dir = save_dir
        self.eval_steps = eval_steps
        self.save_strategy = save_strategy
        self.use_wandb = use_wandb
        self.last_eval_step = 0
        self.best_mrr = -1.0

        # 为不同保存策略创建目录
        if self.save_strategy == "both":
            self.best_save_dir = os.path.join(save_dir, "best_model")
            self.latest_save_dir = os.path.join(save_dir, "latest_model")
            os.makedirs(self.best_save_dir, exist_ok=True)
            os.makedirs(self.latest_save_dir, exist_ok=True)
        else:
            os.makedirs(save_dir, exist_ok=True)

    def on_log(self, args, state, control, model, logs=None, **kwargs):
        if (state.global_step > 0 and state.global_step - self.last_eval_step >= self.eval_steps):
            print(f"\n🔍 Evaluating MRR at step {state.global_step}...")
            mrr_score = self.mrr_evaluator.evaluate_mrr()
            print(f"📊 MRR Score: {mrr_score:.4f}")

            # 记录到wandb
            if self.use_wandb:
                wandb.log({
                    "eval/mrr": mrr_score,
                    "eval/best_mrr": max(self.best_mrr, mrr_score),
                    "eval/step": state.global_step,
                    "eval/epoch": state.epoch
                }, step=state.global_step)

            if logs is not None:
                logs["eval_mrr"] = mrr_score

            # 根据保存策略决定如何保存模型
            if self.save_strategy == "best_only":
                # 只保存最佳模型
                if mrr_score > self.best_mrr:
                    print(f"🏆 New best MRR! Saving model to {self.save_dir} "
                          f"(MRR improved from {self.best_mrr:.4f} to {mrr_score:.4f})")
                    self.best_mrr = mrr_score
                    model.save_pretrained(self.save_dir)

                    # 记录新的最佳MRR到wandb
                    if self.use_wandb:
                        wandb.log({
                            "eval/new_best_mrr": mrr_score,
                            "eval/mrr_improvement": mrr_score - (self.best_mrr if self.best_mrr != mrr_score else 0)
                        }, step=state.global_step)
                else:
                    print(f"📊 MRR {mrr_score:.4f} did not improve from best {self.best_mrr:.4f}. Not saving.")

            elif self.save_strategy == "always":
                # 每次都保存模型
                print(f"💾 Saving latest model to {self.save_dir} (MRR: {mrr_score:.4f})")
                model.save_pretrained(self.save_dir)
                if mrr_score > self.best_mrr:
                    print(f"🏆 New best MRR achieved! (improved from {self.best_mrr:.4f} to {mrr_score:.4f})")
                    old_best = self.best_mrr
                    self.best_mrr = mrr_score

                    # 记录新的最佳MRR到wandb
                    if self.use_wandb:
                        wandb.log({
                            "eval/new_best_mrr": mrr_score,
                            "eval/mrr_improvement": mrr_score - old_best
                        }, step=state.global_step)

            elif self.save_strategy == "both":
                # 既保存最佳模型，也保存最新模型
                print(f"💾 Saving latest model to {self.latest_save_dir} (MRR: {mrr_score:.4f})")
                model.save_pretrained(self.latest_save_dir)

                if mrr_score > self.best_mrr:
                    print(f"🏆 New best MRR! Saving best model to {self.best_save_dir} "
                          f"(MRR improved from {self.best_mrr:.4f} to {mrr_score:.4f})")
                    old_best = self.best_mrr
                    self.best_mrr = mrr_score
                    model.save_pretrained(self.best_save_dir)

                    # 记录新的最佳MRR到wandb
                    if self.use_wandb:
                        wandb.log({
                            "eval/new_best_mrr": mrr_score,
                            "eval/mrr_improvement": mrr_score - old_best
                        }, step=state.global_step)
                else:
                    print(f"📊 MRR {mrr_score:.4f} did not improve from best {self.best_mrr:.4f}. "
                          f"Best model not updated.")

            model.train()
            self.last_eval_step = state.global_step


class MultiModalTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infonce_loss = InfoNCELoss().cuda()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].cuda()

        user_embeddings = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            external_embeddings=inputs['external_embeddings']
        )

        loss = self.infonce_loss(
            user_embeddings,
            inputs['positive_item_embeddings'],
            inputs['negative_item_embeddings'],
            inputs.get('negative_masks', None)
        )

        return (loss, user_embeddings) if return_outputs else loss


def load_all_data(file_path: str):
    """
    Load all user data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the JSON file is invalid.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def create_dataset_from_data(data_list: List[Dict],name_row_dict,embedding_data,binary_list, tokenizer, max_length: int = 512, max_negatives: int = 99):
    class ListDataset(Dataset):
        def __init__(self, data_list, tokenizer, max_length, max_negatives):
            self.data = data_list
            self.data = {i: v for i, v in enumerate(self.data.values())}
            self.name_row_dict=name_row_dict
            self.name_row_dict_reverse = {v: k for k, v in self.name_row_dict.items()}
            self.embedding_data=embedding_data
            self.index_list = list(range(len(embedding_data)))
            self.binary_list=binary_list
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.max_negatives = max_negatives

        def __len__(self):
            return len(self.data)

        def get_kmeans_cluster_centers(self, tem_all_embedding, n_clusters=20, random_state=42):
            """
            Perform KMeans clustering on the given embeddings and return cluster centers.

            Args:
                tem_all_embedding (list or np.ndarray): List/array of embeddings, shape (N, D).
                n_clusters (int): Number of clusters.
                random_state (int): Random seed for reproducibility.

            Returns:
                list: A list of cluster center embeddings (each a list of floats).
            """
            # 转换成 numpy 数组
            embeddings = np.array(tem_all_embedding, dtype=np.float32)

            if embeddings.shape[0] < n_clusters:
                raise ValueError(f"Number of samples ({embeddings.shape[0]}) "
                                 f"is less than n_clusters ({n_clusters})")

            # 聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(embeddings)
            labels = kmeans.labels_

            # 计算每个 cluster 的均值向量
            cluster_centers = []
            for k in range(n_clusters):
                cluster_embeddings = embeddings[labels == k]
                cluster_mean = cluster_embeddings.mean(axis=0)
                cluster_centers.append(cluster_mean.tolist())

            return cluster_centers

        def build_train_sample(self, seq, window_size=20):
            """
            Build one training sample from the sequence.
            - Last item of seq -> test label (excluded from training window)
            - Second last item -> validation label (excluded from training window)
            - Training window: window_size features + 1 training label
            """

            # 可用于训练的部分（排除验证和测试 label）
            train_seq_part = seq[:-2]

            # 训练窗口长度 = 特征数 + 1 个训练 label
            win_len = window_size + 1

            # 随机起始位置，保证窗口不越界
            start = random.randint(0, len(train_seq_part) - win_len)

            # 提取训练窗口
            sub_seq = train_seq_part[start: start + win_len]

            return sub_seq

        def __getitem__(self, idx):
            item = self.data[idx]
            sequence=self.build_train_sample(item)
            positive_idx = sequence[-1]
            negative_idx = [x for x in self.index_list if x != positive_idx]
            recent_20_idx=sequence[-2 - 19: -1]
            recent_20_idx_name=[self.name_row_dict_reverse[x] for x in recent_20_idx]
            text = "These are the most recent 20 user interaction records, listed from oldest to newest:"+str(recent_20_idx_name)

            # 从 self.binary_list 中随机选一个值
            neg_num = random.choice(self.binary_list)

            # 从 negative_idx 中随机选 neg_num 个不重复的值
            binary_neg_idx = random.sample(negative_idx, neg_num)
            tem_all_idx=binary_neg_idx+[positive_idx]
            tem_all_embedding=self.embedding_data[tem_all_idx]
            embedding = np.array(self.get_kmeans_cluster_centers(tem_all_embedding), dtype=np.float32)

            positive_item_embedding = np.array(self.embedding_data[positive_idx], dtype=np.float32)
            negative_item_embeddings = np.array(self.embedding_data[binary_neg_idx], dtype=np.float32)

            if len(negative_item_embeddings) > self.max_negatives:
                indices = np.random.choice(len(negative_item_embeddings), self.max_negatives, replace=False)
                negative_item_embeddings = negative_item_embeddings[indices]

            num_negatives = len(negative_item_embeddings)
            input_text = f"<|embedding|> {text}"
            
            encoding = self.tokenizer(
                input_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'external_embeddings': torch.tensor(embedding, dtype=torch.float32),
                'positive_item_embedding': torch.tensor(positive_item_embedding, dtype=torch.float32),
                'negative_item_embeddings': torch.tensor(negative_item_embeddings, dtype=torch.float32),
                'num_negatives': num_negatives
            }

    return ListDataset(data_list, tokenizer, max_length, max_negatives)

def get_binary_list(length: int):
    # 初始化列表，先放原始长度
    results = [length]

    num = length
    while num >= 100:
        num = num // 2
        if num >= 100:
            results.append(num)

    # 每个值减 1
    binary_list = [x - 1 for x in results]
    return binary_list


def sinusoidal_position_encoding(n_items, d_model=1024, device="cpu"):
    """
    生成 [n_items, d_model] 的 sinusoidal PE 矩阵
    """
    position = torch.arange(0, n_items, dtype=torch.float, device=device).unsqueeze(1)  # [n_items, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
                         -(math.log(10000.0) / d_model))

    pe = torch.zeros(n_items, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
    return pe

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用！请检查GPU和CUDA安装。")

    print(f"Using device: {torch.cuda.get_device_name(0)}")

    base_model_name = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim = 1024
    max_length = 512

    # 修改为分别指定训练和测试数据路径

    all_user_data_path="/data/taofeng2/Lranker/zijie/LRanker_data_binary_search_pt_seperate_json/movie/movie_user_over_53.json"
    user_all=load_all_data(all_user_data_path)

    embedding_file_path = "/data/taofeng2/Lranker/zijie/LRanker_data_binary_search_pt_seperate_json/movie/ML-1M_unique_movies_with_embeddings.pt"
    embedding_data = torch.load(embedding_file_path)
    device = embedding_data.device
    pe = sinusoidal_position_encoding(embedding_data.size(0), embedding_data.size(1), device=device)
    # 各自 L2 norm
    # embedding_data = F.normalize(embedding_data, p=2, dim=-1)
    # pe = F.normalize(pe, p=2, dim=-1)

    # 再相加
    # final_embedding = embedding_data + pe
    final_embedding =  pe
    binary_list = get_binary_list(len(embedding_data))

    name_row_dict_path="/data/taofeng2/Lranker/zijie/LRanker_data_binary_search_pt_seperate_json/movie/ML-1M_unique_movies_with_embeddings_mapping.json"
    name_row_dict=load_all_data(name_row_dict_path)


    # ============ 配置保存策略 ============
    # 可选值:
    # "best_only" - 只保存最佳MRR的模型（推荐用于最终训练）
    # "always"    - 每次评估都保存模型（用于调试或想要看到训练进程）
    # "both"      - 既保存最佳模型，也保存最新模型（用于完整监控）
    SAVE_STRATEGY = "both"  # 在这里修改保存策略
    USE_WANDB = True  # 是否使用wandb记录

    # 初始化wandb
    if USE_WANDB:
        wandb.init(
            project="lora-qwen-embedding",  # 项目名称，可以根据需要修改
            name=f"qwen3-embedding-lora-{SAVE_STRATEGY}",  # 运行名称
            config={
                "base_model": base_model_name,
                "embedding_dim": embedding_dim,
                "max_length": max_length,
                "save_strategy": SAVE_STRATEGY,
                "lora_r": 2,
                "lora_alpha": 32,
                "lora_dropout": 0.2,
                "learning_rate": 1e-5,
                "per_device_train_batch_size": 20,
                "num_train_epochs": 3,
                "eval_steps": 10,
                "max_negatives": 10
            },
            tags=["lora", "qwen", "embedding", "mrr", "infonce"]
        )
        print("📊 Wandb初始化完成，将记录MRR评估结果")

    print(f"📁 使用保存策略: {SAVE_STRATEGY}")
    if SAVE_STRATEGY == "best_only":
        print("   - 只有当MRR提升时才会保存模型")
    elif SAVE_STRATEGY == "always":
        print("   - 每次评估都会保存最新模型")
    elif SAVE_STRATEGY == "both":
        print("   - 每次评估都保存最新模型，MRR提升时额外保存最佳模型")

    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16,  # LoRA rank，可以调整
        lora_alpha=32,  # LoRA alpha，通常是r的2倍
        lora_dropout=0.2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )

    # 创建使用LoRA的模型
    model = MultiModalQwenEmbedding(
        base_model_name,
        embedding_dim,
        use_lora=True,
        lora_config=lora_config
    )
    model = model.cuda()

    # 打印可训练参数信息
    print("🔍 Model parameter statistics:")
    model.get_trainable_parameters()

    # 创建数据集（现在测试数据完全来自独立的测试文件）
    train_dataset = create_dataset_from_data(user_all,name_row_dict,embedding_data,binary_list, model.tokenizer, max_length, max_negatives=9)

    # 固定随机种子
    random.seed(42)

    # 随机选择 320 个用户
    sampled_items = dict(random.sample(list(user_all.items()), 320))

    val_dataset = ValidationDataset(
        sampled_items,
        name_row_dict,
        embedding_data,
        model.tokenizer,
        max_length
    )


    # 创建MRR评估器和callback
    mrr_evaluator = MRREvaluator(model, model.tokenizer, val_dataset)
    best_mrr_callback = BestMRRCallback(
        mrr_evaluator,
        save_dir="./lora_model_output",
        eval_steps=20,
        save_strategy=SAVE_STRATEGY,  # 使用配置的保存策略
        use_wandb=USE_WANDB  # 传递wandb配置
    )

    # 数据collator
    data_collator = MultiModalDataCollator(model.tokenizer, max_negatives=9)

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./lora_embedding_model_output_",
        per_device_train_batch_size=20,
        num_train_epochs=5,
        learning_rate=1e-5,
        logging_steps=10,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        save_strategy="no",
        eval_strategy="no",
        warmup_steps=1,
        max_grad_norm=0.5,
        dataloader_num_workers=10,
        report_to="wandb" if USE_WANDB else None,  # 设置report_to
    )

    # 创建训练器
    trainer = MultiModalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[best_mrr_callback],
    )

    print(f"🚀 开始LoRA训练，使用InfoNCE loss和MRR评估...")
    print(f"📁 保存策略: {SAVE_STRATEGY}")

    # 训练前先评估一次MRR
    print("📊 Initial MRR evaluation on test set...")
    initial_mrr = mrr_evaluator.evaluate_mrr()
    print(f"📊 Initial MRR Score: {initial_mrr:.4f}")

    # 记录初始MRR到wandb
    if USE_WANDB:
        wandb.log({
            "eval/initial_mrr": initial_mrr,
            "eval/step": 0
        }, step=0)

    # 开始训练
    trainer.train()

    print("✅ LoRA训练完成!")

    # 最终MRR评估
    print("📊 Final MRR evaluation on test set...")
    final_mrr = mrr_evaluator.evaluate_mrr()
    print(f"📊 Final MRR Score: {final_mrr:.4f}")
    print(f"📈 MRR Improvement: {final_mrr - initial_mrr:.4f}")

    # 记录最终结果到wandb
    if USE_WANDB:
        wandb.log({
            "eval/final_mrr": final_mrr,
            "eval/total_improvement": final_mrr - initial_mrr,
            "eval/best_mrr_achieved": best_mrr_callback.best_mrr
        })

        # 创建MRR改进的summary
        wandb.run.summary["best_mrr"] = best_mrr_callback.best_mrr
        wandb.run.summary["initial_mrr"] = initial_mrr
        wandb.run.summary["final_mrr"] = final_mrr
        wandb.run.summary["total_improvement"] = final_mrr - initial_mrr

    # 根据保存策略显示保存位置
    if SAVE_STRATEGY == "best_only":
        print(f"📁 最佳LoRA模型已保存到 ./lora_model_output")
        print(f"🏆 最佳MRR分数: {best_mrr_callback.best_mrr:.4f}")
    elif SAVE_STRATEGY == "always":
        print(f"📁 最新LoRA模型已保存到 ./lora_model_output")
        print(f"🏆 训练过程中最佳MRR分数: {best_mrr_callback.best_mrr:.4f}")
    elif SAVE_STRATEGY == "both":
        print(f"📁 最新LoRA模型已保存到 ./lora_model_output/latest_model")
        print(f"📁 最佳LoRA模型已保存到 ./lora_model_output/best_model")
        print(f"🏆 最佳MRR分数: {best_mrr_callback.best_mrr:.4f}")

    # 展示最终的参数统计
    print("\n🎯 Final model statistics:")
    model.get_trainable_parameters()

    # 完成wandb运行
    if USE_WANDB:
        wandb.finish()
        print("📊 Wandb运行已完成，可在wandb.ai查看详细结果")


if __name__ == "__main__":
    main()