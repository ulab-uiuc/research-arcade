import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import numpy as np


class ExternalEmbedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize the ExternalEmbedder with a SentenceTransformer model.
        
        Args:
            model_name (str): Name of the SentenceTransformer model to use.
                             Default is "all-mpnet-base-v2" to match your existing code.
                             Note: "Qwen/Qwen3-0.6B" is not a SentenceTransformer model,
                             so using the same embedding model as in your dataset class.
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def get_embeddings(self, external_texts):
        emb = self.model.encode(external_texts, convert_to_tensor=True)
        return emb        
        
    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the model.
        
        Returns:
            int: The embedding dimension
        """
        return self.embedding_dim


# Helper function (same as in your code)
def get_mean_pooling(all_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling over a batch of external embeddings.
    
    Args:
        all_embeddings (torch.Tensor): [B, N, D] where
            B = batch size
            N = number of embeddings per sample
            D = embedding dimension
    
    Returns:
        torch.Tensor: [B, 1, D] mean-pooled embedding for each sample
    """
    if all_embeddings.dim() != 3:
        raise ValueError(f"Expected input shape [B, N, D], got {all_embeddings.shape}")
    
    # Average over the N dimension
    pooled = all_embeddings.mean(dim=1, keepdim=True)  # [B, 1, D]
    return pooled


# Example usage:
if __name__ == "__main__":
    # Initialize embedder
    embedder = ExternalEmbedder()
    
    # Example data
    paragraphs = ["This is a sample paragraph about machine learning.", 
                  "Another paragraph about neural networks."]
    figures = ["Figure 1 shows the training accuracy over time."]
    tables = ["Table 1: Comparison of different models\nModel A: 95%\nModel B: 92%"]
    bib_keys = "Smith et al. 2023; Johnson 2022"
    
    # Embed individual components
    para_emb = embedder.embed_paragraph(paragraphs[0])
    fig_emb = embedder.embed_figure_descriptions(figures)
    table_emb = embedder.embed_tables(tables)
    bib_emb = embedder.embed_bib_keys(bib_keys)
    
    print(f"Paragraph embedding shape: {para_emb.shape}")
    print(f"Figure embeddings shape: {fig_emb.shape}")
    print(f"Table embeddings shape: {table_emb.shape}")
    print(f"Bibliography embeddings shape: {bib_emb.shape}")
    
    # Embed all sources together (following your dataset logic)
    all_external_emb = embedder.embed_all_external_sources(
        adjacent_paragraphs=paragraphs,
        image_description_list=figures,
        table_contents=tables,
        bib_keys=bib_keys
    )
    print(f"All external sources embedding shape: {all_external_emb.shape}")