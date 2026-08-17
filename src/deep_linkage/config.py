# config.py
from pydantic import BaseModel, Field


class DeepERConfig(BaseModel):
    """
    Master configuration for Deep Entity Resolution pipeline.
    """

    # Base Transformer settings
    model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Base HuggingFace model"
    )
    custom_model_out_path: str = Field(
        default="models/historical_deeper_model",
        description="Path to save fine-tuned model",
    )

    # Synthetic Data & Contrastive Training hyperparameters
    corruption_rate: float = Field(
        default=0.15, description="Probability of character-level OCR noise"
    )
    epochs_of_synthetic_data: int = Field(
        default=5, description="Synthetic corruption passes per real record"
    )
    training_batch_size: int = Field(
        default=32, description="Batch size for Triplet contrastive learning"
    )
    training_epochs: int = Field(
        default=3, description="Number of fine-tuning epochs"
    )

    # FAISS & Louvain Graph Clustering hyperparameters
    similarity_threshold: float = Field(
        default=0.85,
        description="Minimum cosine similarity for FAISS candidate edges",
    )
    top_k_neighbors: int = Field(
        default=10, description="Number of ANN neighbors to retrieve in FAISS"
    )
    louvain_resolution: float = Field(
        default=1.25,
        description="Modularity resolution (>1.0 favors denser clusters to prevent chaining)",
    )
    prune_weak_bridges: bool = Field(
        default=True,
        description="Sever unclosed triangle bridge edges before community detection",
    )