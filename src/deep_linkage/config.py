from pydantic import BaseModel


class DeepERConfig(BaseModel):
    # Data paths
    default_data_path: str = r"D:\Pycharm Projects\ktt-transcription-main\src\deep_linkage\test_human_ground_truth.csv"

    # Model settings
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 256

    # FAISS / Blocking settings
    top_k_neighbors: int = 10
    distance_threshold: float = 0.15

    # Training settings
    corruption_rate: float = 0.15
    epochs_of_synthetic_data: int = 5
    training_epochs: int = 3
    training_batch_size: int = 16
    custom_model_out_path: str = "./custom_ktt_ocr_model"