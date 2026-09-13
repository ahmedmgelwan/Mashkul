import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class PathConfig:
    assets_dir: str = os.getenv("ASSETS_DIR", "assets")
    model_path: str = os.getenv("MODEL_PATH", "assets/model.pt")
    tokenizer_path: str = os.getenv("TOKENIZER_PATH", "assets/tokenizer.json")
    device: str = os.getenv("DEVICE", "cpu")
    api_url: str = os.getenv("API_URL", "http://localhost:8000")

@dataclass
class DataConfig:
    dataset_name: str = "arbml/tashkeela"
    train_size: int = 50_000
    val_size: int = 10_000
    test_size: int = 10_000
    window_size: int = 128


@dataclass
class ModelConfig:
    embedding_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    pad_idx: int = 0


@dataclass
class TrainConfig:
    batch_size: int = 512
    n_epochs: int = 10
    lr_patience: int = 2
    lr_factor: float = 0.5
    seed: int = 42

def get_device() -> str:
    env_device = os.getenv("DEVICE")
    if env_device:
        return env_device
    import torch  

    return "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ModelInfo:
    model_name = os.getenv("MODEL_NAME", "DiacritizationModelBiGRU")
    model_version = os.getenv("MODEL_VERSION", "1.0.0")
    model_description = os.getenv("MODEL_DESCRIPTION", "Arabic Diacritization Model")

DEVICE = get_device()
data_config = DataConfig()
model_config = ModelConfig()
train_config = TrainConfig()
model_info = ModelInfo()