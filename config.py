from dataclasses import dataclass

@dataclass
class Config:
    clip_model: str = "ViT-B-16"
    clip_pretrained: str = "openai"
    feature_dim: int = 512
    similarity_dim: int = 256
    margin: float = 0.5
    heads: int = 4
    Gk: int = 10
    L: int = 6
    mu: float = 8.0
    eta: float = 1.0
    train_batch_size: int = 64
    test_batch_size: int = 128
    epochs: int = 20
    clip_lr: float = 1e-6
    framework_lr: float = 3e-5
