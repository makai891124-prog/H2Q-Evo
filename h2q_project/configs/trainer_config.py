from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TrainerConfig:
    # Basic training parameters
    max_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: str = "Adam"
    loss_function: str = "CrossEntropyLoss"
    scheduler: str = "StepLR"
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.1

    # Device configuration
    device: str = "cuda"  # or "cpu"

    # Logging and checkpointing
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    log_frequency: int = 10

    # Evaluation
    evaluate_every: int = 1

    # Custom parameters (for extensibility)
    custom_params: Dict[str, Any] = field(default_factory=dict)
