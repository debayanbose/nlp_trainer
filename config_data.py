from dataclasses import dataclass
import torch
from defaults import *

@dataclass
class TrainingConfig:
    checkpoint_dir: str
    pretrained_model_path: str
    training_data: str
    validation_data: str
    results_pathL str = DEFAULT_RESULTS_PATH
    device: torch.device('cuda') if torch.cuda is_available() else torch.device('cpu')
    lr: int = LEARNING_RATE
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    max_len: int = MAX_LEN
    n_categories : int = N_CATEGORIES


@dataclass
class NovDataFrameConfig:
    source_sequence_column: str = 'transcript'
    nov_count: str = 'nov'
