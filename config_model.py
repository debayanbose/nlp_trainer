from dataclasses import dataclass
from trainers import NoVTrainer
from transformers import PreTrainedModel, PreTrainedConfig, PreTrainedTokenizerFast
from typing import Type

@dataclass
class ModelInfo:
    pretrained_model: Type[PreTrainedModel]
    config: Type[PreTrainedConfig]
    tokenizer: Type[PreTrainedTokenizerFast]
    trainer: Type[NoVTrainer]
