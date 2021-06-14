from torch import nn
from transformers import PreTrainedTokenizerFast
from model_map import model_config


class PretrainedModelSelector:
    def __init__(self, models_dir:str):
        self.models_dir = models_dir

    def get_model(self, model_name: str, )->nn.Module:
        model_info = model_config[model_name]
        model = model_info.pretrained_model
        return model.from_pretrained(
                self.models_dir + "/" + model_name,
                config = model_info.config(num_labels=model_info.number_of_categories))

    def get_tokenizer(self, model_name) -> PreTrainedTokenizerFast:
        model_info = model_config[model_name]
        tokenizer = model_info.tokenizer
        return tokenizer.from_pretrained(
                self.models_dir + "/" + model_name))
