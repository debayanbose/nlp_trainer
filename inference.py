import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast


class NoVPredictor:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, training_config) -> None:
        self.model = model.to(training_config.device)
        self.tokenizer = tokenizer
        self.training_config = training_config

    def predict(self, input_text: str) ->int:
        full_text_encoding = self.tokenizer(input_text,
                                            truncation=True,
                                            padding=True,
                                            max_length=512,
                                            return_tensors='pt')
        input_ids, attention_mask = full_text_encoding['input_ids'].to(self.training_config.device), full_text_encoding['attention_mask'].to(self.training_config.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask = attention_mask)
            nov_pred = torch.argmax(output.logits,dim=1).cpu().numpy()[-1]
            return nov_pred.item()
