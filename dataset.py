from torch.utils.data import Dataset, DataLoader
import torch
from config_data import NovDataFrameConfig

class NoVDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df_config = NovDataFrameConfig()

    def __getitem__(self, index):
        title = str(self.data[self.df_config.input_texts][index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
                                            title,
                                            None,
                                            add_special_tokens=True,
                                            max_length=self.max_len,
                                            pad_to_max_length=True,
                                            return_token_type_ids=True,
                                            truncation=True
                                            )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
        'input_ids' : torch.tensor(ids,dtype=torch.long),
        'attention_mask' : torch.tensor(mask, dtype=torch.long),
        'labels': torch.tensor(self.data.ENCODE_CAT[index],dtype=torch.long)
        }
