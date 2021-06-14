import pandas as pd
from config_data import TrainingConfig, NovDataFrameConfig
from dataset import *
from inference import *


def level_encoding(training_config: TrainingConfig):
    if training_config.training_data.split(".")[-1] == "xlsx":
        df = pd.read_excel(training_config.training_data, engine = "openpyxl")
    else:
        df = pd.read_csv(training_config.training_data)

    if len(df) > 0:
        def encode_cat(x):
            return int(x - 1) if x<training_config.n_categories else training_config.n_categories - 1
        df['ENCODE_CAT'] = df[NovDataFrameConfig.nov_count].apply(lambda x: encode_cat(x))
        return df
    else:
        raise ValueError("Blank Data")


def train_test_split(training_config:TrainingConfig, tokenizer,fraction = 0.8):
    df = level_encoding(training_config)
    if len(df) < 2:
        raise ValueError('Insufficient Data size for train and test split')
    else:
        train_dataset = df.sample(frac=fraction, random_state=200).reset_index(drop=True)
        test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
        training_set = NoVDataset(train_dataset, tokenizer, training_config.max_len)
        testing_set = NoVDataset(test_dataset, tokenizer, training_config.max_len)
        training_loader = DataLoader(training_set,batch_size=training_config.batch_size, shuffle=True)
        testing_loader = DataLoader(testing_set,batch_size=training_config.batch_size, shuffle=True)
        return training_loader, testing_loader

def calculate_accu(big_idx: torch.Tensor, targets: torch.Tensor):
    return (big_idx == targets).sum().item()


def calculate_avg_accuracy_scores(nov_predictor: NoVPredictor, training_config: TrainingConfig,nov_dataframe_config: NovDataFrameConfig):
    df = level_encoding(training_config)
    total_size = len(df)
    correct = 0
    for _,row in df.iterrows():
        generated_prediction = nov_predictor.predict(str(row[nov_dataframe_config.source_sequence_column]))
        actual_answer = row['ENCODE_CAT']
        if generated_prediction == actual_answer:
            correct += 1
    return correct/total_size
