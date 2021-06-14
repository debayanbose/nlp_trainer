from model_map import model_config
from utils import *

class NumberOfVehiclesModelValidator:
    nov_df_config = NovDataFrameConfig()

    def __init__(self, training_config: TrainingConfig):
        self.training_config = training_config
        pretrained_model_name = training_config.pretrained_model_path.split("/")[-1]
        model_info = model_config(pretrained_model_name)
        tokenizer = model_info.tokenizer.from_pretrained(training_config.pretrained_model_path)
        pretrained_model = model_info.pretrained_model.from_pretrained(training_config.checkpoint_dir,
                                                                       config=model_info.config(num_labels=training_config.n_categories))
        self.nov_predictor = NoVPredictor(pretrained_model, tokenizer, training_config)

    def get_accuracy(self):
        df = level_encoding(self.training_config)
        total_size = len(df)
        correct = 0
        for _, row in df.iterrows():
            generated_prediction = self.nov_predictor.predict(str(row[NumberOfVehiclesModelValidator.nov_df_config.source_sequence_column]))
            actual_answer = row['ENCODE_CAT']
            if generated_prediction == actual_answer:
                correct += 1
        return correct/total_size
