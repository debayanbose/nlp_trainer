import json
import os
import click
from transformers import AdamW

import defaults
from model_map import model_config
from utils import *
from config_data import TrainingConfig
from validation import NumberOfVehiclesModelValidator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

@click.command()
@click.option("--lr", type=click.FLOAT, default=defaults.LEARNING_RATE)
@click.option("--epochs", type=click.INT, default=defaults.EPOCHS)
@click.option("--batch-size", type=click.INT, default=defaults.BATCH_SIZE)
@click.option("--pretrained-dir", type=click.Path(exists=True, resolve_path=True))
@click.option("--training_data", type=click.Path(exists=True, resolve_path=True))
@click.option("--checkpoint-dir", type=click.Path(exists=True, resolve_path=True))
@click.option("--validation-csv", type=click.STRING)
@click.option("--n-categories", type=click.INT, default=defaults.N_CATEGORIES)
@click.option("--validate-only", type=click.BOOL, default=False)
def main(pretrained_dir,
         training_data,
         checkpoint_dir,
         lr,
         epochs,
         batch_size,
         validation_csv,
         n_categories,
         validate_only):

         training_config = TrainingConfig(pretrained_model_path = pretrained_dir,
                                          training_data = training_data,
                                          checkpoint_dir = checkpoint_dir,
                                          lr = lr,
                                          epochs = epochs,
                                          batch_size = batch_size,
                                          validation_data = validation_csv,
                                          n_categories = n_categories)
         print(f"program called with params \n{training_config}")
         if not validate_only:
             orchestrate_training(training_config)
         do_validation(training_config)

def orchestrate_training(train_config: TrainingConfig):
    print('Importing model ...')
    pretrained_model_name = train_config.pretrained_model_path.split("/")[-1]
    print(train_config.pretrained_model_path)
    model_info = model_config[pretrained_model_name]
    tokenizer = model_info.tokenizer.from_pretrained(train_config.pretrained_model_path)
    train_dataloader, test_dataloader = train_test_split(train_config, tokenizer, 0.8)
    pretrained_model = model_info.pretrained_model.from_pretrained(train_config.pretrained_model_path,
                                                                   config=model_info.config(num_labels = train_config.n_categories))
    optimizer = AdamW(pretrained_model.parameters())
    trainer = model_info.trainer(pretrained_model, train_config, train_dataloader, test_dataloader, optimizer)
    trainer.train()

def do_validation(training_config: TrainingConfig):
    print("Performing Validation ...")
    validator = NumberOfVehiclesModelValidator(training_config)
    results = {"val_accu": validator.get_accuracy()}
    print(results)
    results_file = training_config.results_path + 'results.json'
    with open(results_file,"w") as fp:
        json.dump(results, fp)
    print(f"results saved tp{results_path}")


if __name__ == '__main__':
    main()
