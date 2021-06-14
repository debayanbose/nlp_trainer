from typing import Dict

import torch
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedModel

from config_data import TrainingConfig
from utils import calculate_accu


class NoVTrainer:
    def __init__(self, model: PreTrainedModel, training_config: TrainingConfig,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 optimizer: Optimizer):
                 self.model = model
                 self.training_config = training_config
                 if torch.cuda.device_count() > 1:
                     print(f"using {torch.cuda.device_count()} GPUs!")
                     self.model = nn.DataParallel(model)
                 self.model.to(training_config.device)
                 self.train_dataloader = train_dataloader
                 self.test_dataloader = test_dataloader
                 self.optimizer = optimizer

    def convert_to_device(self, batch: Dict[str, Tensor]) -> None:
        for key, value in batch.items():
            batch[key] = value.to(self.training_config.device)

    def train_loop(self, epoch):
        self.model.train()
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        for batch in self.train_dataloader:
            logits, tr_loss = self.train_step(batch, tr_loss)
            big_val, big_idx = torch.max(logits, dim =1)
            n_correct += calculate_accu(big_idx, batch['labels'])
            nb_tr_steps += 1
            nb_tr_examples += batch['labels'].size(0)
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss :{loss_step}")
            print(f"Training Accuracy : {accu_step}")
        print(f"Average training Loss for Epoch {epoch}: {tr_loss / nb_tr_steps}")
        print(f"Total training accuracy for epoch {epoch}: {(n_correct * 100)/ nb_tr_examples}%")

    def train_step(self, batch, tr_loss):
        self.optimizer.zero_grad()
        self.convert_to_device(batch)
        outputs = self.model(**batch)

        tr_loss += outputs.loss.mean().item()
        logits = outputs.logits
        outputs.loss.mean().backward()
        self.optimizer.step()
        return logits, tr_loss

    def test_loop(self, epoch):
        self.model.eval()
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        total_loss = 0
        for batch in self.test_dataloader:
            with torch.no_grad():
                logits, total_loss = self.test_step(batch, total_loss)
                big_val, big_idx = torch.max(logits, dim = 1)
                n_correct += calculate_acc(big_idx, batch['labels'])

                nb_tr_steps += 1
                nb_tr_examples += batch['labels'].size(0)
        epoch_loss = total_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        print(f"Validation loss for Epoch {epoch}: {epoch_loss}")
        print(f"Validation Accuracy for Epoch {epoch}: {epoch_accu}")

    def test_step(self, batch, total_loss):
        self.convert_to_device(batch)
        outputs = self.model(**batch)
        total_loss += output.loss.mean().item()
        return outputs.logits, total_loss

    def train(self):
        print("beginning training and test loop")
        for epoch in range(self.training_config.epochs):
            print(f"EPOCH: {epoch}")
            self.train_loop(epoch)
            self.test_loop(epoch)
            if self.training_config.checkpoint_dir is not None:
                if type(self.model) is nn.DataParallel:
                    self.model.module.save_pretrained(self.training_config.checkpoint_dir)
                else:
                    self.model.save_pretrained(self.training_config.checkpoint_dir)
        return self.model
    
