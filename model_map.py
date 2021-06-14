from transformers import DistilBertForSequenceClassification, DistilBertConfig,DistilBertTokenizerFast,
                         XLNetForSequenceClassification, XLNetConfig, XLNetTokenizerFast

from config_model import ModelInfo
from trainers import NovTrainer

model_config = {

'xlnet-large-cased': ModelInfo(pretrained_model = XLNetForSequenceClassification,
                               config=XLNetConfig,
                               tokenizer=XLNetTokenizerFast,
                               trainer=NovTrainer),

'distilbert-base-uncased': ModelInfo(pretrained_model = DistilBertForSequenceClassification,
                               config=DistilBertConfig,
                               tokenizer=DistilBertTokenizerFast,
                               trainer=NovTrainer)
}
