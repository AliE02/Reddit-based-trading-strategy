"""
BERTSent is a model that uses ahmedrachid/FinancialBERT-Sentiment-Analysis model from huggingface to predict the sentiment of a given text.
it inherits from SentimentClassifier and implements the abstract methods.
We also use OmegaConf and hydra to manage configurations of the model.
The configs are stored in the config.yaml file in the Config directory.
Config directory contains folders:
- dataset: contains the dataset configurations
- model: contains the model configurations
- optimizer: contains the optimizer configurations
- scheduler: contains the scheduler configurations
The configs for the BERT variant are stored under bert.yaml in each of the folders.
make sure to use hydra to initialize the model with the configurations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from typing import List, Tuple
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore

from Models.SentimentClassifier import SentimentClassifier
from Models.TextDataset import TextDataset, collate_batch

hydra_output = hydra.utils.get_original_cwd() + "/Outputs"
cs = ConfigStore.instance()

class BERTSent(SentimentClassifier, nn.Module):
    @hydra.main(config_path="../Config", config_name="bert_config.yaml")
    def __init__(self, cfg: DictConfig):
        """
        Initializes the BERTSent model.

        Args:
        cfg: DictConfig: The configurations for the model.
        """
        super(BERTSent, self).__init__()
        self.cfg = cfg
        self.label_encoder = LabelEncoder()
        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model.bert_model)
        self.model = BertForSequenceClassification.from_pretrained(self.cfg.model.bert_model, num_labels=self.cfg.model.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.optimizer.lr)
        self.model.to(self.cfg.model.device)

    def train(self, X: List[str], y: List[str]) -> None:
        """
        Trains the model on the given data.

        Args:
        X: List[str]: A list of texts to train on.
        y: List[str]: A list of labels corresponding to the texts.
        """
        self.label_encoder.fit(y)
        y = self.label_encoder.transform(y)
        dataset = TextDataset(X, y, self.tokenizer, self.cfg.model.max_length)
        dataloader = DataLoader(dataset, batch_size=self.cfg.model.batch_size, shuffle=True, collate_fn=collate_batch)
        self.model.train()
        for epoch in range(self.cfg.model.epochs):
            for batch in dataloader:
                X, y = batch
                X = X.to(self.cfg.model.device)
                y = y.to(self.cfg.model.device)
                self.optimizer.zero_grad()
                outputs = self.model(X, labels=y)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

    def predict(self, X: str) -> str:
        """
        Predicts the sentiment of a given text.

        Args:
        X: str: The text to predict the sentiment of.

        Returns:
        str: The predicted sentiment of the text.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(X, return_tensors="pt", max_length=self.cfg.model.max_length, truncation=True, padding=True)
            inputs = {k: v.to(self.cfg.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
            return self.label_encoder.inverse_transform([predicted_class])[0]
        
    def batch_predict(self, X: List[str]) -> List[str]:
        """
        Predicts the sentiment of a batch of texts.

        Args:
        X: List[str]: A list of texts to predict the sentiment of.

        Returns:
        List[str]: A list of predicted sentiments of the texts.
        """
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for x in X:
                inputs = self.tokenizer(x, return_tensors="pt", max_length=self.cfg.model.max_length, truncation=True, padding=True)
                inputs = {k: v.to(self.cfg.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
                predictions.append(self.label_encoder.inverse_transform([predicted_class])[0])
        return predictions
    
    def evaluate(self, X: List[str], y: List[str]) -> Tuple[float, float]:
        """
        Evaluates the model on the given data.

        Args:
        X: List[str]: A list of texts to evaluate the model on.
        y: List[str]: A list of labels corresponding to the texts.

        Returns:
        Tuple[float, float]: A tuple of the accuracy and f1 score of the model.
        """
        predictions = self.batch_predict(X)
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average="weighted")
        return accuracy, f1
    
    def save_model(self, file_path: str) -> None:
        """
        Saves the model to a file.

        Args:
        file_path: str: The path to save the model to.
        """
        torch.save(self.model.state_dict(), file_path)

cs.store(name="bert_config", node=BERTSent)