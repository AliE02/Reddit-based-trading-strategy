"""
LSTMSent is a class that inherits from SentimentClassifier
It is a sentiment classifier that uses an LSTM model to predict the sentiment of a given text.
This class is implemented using pytorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from typing import List, Tuple

from Models.SentimentClassifier import SentimentClassifier
from Models.TextDataset import TextDataset, collate_batch

class LSTMSent(SentimentClassifier, nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_layers: int, bidirectional: bool, num_classes: int, batch_size: int, device: str):
        """
        Initializes the LSTMSent model.

        Args:
        embedding_dim: int: The dimension of the word embeddings.
        hidden_dim: int: The dimension of the hidden states of the LSTM.
        num_layers: int: The number of layers in the LSTM.
        bidirectional: bool: Whether the LSTM is bidirectional.
        num_classes: int: The number of classes in the classification task.
        batch_size: int: The batch size to use during training.
        device: str: The device to run the model on.
        """
        super(LSTMSent, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device
        self.label_encoder = LabelEncoder()
        self.model = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.model.to(device)

    def train(self, X: List[str], y: List[str]) -> None:
        """
        Trains the model on the given data.

        Args:
        X: List[str]: A list of texts to train on.
        y: List[str]: A list of labels corresponding to the texts.
        """
        self.label_encoder.fit(y)
        y = self.label_encoder.transform(y)
        dataset = TextDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_batch)
        self.model.train()
        for epoch in range(3):
            for i, (X_batch, y_batch) in enumerate(dataloader):
                self.optimizer.zero_grad()
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
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
            X = torch.tensor([X])
            X = self.vectorize(X)
            X = X.to(self.device)
            output = self.model(X)
            _, predicted = torch.max(output, 1)
            predicted = self.label_encoder.inverse_transform(predicted.cpu().numpy())
            return predicted[0]
        
    def batch_predict(self, X: List[str]) -> List[str]:
        """
        Predicts the sentiment of a batch of texts.

        Args:
        X: List[str]: A list of texts to predict the sentiment of.

        Returns:
        List[str]: A list of predicted sentiments of the texts.
        """
        self.model.eval()
        with torch.no_grad():
            X = [torch.tensor(x) for x in X]
            X = self.vectorize(X)
            X = X.to(self.device)
            output = self.model(X)
            _, predicted = torch.max(output, 1)
            predicted = self.label_encoder.inverse_transform(predicted.cpu().numpy())
            return predicted.tolist()
        
    def evaluate(self, X: List[str], y: List[str]) -> Tuple[float, float]:
        """
        Evaluates the model on the given data.

        Args:
        X: List[str]: A list of texts to evaluate the model on.
        y: List[str]: A list of labels corresponding to the texts.

        Returns:
        Tuple[float, float]: A tuple of the accuracy and f1 score of the model.
        """
        y = self.label_encoder.transform(y)
        y_pred = self.batch_predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        return accuracy, f1
    
    def save_model(self, file_path: str) -> None:
        """
        Saves the model to a file.

        Args:
        file_path: str: The path to save the model to.
        """
        torch.save(self.model.state_dict(), file_path)

    def vectorize(self, X: List[str]) -> torch.Tensor:
        """
        Vectorizes a list of texts.

        Args:
        X: List[str]: A list of texts to vectorize.

        Returns:
        torch.Tensor: A tensor of the vectorized texts.
        """
        X = [torch.tensor(x) for x in X]
        X = pad_sequence(X, batch_first=True)
        return X
