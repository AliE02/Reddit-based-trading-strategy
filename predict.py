import argparse
import torch
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
from Models.BERTSent import BERTSent
from Models.LSTMSent import LSTMSent
import os
from datetime import datetime
from Preprocessing.Preporcessor import Preprocessor

sentiment_mapping = {
    'negative': -1,
    'neutral': 0,
    'positive': 1

}


def load_model(model_cfg, model_path=None):
    """Load the model based on the specified type."""
    # Load configuration specific to the model type
    
    # Instantiate model based on type
    if model_cfg.type.startswith('bert'):
        classifier = BERTSent(model_cfg)
    elif model_cfg.type.startswith('lstm'):
        classifier = LSTMSent(model_cfg)
    else:
        raise ValueError("Unsupported model type specified.")

    # Load the pre-trained model
    if model_path:
        classifier.load_model(model_path)
    return classifier

def load_preprocessor(preproc_cfg):
    """Load the preprocessor based on the specified configuration."""
    preprocessor = Preprocessor(preproc_cfg)
    return preprocessor

def load_data(data_path):
    """Load data from a file."""
    with open("../../../" + data_path, 'r') as file:
        data = json.loads(file.read())

    #texts = [entry['title'] + ' ' + entry['text'] for entry in data]
    texts = [entry['text'] for entry in data]
    dates = [entry['date'] for entry in data]
    return texts, dates

def load_data_bis(data_path):
    """Load data from a file."""
    data = pd.read_csv("../../../" + data_path, sep=';')
    # remove rows where sentiment is nan
    data = data.dropna(subset=['sentiment'])
    texts = data['text'].tolist()
    sentiments = [sentiment_mapping[sentiment] for sentiment in data['sentiment']]
    return texts, sentiments

def load_data_ter(data_path):
    """Load data from a file."""
    data = pd.read_csv("../../../" + data_path, sep=',')
    # remove rows where sentiment is nan
    data = data.dropna(subset=['Sentiment'])
    texts = data['Sentence'].tolist()
    sentiments = [sentiment_mapping[sentiment] for sentiment in data['Sentiment'].tolist()]
    return texts, sentiments


def predict(classifier, texts):
    """Make predictions on a list of texts."""
    predictions = [classifier.predict(text) for text in texts]
    return predictions

@hydra.main(config_path="Configs", config_name="config")
def main(cfg: DictConfig):

    # Load the preprocessor
    preprocessor = load_preprocessor(cfg.preprocessor.type)
    
    # Load the model
    classifier = load_model(cfg.model)
    

    # Load the data
    texts, dates = load_data(cfg.data.test_data_path)

    # Preprocess the data
    texts = preprocessor.preprocess(texts)

    # Make predictions
    predictions = classifier.batch_predict(texts)
    predictions = [int(pred) for pred in predictions]

    assert len(texts) == len(predictions) # == len(sentiments) # Ensure that the number of texts, sentiments and predictions are the same

    results = [
        {
            'text': text,
            'prediction': prediction,
            'date': date
        }
        for text, prediction, date in zip(texts, predictions, dates)
    ]

    # accuracy, precision and recall and f1-score for each class (negative, neutral, positive), the macro-average and the weighted average
    # y_true = np.array(sentiments)
    # y_pred = np.array(predictions)
    # accuracy = accuracy_score(y_true, y_pred)
    # precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    # macro_average = precision_recall_fscore_support(y_true, y_pred, average='macro')
    # weighted_average = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # create directory if it does not exist
    if not os.path.exists('Results'):
        os.makedirs('Results')

    
    # save the predictions in Results folder under the name of the model and the date
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # write all the metrics in a file in a very well formatted way with labels for each class and metric
    # with open(f'Results/{cfg.model.type}_{now}_metrics.txt', 'w') as file:
    #     file.write(f"Accuracy: {accuracy:.4f}\n\n")
        
    #     # Writing class-specific metrics with labels
    #     for i, (pr, re, f1) in enumerate(zip(precision, recall, f1_score)):
    #         file.write(f"Class {i} Precision: {pr:.4f}\n")
    #         file.write(f"Class {i} Recall: {re:.4f}\n")
    #         file.write(f"Class {i} F1-Score: {f1:.4f}\n\n")
        
    #     # Writing macro and weighted averages
    #     file.write(f"Macro-Average Precision: {macro_average[0]:.4f}\n")
    #     file.write(f"Macro-Average Recall: {macro_average[1]:.4f}\n")
    #     file.write(f"Macro-Average F1-Score: {macro_average[2]:.4f}\n\n")
        
    #     file.write(f"Weighted-Average Precision: {weighted_average[0]:.4f}\n")
    #     file.write(f"Weighted-Average Recall: {weighted_average[1]:.4f}\n")
    #     file.write(f"Weighted-Average F1-Score: {weighted_average[2]:.4f}\n")


    with open(f'Results/{cfg.model.type}_{now}_predictions.json', 'w') as file:
        # write in a well formatted json file
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    main()
