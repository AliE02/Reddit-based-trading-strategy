import argparse
import torch
import json
from omegaconf import DictConfig, OmegaConf
import hydra
from Models.BERTSent import BERTSent
from Models.LSTMSent import LSTMSent
import os

from Preprocessing.Preporcessor import Preprocessor

def load_model(model_cfg, model_path=None):
    """Load the model based on the specified type."""
    # Load configuration specific to the model type
    
    # Instantiate model based on type
    if model_cfg.type == 'bert':
        classifier = BERTSent(model_cfg)
    elif model_cfg.type == 'lstm':
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
    with open(data_path, 'r') as file:
        data = json.loads(file.read())

    texts = [entry['title'] + ' ' + entry['text'] for entry in data]
    return texts

def predict(classifier, texts):
    """Make predictions on a list of texts."""
    predictions = [classifier.predict(text) for text in texts]
    return predictions

@hydra.main(config_path="Configs", config_name="config")
def main(cfg: DictConfig):

    # Load the preprocessor
    preprocessor = load_preprocessor(cfg.preprocessor)
    
    # Load the model
    classifier = load_model(cfg.model)
    

    # Load the data
    texts = load_data(cfg.data.test_data_path)

    # Preprocess the data
    texts = preprocessor.preprocess(texts)
    
    # Perform predictions
    predictions = predict(classifier, texts)

    results  = [
        {
            'text': text,
            'prediction': prediction
        }
        for text, prediction in zip(texts, predictions)
    ]
    
    # save the predictions in Results folder under the name of the model and the date
    with open(f'Results/{cfg.model.type}_predictions.json', 'w') as file:
        json.dump(results, file)

if __name__ == "__main__":
    main()
