import argparse
import torch
import json
from omegaconf import DictConfig, OmegaConf
import hydra
from Models.BERTSent import BERTSent # Assuming this class is defined in BERTSent.py
from Models.LSTMSent import LSTMSent  # Assuming this class is defined in LSTMSent.py

def load_model(model_type, model_path):
    """Load the model based on the specified type."""
    # Load configuration specific to the model type
    cfg = OmegaConf.load(f'Config/model/{model_type}.yaml')
    
    # Instantiate model based on type
    if model_type == 'bert':
        classifier = BERTSent.BERTSent(cfg)
    elif model_type == 'lstm':
        classifier = LSTMSent.LSTMSent(cfg)
    else:
        raise ValueError("Unsupported model type specified.")

    # Load the pre-trained model
    classifier.load_model(model_path)
    return classifier

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

def main():
    parser = argparse.ArgumentParser(description="Load a model and predict on a list of texts.")
    parser.add_argument('model_type', choices=['bert', 'lstm'], help="Type of the model to load (bert or lstm).")
    parser.add_argument('model_path', help="Path to the directory where the model is saved.")
    parser.add_argument('data_path', help="Path to the data file containing the texts to predict on.")
    
    args = parser.parse_args()
    
    # Load the model
    classifier = load_model(args.model_type, args.model_path)

    # Load the data
    texts = load_data(args.data_path)
    
    # Perform predictions
    predictions = predict(classifier, texts)
    
    # Print predictions
    for text, prediction in zip(texts, predictions):
        print(f"Text: {text}\nPrediction: {prediction}")

if __name__ == "__main__":
    main()
