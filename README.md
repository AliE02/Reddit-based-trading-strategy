# Reddit-based-trading-strategy
 FIN-407 course project through which we aim to create a trading strategy based on reddit comments' sentiment analysis


## Table of Contents
1. [Introduction](#introduction)
2. [Data Collection](#data-collection)
3. [Data Preprocessing](#data-preprocessing)
4. [Sentiment Analysis](#sentiment-analysis)
5. [Trading Strategy](#trading-strategy)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [References](#references)

## Introduction
This project aims at creating a trading strategy based on sentiment analysis of Reddit comments.
We decided to focus on the stock market and more specifically the SPY ETF.

The data will be collected from the subreddit [wallstreet](https://www.reddit.com/r/wallstreet/), which is known for its high volatility and the impact it has on the stock market.
We will then preprocess the data and perform **sentiment analysis** on the comments, using two different methods: **LSTM** and **LLM Fine-Tuning**.

We will then create a trading strategy based on the daily sentiment of the subreddit and backtest it on historical data.

## Data Collection

## Data Preprocessing
The Data Preprocessing strategy will differ depending on the sentiment analysis method used.

### LSTM
For the simpler LSTM Model, we have to put more emphasis on the preprocessing of the data to get the best results.
The data will be preprocessed as follows:
- **Replace emoticons** with a more meaningful keyword.
- **Decontract** negations and other contractions.
- Removing **Stopwords**.
- Correct the text by **lemmatizing** the words.

### LLM Fine-Tuning
Since we will be using a BERT model pretrained on twitter comments, we assume that the slang and abbreviations are already taken care of.
The preprocessing will be limited to:
- **Tokenization** of the text.
- **Padding** the text to the same length.

## Sentiment Analysis
As mentioned earlier, we will be using two different models for sentiment analysis, the first being an LSTM model and the second a BERT model pretrained not only on text based supervision MLM (Masked Language Modeling) but also on a social objective TwHIN (Twitter Heterogeneous Information Network) which is a large scale heterogeneous network of Twitter users and their tweets.

### LSTM
The LSTM model will be trained on the data collected from the subreddit and will output a sentiment score for each comment.

### LLM Fine-Tuning
The BERT model will be fine-tuned on the data collected from the subreddit, we will add a classification layer on top of the BERT model to output a sentiment score for each comment.


