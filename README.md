# Reddit-based-trading-strategy
 FIN-407 course project through which we aim to create a trading strategy based on reddit comments' sentiment analysis

## Introduction
This project aims at creating a trading strategy based on sentiment analysis of Reddit comments.
We decided to focus on the stock market and more specifically the AAPL, GOOG, TSLA and AMZN stocks.

The data will be collected from the subreddits [wallstreetbets](https://www.reddit.com/r/wallstreetbets/), [investing](https://www.reddit.com/r/investing/), [daytrading](https://www.reddit.com/r/Daytrading/) and [stocks](https://www.reddit.com/r/stocks/)  which are known among the finance subreddits.
We will then preprocess the data and perform **sentiment analysis** on the comments, and test different trading strategies with and without the sentiment.

## Redirecting
You can find all of the information on the trading strategies in the notebooks section.
The preprocessing has its own directory `Preprocessing`.
As for the LLM Trading Strategy, the results and prompts can be found in the `prompts` `llm_outputs`
For the sentiment analysis, the scripts that were used are `predict.py` and `text_split.py` scripts.

## Credits
This project was implemented by:
- [Essonni Ali](https://www.linkedin.com/in/ali-essonni-54b2051b1/)
- [Badri Omar](https://www.linkedin.com/in/omar-badri-21942120a/)
- [Sefrioui Karim Mhamed](https://www.linkedin.com/in/mhamed-sefrioui-174146264/)
- [Karime Hamza](https://www.linkedin.com/in/hamza-karime-60095418a/)