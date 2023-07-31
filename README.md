# Predicting S&P 500 Motion using Reddit Posts

## Introduction
Predicting stock market trends is a challenging task due to the high volatility of stock markets. In this project, we explore the potential of using Reddit data to forecast the S&P 500 index's behavior. We focus on two relevant subreddit pages, r/WallStreetBets and r/Stocks, and analyze their posts from 2008-01-01 to 2022-12-31.

## Dataset
We collected Reddit posts from r/WallStreetBets and r/Stocks using data dumps as Reddit's API is no longer available. The data was preprocessed to remove irrelevant posts and transformed into a pandas DataFrame. The posts were labeled as "Increased," "Decreased," or "Neutral" based on the change in the S&P 500 index on the following day.

## Methods
We experimented with different models to predict market trends:
1. Baseline: A pre-trained RoBERTa-based sentiment analysis model.
2. Comment-weighted naive model: A model that predicted the daily sentiment based on the weighted average of the comments on each post.
3. Comment-weighted fine-tuned model: A model fine-tuned on our dataset using Huggingface trainer and AdamW optimizer.
4. Unified-signal model: A novel architecture that utilized gradients from all available posts per day and predicted a unified daily sentiment.

## Results
The unified-signal model achieved the best performance in predicting stock market trends, but the results did not surpass random-level choice. More hyper-parameter tuning and computational resources could potentially improve the model's performance.

## How to Use
The repository contains the code for data collection, preprocessing, model training, and evaluation. Follow the instructions in the repository to set up the environment and run the code.

## References
- Vikram, P., et al. (2020). Tweeter Sentiment Analysis for Predicting Stock Market Trends.
- Pagolu, N., et al. (2016). Sentiment Analysis of Financial News Articles for Predicting Stock Trends.
- Sethia, S., et al. (2022). Stock Market Prediction using Financial Articles.
- Bradley, B., et al. (2021). The Influence of Reddit on the Gamestop Short Squeeze.
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
- Loureiro, H., et al. (2022). Fine-tuned RoBERTa Model for Sentiment Analysis on Twitter Data.
