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

## References
1. Kolasani, Sai Vikram. "Predicting stock movement using sentiment analysis of Twitter feed with neural networks." Journal of Data Analysis and Information Processing, 8 (2020): 309-319. [Link](https://www.scirp.org/pdf/jdaip_2020111613521357.pdf)
2. Sethia, Divyasikha et al. "Stock Price Prediction Using News Sentiment Analysis." 2022 2nd International Conference on Intelligent Technologies (CONIT), (2022): 1-6. [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9847747)
3. Liu, Yinhan et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." ArXiv (2019). [Link](https://api.semanticscholar.org/CorpusID:198953378)
4. Loshchilov, Ilya and Frank Hutter. "Decoupled weight decay regularization." arXiv preprint arXiv:1711.05101 (2017). [Link](https://arxiv.org/abs/1711.05101)
5. Steinbacher, Matej. "Predicting Stock Price Movement as an Image Classification Problem." ArXiv (2023). [Link](https://arxiv.org/abs/2303.01111)
6. Pagolu, Venkata Sasank et al. "Sentiment analysis of Twitter data for predicting stock market movements." 2016 international conference on signal processing, communication, power and embedded system (2016): 1345-1350. [Link](https://ieeexplore.ieee.org/abstract/document/7955659?casa_token=Ac0IkivHgjQAAAAA:ovFEhuqknWXIVsgdCqqMthA_oGmRdEdPmq4IyyB5Y1MZx5MwcTIcKP6ftSo_kx2QGDS9V-E4VZE)
7. Loureiro, Daniel et al. "TimeLMs: Diachronic Language Models from Twitter." ArXiv (2022). [Link](https://arxiv.org/abs/2202.03829)
8. He, Pengcheng et al. "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing." ArXiv (2023). [Link](https://arxiv.org/abs/2111.09543)
9. Bradley, Daniel et al. "Place your bets? The market consequences of investment research on Reddit's Wallstreetbets." The Market Consequences of Investment Research on Reddit's Wallstreetbets (March 15, 2021). [Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3806065)
