from data_ingestion import df, financial_news_dataset
from preprocess import preprocess_stock_data, preprocess_news_data
from train_price_model import train_price_model
from train_sentiment_analyzer import train_sentiment_model
import joblib
import pandas as pd
import requests

df_clean= preprocess_stock_data(df)
news_clean= preprocess_news_data(financial_news_dataset)
joblib.dump(df_clean, 'data/df.pkl')
joblib.dump(news_clean, 'data/financial_news_dataset.pkl')
# Train models
train_sentiment_model(news_clean)
train_price_model(df_clean)
