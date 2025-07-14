from data_ingestion import df, financial_news_dataset
from preprocess import preprocess_stock_data, preprocess_news_data
from train_price_model import train_price_model
from train_sentiment_analyzer import train_sentiment_model
import joblib
import pandas as pd

# Preprocess
df_clean = preprocess_stock_data(df)
news_clean = preprocess_news_data(financial_news_dataset)

joblib.dump(df_clean, 'data/df.pkl')
joblib.dump(news_clean, 'data/financial_news_dataset.pkl')

# Train models
sentiment_bundle=joblib.load('sentiment_model_bundle.pkl')
price_bundle=joblib.load('best_model_bundle.pkl')

price_model=price_bundle['model']
sentiment_model=sentiment_bundle['model']

price_features=price_bundle['features']
sentiment_features=sentiment_bundle['tfidf']

scalers=price_bundle['scalers']


import requests

def get_real_time_news(ticker):
    api_key = "1u0XY1kuOy0jVyjU8uT3rKUTDOZwvPGLamuWfuQl"
    url = "https://api.marketaux.com/v1/news/all"

    params = {
        "symbols": ticker,
        "language": "en",
        "countries": "in",
        "filter_entities": "true",
        "api_token": api_key
    }

    response = requests.get(url, params=params)

    headlines = []

    if response.status_code == 200:
        data = response.json()
        for article in data.get("data", [])[:20]:  # Top 10 articles
            title = article['title']
            headlines.append(title)  # Collect for model
            # top headlines
        return headlines
    else:
        print("Error:", response.status_code, response.text)
        return None

news= get_real_time_news(ticker="RELIANCE.BSE")


def predict_news_sentiment(news_list):
    # Load components
    sentiment_bundle = joblib.load('sentiment_model_bundle.pkl')
    tfidf = sentiment_bundle['tfidf']
    model = sentiment_bundle['model']
    le = sentiment_bundle['label_encoder']
def predict_news_sentiment(news_text):
    # Load components
    sentiment_bundle = joblib.load('sentiment_model_bundle.pkl')
    tfidf = sentiment_bundle['tfidf']
    model = sentiment_bundle['model']
    le = sentiment_bundle['label_encoder']

    # Predict
    X = tfidf.transform([news_text]).toarray()
    y_pred = model.predict(X)

    return y_pred[0]  # Return numeric label (e.g., 0, 1)



