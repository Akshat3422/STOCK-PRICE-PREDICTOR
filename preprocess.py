import joblib
import pandas as pd
import numpy as np
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words=stopwords.words('english')

def preprocess_stock_data(df):
    df = df.dropna().reset_index(drop=True)
    df = df.iloc[1:].reset_index()
    df.rename(columns={'Price': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.drop('index', axis=1, inplace=True)
    df = df.sort_values(by='date').set_index('date')
    
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Close_lag1'] = df['Close'].shift(1)
    df['log_return'] = np.log(df['Close']).diff()
    df['Return_lag1'] = df['log_return'].shift(1)
    df['Volatility_14'] = df['log_return'].rolling(14).std()
    df['Volume_lag1'] = df['Volume'].shift(1)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['OBV'] = (np.sign(df['Close'].diff()).fillna(0) * df['Volume']).cumsum()
    df['price_sentiment'] = np.sign(df['log_return'])

    df.dropna(inplace=True)
    return df

# Preprocessing the financial news sentiment analysis data
def preprocess_news_data(news_df):
    news_df['Final_Status'] = news_df['Final Status'].str.strip()
    news_df['Full_text'] = news_df['Headline'] + " " + news_df['Full_text']
    news_df["Sliced_Text"] = news_df['Full_text'].str[:1000]

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        return " ".join([word for word in tokens if word not in stop_words])
    
    news_df['Tokenized_Text'] = news_df['Sliced_Text'].apply(clean_text)

    return news_df