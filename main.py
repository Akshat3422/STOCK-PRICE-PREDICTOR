import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_stock_data, preprocess_news_data
from train_sentiment_analyzer import train_sentiment_model
from train_price_model import train_price_model
from predicting import get_real_time_news, predict_news_sentiment
import os
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)
# --- Load Preprocessed Data ---
df = joblib.load("data/df.pkl")
financial_news_dataset = joblib.load("data/financial_news_dataset.pkl")

# --- STREAMLIT APP ---

st.title("📊 Stock Price Prediction Dashboard")
st.sidebar.header("🔧 Settings")

# --- Ticker & News ---
ticker = st.sidebar.text_input("Enter Stock Ticker", value="RELIANCE.NS")
news = None

sentiment_model=joblib.load("sentiment_model_bundle.pkl")
le=sentiment_model['label_encoder']
# --- Get Real-Time News ---
if st.sidebar.button("📰 Get Real-Time News"):
    fetched_news = get_real_time_news(ticker)
    if fetched_news:
        st.session_state['news'] = fetched_news
        st.sidebar.success("✅ News fetched successfully!")
        for article in fetched_news:
            st.sidebar.write(article)
    else:
        st.sidebar.error("❌ Failed to fetch news.")
        st.session_state['news'] = None

# --- Predict News Sentiment ---
# --- Predict News Sentiment ---
if st.sidebar.button("💬 Predict News Sentiment"):
    news_articles = st.session_state.get('news', [])

    if news_articles:
        st.sidebar.markdown("### 📰 Sentiment Predictions")
        for i, article in enumerate(news_articles, start=1):
            try:
                sentiment = predict_news_sentiment(article)
                st.sidebar.success(f"{i}. 🧠 Sentiment: {sentiment}")
            except Exception as e:
                st.sidebar.error(f"{i}. ❌ Prediction failed: {e}")
    else:
        st.sidebar.warning("⚠️ No news available for sentiment prediction.")

# --- User Inputs for Features ---
st.subheader("📥 Enter Technical Indicators")
input_close_lag1 = st.number_input("Previous Close (Close_lag1)")
input_return_lag1 = st.number_input("Previous Return (Return_lag1)")
input_rsi = st.number_input("RSI 14")
input_volatility = st.number_input("Volatility 14")
input_volume_lag1 = st.number_input("Previous Volume (Volume_lag1)")
input_obv = st.number_input("On-Balance Volume (OBV)")

# --- Preprocessing Helper ---
def preprocess_user_input(df, scalers, features):
    df = df.copy()
    for feature in features:
        for key in scalers:
            if feature in key:
                df[feature] = scalers[key].transform(df[[feature]])
                break
        else:
            st.warning(f"No scaler found for: {feature}")
    return df

# --- Predict Button ---
if st.button("📤 Submit & Predict"):
    news = get_real_time_news(ticker)
    sentiment = predict_news_sentiment(news[0])

    user_df = pd.DataFrame([{
        'Close_lag1': input_close_lag1,
        'Return_lag1': input_return_lag1,
        'RSI_14': input_rsi,
        'Volatility_14': input_volatility,
        'Volume_lag1': input_volume_lag1,
        'OBV': input_obv,
        'price_sentiment': sentiment
    }])

    price_bundle = joblib.load('best_model_bundle.pkl')
    price_model = price_bundle['model']
    price_features = price_bundle['features']
    scalers = price_bundle['scalers']

    user_df = preprocess_user_input(user_df, scalers, price_features)

    prediction = price_model.predict(user_df[price_features])
    st.success(f"📈 Predicted Stock Price: ₹{prediction[0]:.2f}")
