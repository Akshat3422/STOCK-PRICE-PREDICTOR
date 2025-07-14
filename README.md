# STOCK-PRICE-PREDICTOR
# 📊 Stock Price Prediction Dashboard with Sentiment Analysis

An interactive Streamlit web application that predicts stock prices based on **technical indicators** and **real-time financial news sentiment analysis**.

---

## 🚀 Features

- 📈 Predict stock prices using ML models trained on technical indicators + sentiment
- 📰 Fetch real-time news based on stock ticker (e.g., RELIANCE.NS)
- 💬 Predict sentiment of fetched news headlines
- 📉 Inputs for RSI, OBV, Return Lag, etc.
- 🖥️ Built with Streamlit for clean and fast UI

---

## 🧰 Tech Stack

- **Frontend:** Streamlit
- **ML Libraries:** Scikit-learn, Joblib, Pandas
- **NLP:** TF-IDF + Classifier
- **Data:** Yahoo Finance + Financial News APIs

---

## 📁 Project Structure
project/
│
├── main.py # Main Streamlit app

├── preprocess.py # Preprocessing functions

├── predicting.py # Real-time news + sentiment inference

├── train_sentiment_analyzer.py # Sentiment model training

├── train_price_model.py # Price model training

│
├── data/
│ ├── df.pkl # Stock data

│ └── financial_news_dataset.pkl # News + sentiment dataset
│
├── sentiment_model_bundle.pkl # Trained sentiment model 

├── best_model_bundle.pkl # Trained price model + scalers + features

└── README.md # You're reading this!


---

## 📝 How It Works

1. Enter stock ticker (e.g., RELIANCE.NS)
2. Click "Get Real-Time News" to fetch financial news
3. Click "Predict News Sentiment" to analyze headlines
4. Enter technical indicators like RSI, OBV, etc.
5. Click "Submit & Predict" to see predicted price

---

## 🧠 ML Models

- **Sentiment Model:** TF-IDF Vectorizer + Logistic Regression
- **Price Prediction Model:** Random Forest / Gradient Boosting using:
  - `Close_lag1`, `Return_lag1`, `RSI_14`, `OBV`, `Volatility`, etc.
  - Plus `price_sentiment` from news headlines

---

## ▶️ Running the App

### 1. Clone the repo
```bash
git clone https://github.com/Akshat3422/ STOCK-PRICE-PREDICTOR
cd  STOCK-PRICE-PREDICTOR
```

### 2. Install dependencies
pip install -r requirements.txt

### 3.3. Run the app
First run the training file then run the stremlit one 

