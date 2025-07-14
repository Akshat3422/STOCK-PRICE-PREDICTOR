import os
import pandas as pd
import joblib
import yfinance as yf
# This script downloads stock price data for Reliance Industries Limited from Yahoo Finance
# Create a directory for data if it doesn't exist
os.makedirs("data",exist_ok=True)


# Downloading stock price data for Reliance Industries Limited
data=yf.download("RELIANCE.NS",start="2000-01-01", end="2025-01-01")
data.to_csv("data/stock_prices.csv")

df=pd.read_csv("data/stock_prices.csv")

# Now i am using the another dataset for financial_news sentiment_analysis
financial_news_dataset= pd.read_csv("C:\\Users\\user\\Desktop\\final_projects\\Finance\\Stock_Price_Prediction\\Fin_Cleaned.csv")





