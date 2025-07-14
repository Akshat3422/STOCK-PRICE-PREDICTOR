import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_sentiment_model(news_df):
    tfidf=TfidfVectorizer(max_features=1000)
    X=tfidf.fit_transform(news_df['Tokenized_Text']).toarray()
    y=news_df['Final_Status']
    le=LabelEncoder()
    y=le.fit_transform(y) #0 for negative and 1 for positive
    # Split the data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train a simple model for sentiment analysis

    model=XGBClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy=model.score(X_test,y_test)
    # Save the sentiment analysis model and the label encoder
    print(f"Sentiment Analysis Model Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    joblib.dump({'model': model, 'tfidf': tfidf, 'label_encoder': le}, 'sentiment_model_bundle.pkl')
    return model.score(X_test, y_test)