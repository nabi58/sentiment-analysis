import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Function to predict sentiment
def predict_sentiment(review):
    review_tfidf = vectorizer.transform([review])
    sentiment = model.predict(review_tfidf)
    return sentiment[0]

# Load the model and vectorizer
model = joblib.load('/Users/nabiurrehman/Desktop/SAMPLE TESTING/classifier.pkl')
vectorizer = joblib.load('/Users/nabiurrehman/Desktop/SAMPLE TESTING/vectorizer.pkl')

st.title("Sentiment Analysis App")

review_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):
    if review_input:
        predicted_sentiment = predict_sentiment(review_input)
        st.write(f"Predicted sentiment: {predicted_sentiment}")
    else:
        st.warning("Please enter a review.") 