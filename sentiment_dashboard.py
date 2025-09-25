import streamlit as st
import joblib


# Load the trained model and vectorizer
model = joblib.load(r"sentiment_model.pkl")
tfidf_vectorizer = joblib.load(r"vectorizer.pkl")
# Title of the dashboard
st.title("🌟 Product Review Sentiment Analysis 🌟")
# Input section for user reviews
st.header("✍️ Input Your Review")
user_review = st.text_area("Enter your product review:", "")

# Button to predict sentiment
if st.button("Predict Sentiment"):
    if user_review:
        # Preprocess and vectorize the input review
        review_vector = tfidf_vectorizer.transform([user_review])

        # Predict sentiment
        prediction = model.predict(review_vector)
        
        # Map numerical prediction to sentiment label
        sentiment_map = {0: "Negative (1-2 stars)", 1: "Neutral (3 stars)", 2: "Positive (4-5 stars)"}
        predicted_sentiment = sentiment_map[prediction[0]]
        
        # Display the predicted sentiment
        st.success(f"The predicted sentiment is: **{predicted_sentiment}**")
    else:
        st.warning("Please enter a review before predicting.")