import streamlit as st
import joblib
import re
from nltk.corpus import stopwords

# Preprocessing function (should match your training code)
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load the trained model and vectorizer
model = joblib.load(r"sentiment_model.pkl")
tfidf_vectorizer = joblib.load(r"vectorizer.pkl")

st.title("🌟 Product Review Sentiment Analysis 🌟")
st.header("✍️ Input Your Review")
user_review = st.text_area("Enter your product review:", "")

if st.button("Predict Sentiment"):
    if user_review:
        cleaned_review = clean_text(user_review)
        review_vector = tfidf_vectorizer.transform([cleaned_review])
        prediction = model.predict(review_vector)[0]

        # Map model output to 3 categories
        if prediction in [0, 1]:
            sentiment = "Negative (1-2 stars)"
        elif prediction == 2:
            sentiment = "Neutral (3 stars)"
        elif prediction in [3, 4]:
            sentiment = "Positive (4-5 stars)"
        else:
            sentiment = f"Predicted grade (raw model output): {prediction}"

        st.success(f"The predicted sentiment is: **{sentiment}**")
    else:
        st.warning("Please enter a review before predicting.")