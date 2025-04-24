import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment App", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and find out if the sentiment is **positive** or **negative**.")

review = st.text_input("Enter Movie Review")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        # Transform the input and make prediction
        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)

        # Show result
        if prediction[0] == 1:
            st.success("The review is **Positive** 😊")
        else:
            st.error("The review is **Negative** 😞")
