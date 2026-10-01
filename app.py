import streamlit as st
import pickle


# Load trained model
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)


# Page title
st.title("Customer Feedback Sentiment Analysis")

st.write(
    "Enter customer feedback below to identify whether "
    "the sentiment is Positive, Negative, or Neutral."
)


# Text input
feedback = st.text_area(
    "Enter Customer Feedback:"
)


# Prediction button
if st.button("Analyze Sentiment"):

    if feedback.strip() == "":
        st.warning("Please enter some feedback.")

    else:
        prediction = model.predict([feedback])[0]

        st.subheader("Sentiment Result")

        if prediction == "Positive":
            st.success("Positive 😊")

        elif prediction == "Negative":
            st.error("Negative 😞")

        else:
            st.info("Neutral 😐")