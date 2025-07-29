import streamlit as st
import pickle
from text_cleaner import TextCleaner  # Assuming your preprocessing is here

# Load the saved model


@st.cache_resource
def load_model():
    with open("model/spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


pipeline = load_model()

# Streamlit UI
st.title("📩 Spam Message Detector")
st.write("Enter a message below to check if it's spam:")

# User input
user_input = st.text_area("Your message", "")

# Predict and display
if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        prediction = pipeline.predict([user_input])[0]
        if prediction == 1:
            st.error("⚠️ This message is **SPAM**.")
        else:
            st.success("✅ This message is **NOT spam**.")
