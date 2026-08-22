import streamlit as st
from transformers import pipeline

classifier = pipeline("text-classification", model="Aishaa11/sentiment-model")

st.title("Sentiment Analysis Demo")
st.write("Fine-tuned DistilBERT model - 79% accuracy")
text = st.text_input("Enter a sentence:")

if text:
    result = classifier(text)[0]
    st.write(f"**Sentiment:** {result['label']}")
    st.write(f"**Confidence:** {result['score']:.2f}")
