import streamlit as st
from transformers import pipeline

classifier = pipeline("text-classification", model="Aishaa11/sentiment-model")

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

st.title("Sentiment Analysis Demo")
st.write("Fine-tuned DistilBERT model - 79% accuracy")
text = st.text_input("Enter a sentence:")

if text:
    result = classifier(text)[0]
    sentiment = label_map.get(result['label'], result['label'])
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Confidence:** {result['score']:.2f}")
