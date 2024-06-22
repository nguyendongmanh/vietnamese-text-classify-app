from src.scraping.dantri import get_content
from src.model.models import get_model, get_label_encoder, get_pipe, TextCleaner

import streamlit as st
import numpy as np

st.title("Vietnamese Text Classification")

model_name = None

option = st.selectbox(
    "Let's pick a model for predicting ...",
    ("Gaussian Navie Bayes", "Support Vector Machines"),
    index=None,
    placeholder="Select pre-trained model",
)

if option == "Gaussian Navie Bayes":
    model_name = "boosting_gnb_clf"
elif option == "Support Vector Machines":
    model_name = "svm_clf"


st.write("You selected:", model_name)

with st.spinner("Load model ..."):
    le = get_label_encoder()
    pipeline = get_pipe()

news_url = st.text_input("Link:")

if news_url:
    with st.spinner("Wait for it..."):
        model = get_model(model_name)
        content, title = get_content(news_url)
        x = pipeline.transform(np.array([content]))
        st.write(
            f"Bài báo '{title}' thuộc loại '{le.inverse_transform(model.predict(x))[0]}'"
        )
else:
    st.write("Let's insert the link of article")
