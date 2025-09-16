import streamlit as st
import requests
from scripts import s3

API_URL = "http://127.0.0.1:8000/api/v1/"

headers = {
    'Content-Type' : 'application/json'
}

st.title('ML Model Serving over FastAPI')

model = st.selectbox("Select Model",["Sentiment Classifier"])

if model == "Sentiment Classifier":
    text = st.text_area("Enter your Text")
    user_id = st.text_input("Enter USer ID","email@email.com")

    data = {
        "text":[text],
        "user_id" : user_id
    }

    model_api = "get_sentiment"


if st.button("Predict"):
    with st.spinner("Predicting... Please Wait"):
        response = requests.post(API_URL+model_api,headers=headers,
                                 json=data)
        
        output = response.json()

    st.write(output)
