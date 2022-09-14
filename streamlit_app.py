from tkinter import Image
import numpy
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Load the model
model = joblib.load(open("model.pkl", "rb"))


def welcome():
    return "Welcome All"


def predict(data):
    prediction = model.predict(data)
    print(prediction)
    return prediction

def main():
    st.title("NLP Classification")
    html_temp = ""
    #<div style="background-color:tomato;padding:10px">
    #<h2 style="color:white;text-align:center;">NLP Classification ML App </h2>
    #</div>
    st.markdown(html_temp, unsafe_allow_html=True)
    image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    activities = ["Prediction"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    if choice == 'Prediction':
        st.subheader("Prediction")
        data = st.text_input("Enter Text")
        if st.button("Predict"):
            result = predict(data)
            st.success('The output is {}'.format(result))
    