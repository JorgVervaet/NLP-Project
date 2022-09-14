from statistics import variance
from unittest import result
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
    st.markdown("This is a NLP classification web app")
    variance = st.text_input("Comment", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict(variance)
    st.success('The output is {}'.format(result))


if __name__ == '__main__':
    main()
    