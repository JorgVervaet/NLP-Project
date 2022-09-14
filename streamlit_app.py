from statistics import variance
from unittest import result
import numpy
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# Load the model
model = joblib.load(open("F:/Metaces/Dataset/finalized_model.sav", "rb"))


def welcome():
    return "Welcome All"


def predict(data):
    load_vectorizer = joblib.load(open("F:/Metaces/Dataset/vectorizer.sav", "rb"))
    data = load_vectorizer.transform(data)
    fit_vector = data.fit_transform(data)
    prediction = model.predict(data)
    print(prediction)
    return prediction



def main():
    st.title("NLP Classification")
    st.markdown("This is a NLP classification web app")
    #variance = st.text_input("Comment", "Type Here")
    variance = st.file_uploader("Upload CSV", type=["csv"])
    result = ""




    
    if st.button("Predict"):
        result = predict(result)
    st.success('The output is {}'.format(result))


if __name__ == '__main__':
    main()
    