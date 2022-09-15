from statistics import variance
from turtle import done
from unittest import result
import numpy
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
import nltk
import string

# Load the model
model = joblib.load(open("", "rb"))
load_vectorizer = joblib.load(open("", "rb"))

french_stopwords = nltk.corpus.stopwords.words('french')
mots = set(line.strip() for line in open('dictionary', encoding="utf8"))
lemmatizer = FrenchLefffLemmatizer()

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

def welcome():
    return "Welcome All"


def predict(data):
    prediction = model.predict(data)
    return prediction

def French_Preprocess_listofSentence(listofSentence):
 preprocess_list = []
 for sentence in listofSentence :
  sentence_w_punct = "".join([i.lower() for i in sentence if i not in string.punctuation])
  sentence_w_num = ''.join(i for i in sentence_w_punct if not i.isdigit())
  tokenize_sentence = nltk.tokenize.word_tokenize(sentence_w_num)
  words_w_stopwords = [i for i in tokenize_sentence if i not in french_stopwords]
  words_lemmatize = (lemmatizer.lemmatize(w) for w in words_w_stopwords)
  sentence_clean = ' '.join(w for w in words_lemmatize if w.lower() in mots or not w.isalpha())
  preprocess_list.append(sentence_clean)
 return preprocess_list


def main():
    st.title("NLP Classification")
    st.markdown("This is a NLP classification web app")
    variance = st.file_uploader("Upload", type=["json", "csv"])
    
    result = ""
    
    if variance is not None:
        if variance.type == "json":
            dataframe = pd.read_json(variance)
                       
        else:
            dataframe = pd.read_csv(variance)
        dataframe.drop_duplicates()
        if "culture" in dataframe.columns:
             dataframe.drop(dataframe[(dataframe["culture"] == "nl-nl")].index, inplace = True)
    
        french_preprocess_list = French_Preprocess_listofSentence(dataframe["text"])
        save_csv = pd.DataFrame(french_preprocess_list)
        save_csv = pd.concat([dataframe["label"], save_csv], axis=1)
        save_csv = save_csv.rename(columns={0: "text"})
        save_csv = save_csv.dropna()
        #dataframe = pd.read_json(variance)
        tokenize = load_vectorizer.transform(save_csv['text'])  
        st.write(dataframe)

    
    if st.button("Predict"):
        with open("", "r") as f:
            label_dictionary = json.load(f)
            #print(label_dictionary)
            result = predict(tokenize)
            for predicted in result:
                st.write("  - Predicted as: '{}'".format(label_dictionary[str(predicted)]))
    #st.success('The output is {}'.format(result))
    

if __name__ == '__main__':
    main()
    