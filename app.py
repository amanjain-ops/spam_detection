import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import regex as re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

ss = SnowballStemmer('english')


nltk.download("punkt")
nltk.download("stopwords")

def text_transformation(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))  
    
    text_lower = text.lower()  
    text_list =  text_lower.split()
    
    return " ".join(ps.stem(word) for word in text_list if word not in stop_words)





tfidf = pickle.load(open('Models/svcVector.pkl', 'rb'))
mnbmodel = pickle.load(open('Models/mnbmodel.pkl', 'rb'))

st.title("Text Spam Classifier")

msg = st.text_area("Enter the message")

if st.button("Proceed"):
    txt = text_transformation(msg)
    txt_vect = tfidf.transform([txt]).toarray()
    pred = mnbmodel.predict(txt_vect)

    print(pred)

    if pred == 1:
        st.write("Spam")
    else:
        st.write("Not Spam")

