from __future__ import unicode_literals

import streamlit as st
import spacy
from spacy import displacy
import numpy as np
import pandas as pd


# Loading TSV file
df_amazon = pd.read_csv ("datasets/amazon_alexa.tsv", sep="\t")

st.write(df_amazon)
st.write(df_amazon.shape)

SPACY_MODEL_NAMES = ["en_core_web_sm", "en_core_web_md", "de_core_news_sm", "pt_core_news_sm"]
DEFAULT_TEXT = 1
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)


st.sidebar.title("Sentiment analyzer")

# loading the language model
spacy_model = st.sidebar.selectbox("Model language name", SPACY_MODEL_NAMES)
model_load_state = st.sidebar.info(f"Loading model '{spacy_model}'...")
#nlp model
nlp = load_model(spacy_model)
model_load_state.empty()

numb = st.number_input("select review number", DEFAULT_TEXT)

# tokenizing words
doc = process_text(spacy_model, df_amazon.verified_reviews[numb])

if "parser" in nlp.pipe_names:
    st.header("Dependency Parse & Part-of-speech tags")
    st.sidebar.header("Dependency Parse")
    split_sents = st.sidebar.checkbox("Split sentences", value=True)
    collapse_punct = st.sidebar.checkbox("Collapse punctuation", value=True)
    collapse_phrases = st.sidebar.checkbox("Collapse phrases")
    compact = st.sidebar.checkbox("Compact mode")
    options = {
        "collapse_punct": collapse_punct,
        "collapse_phrases": collapse_phrases,
        "compact": compact,
    }
    #displacy.render(doc, style = "dep",jupyter = True)

if "ner" in nlp.pipe_names:
    st.header("Text with Named Entities")
    st.sidebar.header("Named Entities")
    label_set = nlp.get_pipe("ner").labels
    labels = st.sidebar.multiselect(
        "Entity labels", options=label_set, default=list(label_set)
    )
    html = displacy.render(doc, style="ent", options={"ents": labels})
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)



st.header("Token attributes")

if st.button("Show token attributes"):
    attrs = [
        "idx",
        "text",
        "lemma_",
        "pos_",
        "tag_",
        "dep_",
        "head",
        "ent_type_",
        "ent_iob_",
        "shape_",
        "is_alpha",
        "is_ascii",
        "is_digit",
        "is_punct",
        "like_num",
    ]
    data = [[str(getattr(token, attr)) for attr in attrs] for token in doc]
    df = pd.DataFrame(data, columns=attrs)
    st.dataframe(df)


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
#nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
#parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = process_text(spacy_model,sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

from sklearn.model_selection import train_test_split

X = df_amazon['verified_reviews'] # the features we want to analyze
ylabels = df_amazon['feedback'] # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train,y_train)

from sklearn import metrics
# Predicting with a test dataset
predicted = pipe.predict(X_test)

st.write(type(X_test))
st.write(pd.concat([X_test.reset_index(drop=True), pd.Series(predicted).reset_index(drop=True)], axis=1))

# Model Accuracy
st.write()
st.write("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
st.write("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
st.write("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))



