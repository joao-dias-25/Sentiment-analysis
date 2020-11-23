from __future__ import unicode_literals

import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
import numpy as np
import pandas as pd


SPACY_MODEL_NAMES = ["en_core_web_sm", "en_core_web_md", "de_core_news_sm", "pt_core_news_sm"]
DEFAULT_TEXT = "Some random text that I wrote."
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
spacy_model = st.sidebar.selectbox("Model name", SPACY_MODEL_NAMES)
model_load_state = st.sidebar.info(f"Loading model '{spacy_model}'...")
#nlp model
nlp = load_model(spacy_model)
model_load_state.empty()

text = st.text_area("Text to analyze", DEFAULT_TEXT)
with st.echo():
    # tokenizing words
    doc = process_text(spacy_model, text)




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


