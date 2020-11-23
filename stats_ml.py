from __future__ import unicode_literals

import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
import numpy as np
import pandas as pd


SPACY_MODEL_NAMES = ["en_core_web_sm", "en_core_web_md", "de_core_news_sm", "pt_core_news_sm"]
DEFAULT_TEXT = """New York City on Tuesday declared a public health emergency and ordered mandatory measles vaccinations amid an outbreak, becoming the latest national flash point over refusals to inoculate against dangerous diseases.

At least 285 people have contracted measles in the city since September, mostly in Brooklyn’s Williamsburg neighborhood. The order covers four Zip codes there, Mayor Bill de Blasio (D) said Tuesday.

The mandate orders all unvaccinated people in the area, including a concentration of Orthodox Jews, to receive inoculations, including for children as young as 6 months old. Anyone who resists could be fined up to $1,000."""
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


