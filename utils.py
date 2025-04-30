import numpy as np
import pandas as pd
import regex as re
import joblib
import en_core_web_sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC

nlp = en_core_web_sm.load()
classifier = LinearSVC()

def clean_text(text):
    # Load text into spaCy for tokenization
    doc = nlp(text)
    
    # Preserve negation words and clean text
    negation_words = {"not", "no", "never", "none", "n't"}
    cleaned_words = [token.text for token in doc if token.text.lower() in negation_words or not token.is_punct]
    
    return " ".join(cleaned_words)


	
def convert_text(text):
    doc = nlp(text)
    ents = {ent.text for ent in doc.ents}
    tokens = []

    for token in doc:
        # Ensure key negation words are preserved
        if token.is_stop and token.text.lower() not in {"not", "no", "never", "none", "n't"}:
            continue
        if token.is_punct:
            continue
        if token.text in ents:
            tokens.append(token.text)
        else:
            tokens.append(token.lemma_.lower())

    return " ".join(tokens)




class preprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(clean_text).apply(convert_text)
