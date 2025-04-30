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
    sent = nlp(text)
    ents = {x.text for x in sent.ents}
    tokens = []
    
    for w in sent:
        # Explicitly preserve "not" and other negation words
        if w.text.lower() in ["not", "no", "never", "none", "n't"]:
            tokens.append(w.text.lower())  # Preserve original negation
        
        elif w.is_stop or w.is_punct:  # Continue filtering other stopwords
            continue
        
        elif w.text in ents:
            tokens.append(w.text)
        
        else:
            tokens.append(w.lemma_.lower())

    return " ".join(tokens)





class preprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(clean_text).apply(convert_text)
