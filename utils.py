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
    # Preserve negation words before processing
    negation_words = ["not", "no", "never", "none", "n't"]
    
    # Tokenize the text
    words = text.split()

    # Reduce multiple spaces and newlines
    text = re.sub(r'(\s\s+|\n\n+)', r'\1', text)

    # Remove double quotes
    text = re.sub(r'"', '', text)

    # Ensure negations are not accidentally removed
    cleaned_words = [w for w in words if w.lower() in negation_words or not re.match(r'^\W+$', w)]
    
    return " ".join(cleaned_words)

	
def convert_text(text):
    sent = nlp(text)
    ents = {x.text: x for x in sent.ents}
    tokens = []
    
    for w in sent:
        # Explicitly preserve "not" and other key negation terms
        if w.is_stop and w.text.lower() not in ["not", "no", "never", "none"]:
            continue
        if w.is_punct:
            continue
        if w.text in ents:
            tokens.append(w.text)
        else:
            tokens.append(w.lemma_.lower())

    text = ' '.join(tokens)
    return text



class preprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(clean_text).apply(convert_text)
