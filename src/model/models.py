from global_settings import MODEL_PATH

import os
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import gensim
import numpy as np
import underthesea as uts


class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned_texts = []
        for text in X:
            cleaned_text = self.clean_text(text)
            cleaned_texts.append(cleaned_text)
        return np.array(cleaned_texts)

    @staticmethod
    def clean_text(content: str):
        tokens = gensim.utils.simple_preprocess(content)
        content = " ".join(tokens)
        # tokenize content
        content = uts.word_tokenize(content, format="text")

        return content


def get_model(model_name: str = "gnb_boosting_clf"):
    with open(os.path.join(MODEL_PATH, f"{model_name}.pkl"), "rb") as f:
        model = pickle.load(f)
    return model


def get_label_encoder():
    with open(os.path.join(MODEL_PATH, "le_.pkl"), "rb") as f:
        model = pickle.load(f)
    return model


def get_pipe():

    with open(os.path.join(MODEL_PATH, f"pipeline.pkl"), "rb") as f:
        model = pickle.load(f)
    return model
