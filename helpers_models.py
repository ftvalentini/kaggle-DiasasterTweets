import helpers_strings as hs

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV


def read_tagged_data():
    """ read and return "train" DF
    """
    train_data = pd.read_csv('data/raw/train.csv')
    return train_data

def read_untagged_data():
    """ read and return "test" DF
    """
    test_data = pd.read_csv('data/raw/test.csv')
    return test_data

class TextSelector(BaseEstimator, TransformerMixin):
    """ selects 'text' from DF in pipeline
    (adds keywords F times to text -- F=0 means no adding)
    """
    def __init__(self, F=0):
        self.F = F
    def fit(self, df, y=None):
        return self
    def transform(self, df):
        F = int(self.F)
        text = df['text']
        keyword = df['keyword'] + ' '
        new_text = np.where(df['keyword'].notnull(), keyword * F + text, text)
        return pd.Series(new_text)

def pipeline_nb():
    """ pipeline to fit NB using only tfidf features from tweets' text
    """
    # sw = hs.get_stopwords()
    text_selector = TextSelector()
    tfidf = TfidfVectorizer(
                preprocessor=hs.full_clean_text
                ,tokenizer=hs.tokeniza
                ,min_df=0.01
                ,ngram_range=(1,4)
                ,stop_words='english'
                ,binary=False
                )
    feat_tfidf = Pipeline([
                        ('selector', text_selector)
                        ,('tfidf', tfidf)
                        ])
    classifier = MultinomialNB(alpha=1)
    pipe = Pipeline([
                ('feat_tfidf', feat_tfidf)
                ,('classifier', classifier)
        ])
    return pipe

def grid_search_kcv_pipe(X, y, pipeline, param_grid, k):
    """
    Run GridSearch with kCV on Pipeline
    """
    grid_search = GridSearchCV(pipeline, param_grid, cv=k
                              ,return_train_score=True, refit=False,
                               scoring={
                                   'acc':'accuracy'
                                   ,'prec':'precision_macro'
                                   ,'rec':'recall_macro'
                                   ,'f1':'f1_macro'
                                }, verbose=1)
    grid_search.fit(X, y)
    return grid_search

def kcv_pipe(modelo, X_data, Y_data, seed, k=5):
    """ perform k CrossValidation on initialised pipeline for classification
    """
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    metrics = {'acc':'accuracy'
               ,'prec':'precision_macro'
               ,'rec':'recall_macro'
               ,'f1':'f1_macro'
               }
    scores = cross_validate(modelo, X_data, Y_data, cv=cv, scoring=metrics
                            ,return_train_score=True)
    return scores
    # return mean and std
    # pd.DataFrame(scores_b).agg(["mean","std"]).round(4).T
