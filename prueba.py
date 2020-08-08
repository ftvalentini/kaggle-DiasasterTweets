import helpers_strings as hs

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression


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

class FeatSelector(BaseEstimator, TransformerMixin):
    """ selects 'features' from DF in pipeline
    """
    def __init__(self, features):
        self.features = features
    def fit(self, df, y=None):
        return self
    def transform(self, df):
        return df[self.features]
    def get_feature_names(self):
        return self.features

def Pipeline_NB(classifier):
    """ pipeline to fit NB using only tfidf features from tweets' text
    """
    sw = hs.get_stopwords()
    feat_selector = FeatSelector(features='text')
    tfidf = TfidfVectorizer(
                preprocessor=hs.full_clean_text
                ,tokenizer=hs.tokenize
                ,min_df=0.01
                ,ngram_range=(1,4)
                ,stop_words=sw
                ,binary=True
                )
    feat_tfidf = Pipeline([
                        ('selector', feat_selector)
                        ,('tfidf', tfidf)
                        ])
    #classifier = MultinomialNB(alpha=1)
    pipe = Pipeline([
                ('feat_tfidf', feat_tfidf)
                ,('classifier', classifier)
        ])
    return pipe

def kCV_pipe(modelo, X_data, Y_data, seed, k=5):
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


import helpers_strings as hs
semilla = 1984
datos = read_tagged_data()
X = datos.drop(columns='target')
y = datos['target']

#%% init pipeline
pipeline = Pipeline_NB(classifier = MultinomialNB(alpha=1))

#%% Performance CV
scores_a = kCV_pipe(pipeline, X, y, seed=semilla, k=5)















def Pipeline_LR():
    """ pipeline to fit NB using only tfidf features from tweets' text
    """
    sw = hs.get_stopwords()
    feat_selector = FeatSelector(features='text')
    tfidf = TfidfVectorizer(
                preprocessor=hs.full_clean_text
                ,tokenizer=hs.tokenize
                ,min_df=0.01
                ,ngram_range=(1,4)
                ,stop_words=sw
                ,binary=True
                )
    feat_tfidf = Pipeline([
                        ('selector', feat_selector)
                        ,('tfidf', tfidf)
                        ])
    classifier = LogisticRegression(random_state=8, penalty='l2')
    pipe = Pipeline([
                ('feat_tfidf', feat_tfidf)
                ,('classifier', classifier)
        ])
    return pipe

def kCV_pipe(modelo, X_data, Y_data, seed, k=5):
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


 

