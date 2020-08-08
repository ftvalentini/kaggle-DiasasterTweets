# trains Naive Bayes only using text of tweets

%reload_ext autoreload
%autoreload 2

import helpers_strings as hs
import helpers_models_fb as hm

import pandas as pd
import numpy as np

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

semilla = 1984

#%% read data
datos = hm.read_tagged_data()
X = datos.drop(columns='target')
y = datos['target']

#%% init Naive Bayes pipeline
NB = {'classifier': MultinomialNB(alpha=1)}
pipeline_nb = hm.Pipeline_FE_Model(param = NB)

#%% Performance CV
scores_nb = hm.kCV_pipe(pipeline_nb, X, y, seed=semilla, k=5)

#%%
pd.DataFrame(scores_nb).agg(["mean","std"]).round(4)


# # save mean and std to csv
# pd.DataFrame(scores_a).agg(["mean","std"]).round(4).T.to_csv('output/cv_scores_tfidf.csv')
# #%% Fit on all data and save
# mod = pipe_a.fit(X_texto, y)
# joblib.dump(mod, 'data/working/mod_tfidf.joblib')

#%% init Logistic pipeline
LR = {'classifier': LogisticRegression(random_state= semilla, penalty='l2')}
pipeline_lr = hm.Pipeline_FE_Model(param =LR)
#%% Performance CV
scores_lr = hm.kCV_pipe(pipeline_lr, X, y, seed=semilla, k=5)

# %%
pd.DataFrame(scores_lr).agg(["mean","std"]).round(4)
