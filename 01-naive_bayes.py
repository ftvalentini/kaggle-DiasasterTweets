# trains Naive Bayes only using text of tweets

import helpers_strings as hs
import helpers_models as hm

import pandas as pd
import numpy as np

semilla = 1984

#%% read data
datos = hm.read_tagged_data()
X = datos.drop(columns='target')
y = datos['target']

#%% init pipeline
pipeline = hm.Pipeline_NB()

#%% Performance CV
hm.kCV_pipe(pipeline, X, y, seed=semilla, k=5)



# # save mean and std to csv
# pd.DataFrame(scores_a).agg(["mean","std"]).round(4).T.to_csv('output/cv_scores_tfidf.csv')
# #%% Fit on all data and save
# mod = pipe_a.fit(X_texto, y)
# joblib.dump(mod, 'data/working/mod_tfidf.joblib')
