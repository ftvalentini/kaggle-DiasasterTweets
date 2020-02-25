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
pipeline = hm.pipeline_nb()

#%% param_grid for GridSearch
parameters = {
    'feat_tfidf__selector__F': [0,1,2,3,4]
    ,'feat_tfidf__tfidf__binary': [True, False]
    ,'feat_tfidf__tfidf__max_df': [0.90, 0.95, 0.99, 1.00]
    ,'feat_tfidf__tfidf__min_df': [0.00, 0.01, 0.02]
    ,'feat_tfidf__tfidf__ngram_range': [(1,4)]
    ,'feat_tfidf__tfidf__use_idf': [True, False]
    ,'classifier__alpha': [0.1, 0.5, 0.8, 1.0, 2.0]
}

#%% GridSearch with K-CV
gs = hm.grid_search_kcv_pipe(X, y, pipeline, parameters, 3)

#%% save results
gs_results = pd.DataFrame(gs.cv_results_)
gs_results.to_csv('output/tables/gridsearch_naivebayes.csv')

# #%% Performance CV
# hm.kCV_pipe(pipeline, X, y, seed=semilla, k=5)
