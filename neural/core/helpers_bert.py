import numpy as np
import pandas as pd
import helpers_models as hm
import re

from tensorflow.keras.models import load_model
import tensorflow_hub as hub

def submission_bert(mod_name, tokenizer):
    """
    For BERT, juntar?
    Predicts on test data based on the model selected (hdf5)
    + generate the csv for submission to Kaggle
    """
    best_mod = load_model(mod_name,custom_objects={'KerasLayer':hub.KerasLayer})
    datos_test = hm.read_untagged_data()
    X_test = datos_test['text']
    test_input = bert_encode(X_test, tokenizer, max_len = 160)
    predictions = best_mod.predict(test_input)
    submission = pd.DataFrame(datos_test['id'])
    submission['target'] = predictions.round().astype(int)
    submission.to_csv(re.sub('.h5', '', mod_name) + '_submission.csv', index = False)

# Thanks to https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
