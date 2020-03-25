import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helpers_strings as hs
import helpers_models as hm

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from keras.layers import Embedding
from sklearn.metrics import f1_score, recall_score, precision_score
from keras.models import load_model

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        X_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_predict = (np.asarray(self.model.predict(X_val))).round()
        val_f1 = f1_score(y_val, y_predict)
        val_recall = recall_score(y_val, y_predict)
        val_precision = precision_score(y_val, y_predict)
        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precision)
        print(
            "- val_f1: {0:.4f} - val_rec: {1:.4f} - val_prec: {2:.4f} ".format(
                val_f1, val_recall, val_precision)
            )
        return

class TrainingHistory(Callback):
    """
    records metrics per batch
    """
    def on_train_begin(self, logs={}):
        self.history = {'loss':[],'val_loss':[],'acc':[],'val_acc':[]}
    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))
    def on_epoch_end(self, epoch, logs={}):
        self.history['acc'].append(logs.get('acc'))
        self.history['val_acc'].append(logs.get('val_acc'))
        self.history['val_loss'].append(logs.get('val_loss'))

def plot_history(training_history):
    """
    plot metrics per epoch/batch of NN training
    """
    f, ax = plt.subplots(1, 2, figsize = (12, 5))
    acc = training_history.history['acc']
    val_acc = training_history.history['val_acc']
    loss = training_history.history['loss']
    val_loss = training_history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    batches = range(1, len(loss) + 1)
    # plot 1: acc and val_acc per epoch
    plt.sca(ax[0])
    plt.plot(epochs, acc, label='Training acc')
    plt.plot(epochs, val_acc, color='green', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    # plot 2: loss per batch
    plt.sca(ax[1])
    plt.plot(batches, loss, linewidth=0.5, label='Training loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

def create_seqs(X_train, X_val, vocab_size=None, pad_type='pre', seq_maxlen=100):
    """
    Tokenize and create sequences on training + validation for classification
    """
    X_train = pd.Series([hs.full_clean_text(t) for t in X_train])
    X_val = pd.Series([hs.full_clean_text(t) for t in X_val])
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_padded = pad_sequences(X_train_seq, maxlen=seq_maxlen, padding=pad_type)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_val_padded = pad_sequences(X_val_seq, maxlen=seq_maxlen, padding=pad_type)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, X_train_padded, X_val_padded

def create_embedding_layer(tokenizer, embeddings_dict):
    """
    Create frozen embedding layer from tokenizer and a dictionary of pretrained embs
    """
    vocab_size = len(tokenizer.word_index) + 1
    dim = len(embeddings_dict['man']) # dim of a common word to find dim general
    embedding_matrix = np.zeros((vocab_size, dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    emb_layer = Embedding(input_dim=vocab_size, output_dim=dim
                          , weights=[embedding_matrix], trainable=False)
    return emb_layer

def submission(mod_name, output, tokenizer, param_tokenizer):
    """
    Predicts on test data based on the model selected (hdf5)
    + generate the csv for submission to Kaggle
    """
    best_mod = load_model(mod_name)
    datos_test = hm.read_untagged_data()
    X_test = datos_test['text']
    X_test = pd.Series([hs.full_clean_text(t) for t in X_test])
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    param_tokenizer_pad = dict(
        maxlen=param_tokenizer['seq_maxlen'],
        padding=param_tokenizer['pad_type']
    )
    X_test_padded = pad_sequences(X_test_seq,**param_tokenizer_pad)
    predictions = best_mod.predict_classes(X_test_padded)
    submission = pd.DataFrame(datos_test['id'])
    submission['target'] = predictions
    submission.to_csv(output, index = False)

# class LearningPlot(Callback):
#     """
#     Deprecated porque mucho quilombo / remplazado por PlotLossesCallback
#     """
#     def __init__(self, epochs):
#         self.epochs = epochs
#     def on_train_begin(self, logs={}):
#         self.i = 1
#         self.epoch = []
#         self.acc = []
#         self.val_acc = []
#         self.fig, self.ax = plt.subplots(figsize=(8,5))
#         self.lines, = self.ax.plot([],[])
#         self.ax.set_xlim([0,self.epochs+1])
#         self.ax.set_ylim(top=1.0)
#         # self.fig = plt.figure()
#         # self.ax = self.fig.add_subplot(1,1,1)
#         # plt.ion()
#     def on_epoch_end(self, epoch, logs={}):
#         self.epoch.append(self.i)
#         self.acc.append(logs.get('acc'))
#         self.val_acc.append(logs.get('val_acc'))
#         self.i += 1
#         if len(self.epoch) > 1:
#             # f, ax = plt.subplots(1, 1, figsize=(8, 5))
#             # clear_output(wait=True)
#             self.lines.set_xdata(self.epoch)
#             self.lines.set_ydata(self.acc)
#             self.fig.canvas.draw()
#             self.fig.canvas.flush_events()
#             plt.show()
#             # self.ax.plot(self.epoch, self.acc, label="Training Acc.")
#             # self.ax.plot(self.epoch, self.val_acc, label="Validation Acc.")
#             # self.fig.canvas.draw()
#             # self.fig.canvas.flush_events()
#             # plt.draw()
#             # plt.clf()
