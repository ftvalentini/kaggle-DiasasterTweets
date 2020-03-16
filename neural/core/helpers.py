import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score
# from keras.callbacks import K
# from IPython.display import clear_output

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        X_val = self.validation_data[0]
        y_val = self.validation_data[1]
        y_predict = (np.asarray(self.model.predict(X_val))).round()
        # y_val = np.argmax(y_val, axis=1)
        # y_predict = np.argmax(y_predict, axis=1)
        val_f1 = f1_score(y_val, y_predict)
        val_recall = recall_score(y_val, y_predict)
        val_precision = precision_score(y_val, y_predict)
        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precision)
        print(
            "- val_f1: {0:.4f} - val_prec: {1:.4f} - val_rec: {2:.4f} ".format(
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
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_train_padded = pad_sequences(X_train_seq, maxlen=seq_maxlen, padding=pad_type)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_val_padded = pad_sequences(X_val_seq, maxlen=seq_maxlen, padding=pad_type)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, X_train_padded, X_val_padded

class LearningPlot(Callback):
    """
    Deprecated porque mucho quilombo / remplazado por PlotLossesCallback
    """
    def __init__(self, epochs):
        self.epochs = epochs
    def on_train_begin(self, logs={}):
        self.i = 1
        self.epoch = []
        self.acc = []
        self.val_acc = []
        self.fig, self.ax = plt.subplots(figsize=(8,5))
        self.lines, = self.ax.plot([],[])
        self.ax.set_xlim([0,self.epochs+1])
        self.ax.set_ylim(top=1.0)
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(1,1,1)
        # plt.ion()
    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(self.i)
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        if len(self.epoch) > 1:
            # f, ax = plt.subplots(1, 1, figsize=(8, 5))
            # clear_output(wait=True)
            self.lines.set_xdata(self.epoch)
            self.lines.set_ydata(self.acc)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.show()
            # self.ax.plot(self.epoch, self.acc, label="Training Acc.")
            # self.ax.plot(self.epoch, self.val_acc, label="Validation Acc.")
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
            # plt.draw()
            # plt.clf()