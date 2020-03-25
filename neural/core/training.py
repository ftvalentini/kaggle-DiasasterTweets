# TODO: lograr reproducibilidad cuando se entrena la red
import numpy as np
from neural.core import helpers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from livelossplot.keras import PlotLossesCallback

def train_nn(model, X_train, y_train, X_val, y_val
            ,model_id
            ,epochs, batch_size=32, early_stopping_n=99999, decay_factor=0.0
            ,decay_patience_n=5
            ,verbose=2):
    """
    train a compiled keras NN
    -- if early_stopping_n > 0: stop after n epochs without ipmrovement
            early_stopping_n arbitrarily high --> no early stopping
    -- if decay_factor > 0:
            reduce LR when a metric does not improve for decay_patience_n epochs
            new_lr = lr * decay_factor (?)
    -- model_id: name to save best model as h5
    """
    mod = model
    # callbacks
    early_stopper = EarlyStopping(monitor='val_acc'
                                  , patience=early_stopping_n, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=decay_factor
                                  ,patience=decay_patience_n
                                  ,min_lr=0.0, cooldown=1, verbose=1)
    mcp_save = ModelCheckpoint(model_id + '_best.hdf5', save_best_only=True
                                , monitor='val_acc', mode='max', verbose=1)
    history = helpers.TrainingHistory()
    metrics = helpers.Metrics()
    # este sirve para hacer un plot en vivo -- pero anula el resto del output!
    # learning_plot = PlotLossesCallback()
    # train
    mod.fit(X_train, y_train, batch_size=batch_size, epochs=epochs
            ,verbose=verbose, validation_data=(X_val, y_val)
            ,callbacks=[early_stopper, reduce_lr, mcp_save, history, metrics])
    helpers.plot_history(history)
    return mod

def full_nn(function_nn, X_train, y_train, X_val, y_val
            ,model_id, epochs, batch_size=32, early_stopping_n=None
            ,decay_factor=1, decay_patience_n=5, sanity_check_n=None
            ,vocab_size=None, pad_type='pre', seq_maxlen=100
            ,verbose=2
            ,**kwargs_nn
            ):
    """
    Tokenize and create sequences + fit NN
    -- if sanity_check_n = n --> fits only n random observations
    """
    tokenizer, X_train_padded, X_val_padded = helpers.create_seqs(
                X_train, X_val, vocab_size, pad_type, seq_maxlen
                )
    if sanity_check_n:
        n = sanity_check_n
        rng = np.random.RandomState(n)
        i_sample = rng.randint(X_train_padded.shape[0], size=n)
        X_train_padded = X_train_padded[i_sample,:]
        y_train = y_train.iloc[i_sample]
    mod = function_nn(tokenizer, **kwargs_nn)
    trained_mod = train_nn(
                    mod, X_train_padded, y_train, X_val_padded, y_val
                    , model_id=model_id
                    , epochs=epochs
                    , batch_size=batch_size
                    , early_stopping_n=early_stopping_n
                    , decay_factor=decay_factor
                    , decay_patience_n=decay_patience_n
                    , verbose=verbose
                    )
    return trained_mod
