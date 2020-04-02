
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Layer
from tensorflow.keras.optimizers import Adam

# BERT
def build_bert(bert_layer, max_len=512):
    in_id = tf.keras.layers.Input(shape=(max_len,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_len,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_len,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    # Instantiate the custom Bert Layer defined above
    bert_output = bert_layer(bert_inputs)

    # Build the rest of the classifier 
    dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    mod = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return mod


def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model