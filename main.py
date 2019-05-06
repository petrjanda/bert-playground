import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from tqdm import tqdm
from tensorflow.keras import backend as K

import architecture
import bert_prep
import loader

# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = 256

# Reduce logging output.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

train_X, train_y, test_X, test_y = loader.dataset(max_seq_length)

# Convert to features
tokenizer = bert_prep.load_tokenizer(sess, bert_path)
processor = bert_prep.InputExampleProcessor(tokenizer, max_seq_length)

train_X, train_y = bert_prep.InputExamples(train_X, train_y).to_features(processor)
test_X, test_y = bert_prep.InputExamples(test_X, test_y).to_features(processor)

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

model = architecture.build_model(bert_path, max_seq_length)

# Instantiate variables
initialize_vars(sess)

model.fit(
    train_X, train_y,
    validation_data=(test_X, test_y),
    epochs=1,
    batch_size=32
)

model.save('BertModel.h5')

# Clear and load model
model = None
model = architecture.build_model(bert_path, max_seq_length)
initialize_vars(sess)
model.load_weights('BertModel.h5')

post_save_preds = model.predict(test_X)
