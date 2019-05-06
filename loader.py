import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []

    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))

    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    print(directory)
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    print('=> get file')
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz", 
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
        # extract=True
    )

    print('=> load train')
    train_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "train"))

    print('=> load test')
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))

    return train_df, test_df

def dataset(max_seq_length):
    train_df, test_df = download_and_load_datasets()

    # Create datasets (Only take up to max_seq_length words for memory)
    train_text = train_df['sentence'].tolist()
    train_text = [' '.join(t.split()[0:max_seq_length]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_df['polarity'].tolist()

    test_text = test_df['sentence'].tolist()
    test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = test_df['polarity'].tolist()

    return train_text, train_label, test_text, test_label