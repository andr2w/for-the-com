import numpy as np 
import pandas as pd 


def build_dataset(config):
    tokenizer = lambda x: x.split(' ')
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    print('f"Vocab Size: {}"'.format(len(vocab)))

    def load_dataset(path, pad_size=50):
        df = pd.read_csv(path)
        text = df.text.values
        label = df.label.values
        for line in text:

