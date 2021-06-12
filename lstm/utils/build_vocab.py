import numpy as np 
import pickle as pkl 
import pandas as pd

def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    train = pd.read_csv(file_path)
    text = train.text.values

    for sentence in text:
        for word in tokenizer(sentence):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


MAXLEN = 10000 # unknown length of the words
emb_dim = 300
UNK, PAD = '<UNK>', '<PAD>'
filename_trimmed_dir = '../../nedata/embedding_SougouNews'

train_dir = '../../nedata/training_data.csv'
vocab_dir = '../../nedata/vocab.pkl'
pretrain_file = '../../wordembedding/sgns.sogou.char'

tokenizer = lambda x: x.split(' ') # tokenize the word
word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAXLEN, min_freq=1)
pkl.dump(word_to_id, open(vocab_dir, 'wb'))


embeddings = np.random.rand(len(word_to_id), emb_dim)
f = open(pretrain_file, 'r', encoding='UTF-8')
for i, line in enumerate(f.readlines()):
    lin = line.strip().split(" ")
    if lin[0] in word_to_id:
        idx = word_to_id[lin[0]]
        emb = [float(x) for x in lin[1:301]]
        embeddings[idx] = np.asarray(emb, dtype='float32')

f.close()
np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)

