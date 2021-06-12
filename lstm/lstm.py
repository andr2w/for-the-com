import torch
import torch.nn as nn
import numpy as np


class Config():

    def __init__(self, dataset, embedding):
        self.model_name = 'LSTM'
        self.training_path = dataset + 'training_data.csv'
        self.test_path = dataset + 'test_data.csv'
        self.dev_path = dataset + 'dev_data.csv'

        self.class_list = [0, 1]
        self.vocab_path = dataset + 'vocab.pkl'
        self.save_path = 'saved_dir/' + self.model_name + '.ckpt'
        self.embedding_pretrained = torch.tensor(np.load(embedding)['embeddings'].astype('float32'))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.n_vocab = 0
        self.num_epochs = 10
        self.batch_size = 128
        self.pad_size = 50
        self.learning_rate = 1e-5
        self.embed = 300
        self.hidden_size = 128
        self.num_layers = 2

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size*2, config.num_classes)


    def forward(self, x):
        x, _ = x
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out
