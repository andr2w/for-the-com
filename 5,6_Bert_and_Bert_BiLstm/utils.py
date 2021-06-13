import time 
import pandas as pd 
from datetime import timedelta
import torch


PAD, CLS = '[PAD]', '[CLS]'


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_dataset(file_path, config):
    contents = []
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        content = row['text']
        label = row['label']
        token = config.tokenizer.tokenize(content)
        token = [CLS] + token
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)

        pad_size = config.pad_size

        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids = token_ids + ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
            contents.append((token_ids, int(label), seq_len, mask))

    return contents



def bulid_dataset(config):
    train = load_dataset(config.train_path, config)
    dev = load_dataset(config.dev_path, config)
    test = load_dataset(config.test_path, config)
    
    return train, dev, test 


class DatasetIterator(object):
    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size
        self.dataset = dataset
        self.n_batches = len(dataset) // batch_size
        self.residue = False
        if len(dataset) % self.n_batches != 0:
            self.residue = True 
        self.index = 0 
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([item[0] for item in datas]).to(self.device)
        y = torch.LongTensor([item[1] for item in datas]).to(self.device)

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)

        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size : len(self.dataset)]
            self.index += 1 
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0 
            raise StopIteration

        else:
            batches = self.dataset[self.index * self.batch_size : (self.index + 1) * self.batch_size]
            self.index += 1 
            batches = self._to_tensor(batches)
            return batches 

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def bulid_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter 
