import torch 
import time 
from datetime import timedelta
import pandas as pd 
from pytorch_pretrained import BertModel, BertTokenizer
from model import Bert
import numpy as np


PAD, CLS = '[PAD]', '[CLS]'



def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_dataset(file_path, config):
    contents = []
    df = pd.read_csv(file_path)
    for index, row in df.iterrows():
        content = row['标题／微博内容']
        '''
        cuz we are doing predict 
        so i just put zeros in the label
        cuz it is better that put things together

        '''
        label = 0
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

class Config():
    def __init__(self, dataset):
          self.model_name = 'Bert'
          self.data = dataset  
          self.class_list = [0, 1]
          self.class_list_str = ['0', '1']
          self.save_path = 'save_dir/' + self.model_name + '.ckpt'
          self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  
          # cut the program if the performance of the model is'not go well after 1000
          self.require_improvement = 1000
          self.num_classes = len(self.class_list)
          self.num_epochs = 20 
          self.batch_size = 128
          self.pad_size = 512
          self.learning_rate = 1e-5
          self.bert_path = 'bert_pretrain'
          # 切词器 tokenizer 
          self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
          self.hidden_size = 768
          self.log_path = dataset + '/log/' + self.model_name




def predict(dataset):
    start_time = time.time() 
    config = Config(dataset)
    data = load_dataset(config.data, config)
    iteror = bulid_iterator(data, config)
    time_dif = get_time_dif(start_time)
    print('Using time:', time_dif)

    model = Bert.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
   
    model.eval() 
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in iteror:
            outputs = model(texts)
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predict)

    return predict_all

if __name__ == '__main__':
    predict = predict('./test_data/en_data.csv')
    print(predict)
    print(len(predict))
    # 8001

