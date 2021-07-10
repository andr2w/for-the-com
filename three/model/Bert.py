import torch
import torch.nn as nn 
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):
    '''
    Config
    '''
    def __init__(self, dataset):
        self.model_name = 'Bert'
        self.train_path = dataset + 'train.csv'
        self.test_path = dataset + 'test.csv'
        self.dev_path = dataset + 'dev.csv'
        self.class_list = [0, 1]
        self.class_list_str = ['0', '1']
        self.save_path = 'save_dir/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # cut the program if the performance of the model is'not go well after 1000
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 20 
        self.batch_size = 8
        self.pad_size = 512
        self.learning_rate = 1e-5
        self.bert_path = 'bert_pretrain'
        # 切词器 tokenizer 
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.log_path = dataset + '/log/' + self.model_name


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
    
    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=True)
        out = self.fc(pooled)
        return out 


        


