import torch
import torch.nn as nn 
from pytorch_pretrained import BertModel, BertTokenizer

class Config(object):
    '''
    Config
    '''
    def __init__(self, dataset):
        self.model_name = 'BertRNN'
        self.train_path = dataset + 'train.csv'
        self.test_path = dataset + 'test.csv'
        self.dev_path = dataset + 'dev.csv'
        self.class_list = [0, 1]
        self.save_path = 'save_dir/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        # cut the program if the performance of the model is'not go well after 1000
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 20 
        self.batch_size = 8
        self.pad_size = 512
        self.learning_rate = 1e-5
        self.bert_path = './bert_pretrain'
        # 切词器 tokenizer 
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

        # RNN 
        self.rnn_hidden = 256
        self.num_layers = 2
        self.dropout = 0.5
        self.log_path = dataset + '/log/' + self.model_name
        self.class_list_str = ['0', '1']


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers, batch_first=True, dropout=config.dropout, bidirectional=True)

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.rnn_hidden *2, config.num_classes)
    
    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask = mask, output_all_encoded_layers = False)
        out, _ = self.lstm(encoder_out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out 

