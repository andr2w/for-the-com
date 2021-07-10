import time 
import utils
import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from sklearn import metrics 
from pytorch_pretrained.optimization import BertAdam
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sn
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params':[p for n,p in param_optimizer  if not any( nd in n for nd in no_decay)], 'weight_decay':0.01},
        {'params':[p for n,p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_deacy':0.0}
    ]

    optimizer = BertAdam(params= optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)

    total_batch = 0
    dev_best_loss = float('inf')
    last_imporve = 0
    flag = False
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime())) 
    model.train()
    loss_trains = []
    loss_devs = []
    acc_trains = []
    acc_devs = []
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}'.format(epoch+1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward(retain_graph=False)
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    imporve = '*'
                    last_imporve = total_batch
                else:
                    imporve = ''
                time_dif = utils.get_time_dif(start_time)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2}, Val Loss:{3:>5.2}, Val Acc:{4:>6.2%}, Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, imporve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch = total_batch + 1
            if total_batch - last_imporve > config.require_improvement:
                print('The model has not improve for a quit long time, stop training')
                flag = True
                break
            loss_trains.append(loss.item())
            loss_devs.append(dev_loss)
            acc_trains.append(train_acc)
            acc_devs.append(dev_acc)
        
        if flag:
            break
    
    x = range(1, len(acc_trains) + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc_trains, 'b', label='Training acc')
    plt.plot(x, acc_devs, 'r', label='Validation acc')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss_trains, 'b', label='Training Loss')
    plt.plot(x, loss_devs, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    
    test(config, model, test_iter)

def evaluate(config, model, dev_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in dev_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total = loss_total + loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data,1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list_str, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(dev_iter), report, confusion

    return acc, loss_total / len(dev_iter)


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss ,test_report, test_confusion = evaluate(config, model, test_iter, test = True)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score")
    print(test_report)
    print("Confusion Maxtrix")
    print(test_confusion)
    time_dif = utils.get_time_dif(start_time)
    print("Using Time",time_dif)
    
    plt.figure()
    plot_confusion_matrix(test_confusion, classes=[0,1], title='Confusion matrix')
    plt.show()
