import time 
import torch
import numpy as np
import lstm

def main():
    dataset = '../nedata/'
    embedding= '../nedata/embedding_SougouNews.npz'
    config = lstm.Config(dataset, embedding)

    # make sure every res is the same
    np.random.seed(66)
    torch.manual_seed(66)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

   


if __name__ == '__main__':
    main() 
