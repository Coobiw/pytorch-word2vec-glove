import math
import torch
import torch.nn as nn
from whole_dataloader import load_data_ptb
from model import GloVe
from tqdm import trange
import os
import random
import numpy as np
import pickle

SEED = 729608
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0.
        self.avg = 0.
    
    def update(self,n,val,multiply=True):
        self.count += n
        if multiply:
            self.sum += n*val
        else:
            self.sum += val
        self.avg = self.sum / self.count

class Log_MSELoss(nn.Module):
    # Log + MSELoss **with mask**
    def __init__(self,c=10,alpha=0.75):
        super().__init__()
        self.c = c
        self.alpha = alpha

    def forward(self, inputs, target, mask):
        log_target = torch.log(target)
        loss = ((inputs - log_target)**2) * self.weight_func(target) * mask 
        loss = loss.sum(dim=1)
        loss = loss / mask.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss
    
    def weight_func(self,target):
        weight = torch.zeros_like(target)
        weight[target>=self.c] = 1.
        weight[target<self.c] = (target[target<self.c]/self.c)**self.alpha
        return weight


def init_weights(m):
    if type(m) == nn.Embedding:
        nn.init.xavier_uniform_(m.weight)

def train_one_epoch(model,optimizer,device,data_iter,vocab,loss):
    train_loss = AverageMeter()
    for batch in data_iter:
        center,contexts,mask,label = [data.to(device) for data in batch]
        pred = model(center,contexts)
        
        t_loss = loss(pred,label,mask)
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()
        train_loss.update(n=center.shape[0],val=t_loss.item(),multiply=True)
    
    return train_loss.avg

@torch.no_grad()
def get_similar_tokens(model,query_token, k, vocab): 
    '''
    in GloVe, x_{ij} = x_{ji}
    so center_embedding and context_embedding are symmetric in math
    but because the two are not same initialized
    so they have a little difference
    **This difference help the model to be robust**(like one model ensemble method——use different initialization method)
    '''
    model.eval()
    W = model.center.weight.data
    W2 = model.context.weight.data
    W = W + W2
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输入词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')


def main():
    bs, max_window_size = 512, 10
    data_iter, vocab = load_data_ptb(bs, max_window_size,mode='train')
    lr = 2e-3
    num_epochs = 50
    embedding_size = 100
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = GloVe(len(vocab),embedding_size).to(device)
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = Log_MSELoss().to(device)
    
    loss_decription = 0.
    pbar = trange(num_epochs)
    for i in pbar:
        loss_decription = train_one_epoch(model,optimizer,device,data_iter,vocab,loss)
        description = f'Epoch: {i+1}  Train_Loss:{loss_decription:.3f}'
        pbar.set_description(description)
    
    # save model
    save_dir = '/home/charon/research/NLP/GloVe/model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir,'model.pth'),'wb') as f:
        torch.save(model,f)
    
    with open(os.path.join(save_dir,'token2idx.pkl'),'wb') as f:
        pickle.dump(vocab.token2idx,f)
    
    with open(os.path.join(save_dir,'idx2token.pkl'),'wb') as f:
        pickle.dump(vocab.idx2token,f)

if __name__ == "__main__":
    # main()
    model_path = '/home/charon/research/NLP/GloVe/model/model.pth'
    save_dir = '/home/charon/research/NLP/GloVe/model/'
    with open(model_path,'rb') as f:
        model = torch.load(f)

    from data_process import Vocab
    vocab = Vocab()
    with open(os.path.join(save_dir,'token2idx.pkl'),'rb') as f:
        vocab.token2idx = pickle.load(f)
    
    with open(os.path.join(save_dir,'idx2token.pkl'),'rb') as f:
        vocab.idx2token = pickle.load(f)

    get_similar_tokens(model,'chip',10,vocab)

