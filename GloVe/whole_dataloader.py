from data_process import read_data,Vocab,subsample,batchify,get_centers_and_contexts,coappearence_computation,batchify2
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import random
import os
import pickle

DATA_DIR = '/home/charon/research/NLP/corpus_dataset/data/ptb'
SEED = 729608
random.seed(SEED)
np.random.seed(SEED)

def load_data_ptb(bs,max_window_size,use_distance_weight=False,mode='train',subsampled=True,load_vocab=True):
    assert mode in ['train','val','test'],'the mode must be in [train,val,test]'
    if mode == 'test':
        file_name = 'ptb.test.txt'
    elif mode == 'train':
        file_name = 'ptb.train.txt'
    else:
        file_name = 'ptb.valid.txt'
    
    text,sentences = read_data(DATA_DIR,file_name)

    if load_vocab:
        vocab = Vocab()
        vocab_dir = './vocab'
        with open(os.path.join(vocab_dir,'token2idx.pkl'),'rb') as f:
            vocab.token2idx = pickle.load(f)
        
        with open(os.path.join(vocab_dir,'idx2token.pkl'),'rb') as f:
            vocab.idx2token = pickle.load(f)
    else:
        vocab = Vocab(sentences,min_freq_threshold=10,special_token_tag=False)
    
    print(f'词表大小为：{len(vocab)}')


    # subsampling
    if subsampled:
        subsampled, counter = subsample(sentences, vocab)
    else:
        subsampled = sentences

    # transfrom token to idx(getitem)
    corpus = [vocab[line] for line in subsampled]

    # generate center-context pair
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)

    # compute the co-appearence matrix
    if use_distance_weight:
        coappearence_matrix,coappearence_matrix_2 = coappearence_computation(len(vocab),all_centers,all_contexts,use_distance_weight=use_distance_weight)
    else:
        coappearence_matrix = coappearence_computation(len(vocab),all_centers,all_contexts,use_distance_weight=use_distance_weight)

    class PTB_DATASET(Dataset):
        def __init__(self, centers, contexts,use_distance_weight,coappearence_matrix):
            assert len(centers) == len(contexts) ,f'({len(centers)},{len(contexts)})'
            self.centers = centers
            self.contexts = []
            for context in contexts:
                self.contexts.append([c[0] for c in context])
            if use_distance_weight:
                self.coappearence_matrix,self.coappearence_matrix_2 = coappearence_matrix
            else:
                self.coappearence_matrix = coappearence_matrix
            self.tag = use_distance_weight

        def __getitem__(self, index):
            center = self.centers[index]
            contexts = self.contexts[index]
            if self.tag:
                label = self.coappearence_matrix_2[center,contexts].tolist()
            else:
                label = self.coappearence_matrix[center,contexts].tolist()
            if not self.tag:
                return (self.centers[index], self.contexts[index],label)
            return (self.centers[index], self.contexts[index],label,self.coappearence_matrix[center,contexts].tolist())

        def __len__(self):
            return len(self.centers)
    
    if not use_distance_weight:
        ptb_dataset = PTB_DATASET(all_centers,all_contexts,use_distance_weight,coappearence_matrix)
        dataloader = DataLoader(
            ptb_dataset, batch_size=bs, shuffle=True,
            collate_fn=batchify, num_workers=4)
    else:
        ptb_dataset = PTB_DATASET(all_centers,all_contexts,use_distance_weight,(coappearence_matrix,coappearence_matrix_2))
        dataloader = DataLoader(
            ptb_dataset, batch_size=bs, shuffle=True,
            collate_fn=batchify2, num_workers=4)
    
    return dataloader,vocab

if __name__ == "__main__":
    names = ['centers', 'contexts', 'masks', 'labels','weights']
    data_iter, vocab = load_data_ptb(512, 2,use_distance_weight=True,mode='train',subsampled=True)
    for batch in data_iter:
        for name, data in zip(names, batch):
            print(name, 'shape:', data.shape)
            print(data[1])
        break
