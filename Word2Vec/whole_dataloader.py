from data_process import read_data,Vocab,subsample,negtaive_sampling,batchify,get_centers_and_contexts
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import random

DATA_DIR = '/home/charon/research/NLP/corpus_dataset/data/ptb'
SEED = 729608
random.seed(SEED)
np.random.seed(SEED)

def load_data_ptb(bs,max_window_size,num_noise_samples,mode='train'):
    assert mode in ['train','val','test'],'the mode must be in [train,val,test]'
    if mode == 'test':
        file_name = 'ptb.test.txt'
    elif mode == 'train':
        file_name = 'ptb.train.txt'
    else:
        file_name = 'ptb.valid.txt'
    
    text,sentences = read_data(DATA_DIR,file_name)

    vocab = Vocab(sentences,min_freq_threshold=10,special_token_tag=False)

    # subsampling 
    subsampled, counter = subsample(sentences, vocab)

    # transfrom token to idx(getitem)
    corpus = [vocab[line] for line in subsampled]

    # generate center-context pair
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)

    # negative sampling
    all_negatives = negtaive_sampling(all_contexts, vocab, counter, num_noise_samples)

    class PTB_DATASET(Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives),f'({len(centers)},{len(contexts)},{len(negatives)})'
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)
    
    ptb_dataset = PTB_DATASET(all_centers,all_contexts,all_negatives)
    dataloader = DataLoader(
        ptb_dataset, batch_size=bs, shuffle=True,
        collate_fn=batchify, num_workers=4)
    
    return dataloader,vocab

if __name__ == "__main__":
    names = ['centers', 'contexts_negatives', 'masks', 'labels']
    data_iter, vocab = load_data_ptb(512, 5, 5,mode='train')
    for batch in data_iter:
        for name, data in zip(names, batch):
            print(name, 'shape:', data.shape)
            print(data[0])
        break
