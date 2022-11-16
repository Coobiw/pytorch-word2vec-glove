import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self,vocab_size,embedding_size) -> None:
        super(Word2Vec,self).__init__()
        self.context = nn.Embedding(vocab_size,embedding_size)
        self.center = nn.Embedding(vocab_size,embedding_size)
    
    def forward(self,center, contexts_and_negatives): # input.shape: B,N,vocab_size
        contexts = self.context(contexts_and_negatives)
        centers = self.center(center)
        similarity = centers @ contexts.transpose(1,2) # shape:[B,N_center,N_context]
        return similarity