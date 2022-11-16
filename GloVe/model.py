import torch
import torch.nn as nn

def init_weights(m,mode='zero'):
    if type(m) == nn.Embedding:
        if mode =='xavier':
            nn.init.xavier_normal_(m.weight)
        else:
            nn.init.zeros_(m.weight)

class GloVe(nn.Module):
    def __init__(self,vocab_size,embedding_size) -> None:
        super(GloVe,self).__init__()
        self.context = nn.Embedding(vocab_size,embedding_size)
        self.center = nn.Embedding(vocab_size,embedding_size)
        self.context_bias = nn.Embedding(vocab_size,1)
        self.center_bias = nn.Embedding(vocab_size,1)
        init_weights(self.context,mode='xavier')
        init_weights(self.center,mode='xavier')
        init_weights(self.center_bias)
        init_weights(self.context_bias)

    
    def forward(self,center, all_contexts): # input.shape: B,N,vocab_size
        bs,max_len = all_contexts.shape
        contexts = self.context(all_contexts) # shape (B,N_context,embedding_size)
        centers = self.center(center) # shape (B,1,embedding_size)
        contexts_bias = self.context_bias(all_contexts) # shape (B,N_context,1)
        centers_bias = self.center_bias(center) # shape (B,1,1)
        similarity = centers @ contexts.transpose(1,2) # shape:[B,1,N_context]
        similarity = similarity.reshape(bs,max_len)
        centers_bias = centers_bias.reshape(bs,1)
        contexts_bias = contexts_bias.reshape(bs,max_len)
        output = contexts_bias + similarity + centers_bias
        return output