import numpy as np
import os
from pathlib import Path
from collections import Counter
import random
import math
import torch

DATA_DIR = '/home/charon/research/NLP/corpus_dataset/data/ptb'
FILE_LIST = [name for name in os.listdir(DATA_DIR)]
SEED = 729608
random.seed(SEED)
np.random.seed(SEED)

def tokenize(text,mode='word'):
    assert mode in ['word','char'],'please choose the tokeniza mode in [word,char]'
    lines = text.split('\n')
    if mode=='word':
        return [line.split() for line in lines]
    else:
        return [list(line) for line in lines]

def read_data(data_dir,file_name):
    file_path = os.path.join(data_dir,file_name)
    with open(file_path,'r') as f:
        text_data = f.read()
    sentences = tokenize(text_data,mode='word')
    return text_data,sentences

# d2l package has this class. I modify some and re-write by myself.
# __init__,__len__,__getitem__ need to be implemented like the torch.utils.data.Dataset
class Vocab: 
    def __init__(self,input_tokens=None,min_freq_threshold=10,special_token_tag=False):
        if special_token_tag: # whether to use pad,begin of sentance,end of sentence,unknown tokens
            self.pad,self.bos,self.eos,self.unk  = 0,1,2,3
            tokens = ['<pad>','<bos>','<eos>','<unk>'] # ?
            self.special_tokens = ['<pad>','<bos>','<eos>','<unk>']
        else:
            self.unk = 0
            tokens = ['<unk>']
            self.special_tokens = ['<unk>']
        
        # when load, we can just initialze a nearly empty Vocab class
        # then load .pkl file to give the value of token2idx,idx2token
        if input_tokens is None: 
            return 
            
        # hashmap to count the tokens(key:token,value:freq)
        assert len(input_tokens),'0 length is not allowed'
        if isinstance(input_tokens[0],list): 
            input_tokens = [token for sentence in input_tokens for token in sentence if token not in self.special_tokens]
            tokens_freq = Counter(input_tokens)
        else:
            tokens_freq = Counter(input_tokens) 
        tokens_freq = sorted(tokens_freq.items(),key=lambda x:x[0]) # sort the tokens_freq dict by dictionary order
        tokens_freq = sorted(tokens_freq,key=lambda x:x[1],reverse=True) # sort the tokens_freq dict by freq order
        tokens_freq = dict(tokens_freq)
        # print(tokens_freq)
        
        # establish two hashmaps to transform index to token or reverse transform
        self.idx2token = []
        self.token2idx = dict()

        # filter the tokens whose freq is letter than min_freq_threshold
        tokens += list(filter(lambda x:tokens_freq[x]>=min_freq_threshold,tokens_freq.keys()))
        # tokens += [token for token,freq in tokens_freq.items() if freq>=min_freq_threshold]

        # add the token to the two hashmaps
        i = 0 # 0 index is unknown token
        for token in tokens:
            self.idx2token.append(token)
            self.token2idx[token] = i
            i += 1
        assert len(self.idx2token) == len(self.token2idx),(len(self.idx2token),len(self.token2idx))
    
    def __len__(self):
        return len(self.idx2token)
    
    def __getitem__(self,tokens): 
        '''
        input : tokens(single char or list/tuple) 
        output : idx(then torch.nn.Embedding automatically transform to one-hot code)
        '''
        if (not isinstance(tokens,tuple)) and (not isinstance(tokens,list)):
            return self.token2idx.get(tokens,self.unk)
        else:
            # recursive call the __getitem__,until the token is a single char
            # use this strategy, we can process higher dim tensor such as :
            # [[a,b,c,...]](shape:[n1,n2]) or even [[[a,b,c],[d,e,f]]](shape:[n1,n2,n3])
            # the return shape is the same as the input
            return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self,indices):
        '''
            input the indices
            output the corresponding tokens
        '''
        if (not isinstance(indices,tuple)) and (not isinstance(indices,list)):
            return self.idx2token[indices]
        else:
            return [self.to_tokens(index) for index in indices]
        

def subsample(sentences,vocab):
    '''
    subsample the sences to reduce the impace of the much high-frequence but less-message tokens like "the","a","in" and so on
    '''
    # 排除未知词元'<unk>'
    sentences_flatten = [token for line in sentences for token in line if vocab[token] != vocab.unk]
    counter = Counter(sentences_flatten)
    num_tokens = sum(counter.values()) # the sum number of the words/tokens

    def keep_func(token,t=1e-4): # t is a hyper-parameter
        # when freq(token) > t,it will be definitely dropped,else the higher freq,the higher prob to be dropped
        freq = counter[token] / num_tokens
        return (random.uniform(0, 1) > max((1 - math.sqrt(t / freq),0)))

    subsampled = []
    for line in sentences:
        temp = []
        for token in line:
            if token in counter.keys() and keep_func(token):
                temp.append(token)
        subsampled.append(temp)
    return subsampled,counter

def compare_counts(token):
    return (f'"{token}"的数量：'
            f'之前={sum([l.count(token) for l in sentences])}, '
            f'之后={subsampled.count(token)}')

# get the centers and contexts token to do the skip-gram language model
def get_centers_and_contexts(corpus, max_window_size):
    """return the center token and the corresponding context token"""
    centers, contexts = [], []
    for line in corpus: # corpus shape: [num_sentence,num_tokens_per_sentence]
        # if length of sentence is less than 2, cannot compose the center-context pair
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # center_idx = i
            window_size = random.randint(1, max_window_size) # randomly generate the window_size
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size))) # get the context token in the window(both left and right)
            # remove the center token
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

class RandomGenerator:
    '''
    randomly generate samples for use, according to the corresponding sampling-weights
    '''
    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates): # means that there isn't new random sample for use
            # cache the k results sampled randomly
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

def negtaive_sampling(all_contexts,vocab,counter,K):
    '''
    all_contexts: the corresponding contexts of all_center token # shape:(num_center,num_context_per_center)
    vocab: the whole Vocabulary
    counter : token counter
    K : the number of negative samples
    '''
    sampling_weights = [counter[vocab.to_tokens(i)]**(3/4) for i in range(1,len(vocab))]
    random_generator = RandomGenerator(sampling_weights=sampling_weights)
    all_negatives = []

    for context in all_contexts:
        while len(all_negatives) < len(all_contexts): 
            # len(all_contexts) = num_center, needing one positive samples,and K negative samples
            # actually, all the positive samples(real context) share the K negative samples
            # this way, the matrix computation is allowed, so easy to run parallelly
            candidate_list = []
            for _ in range(K):
                candidate = random_generator.draw()
                if candidate in context:
                    # noise tokens can't be in the conext token list
                    continue
                else:
                    candidate_list.append(candidate)
            all_negatives.append(candidate_list)
    
    return all_negatives

def batchify(data):
    '''
    if want to batchify the data, we need:
        - padding the data because the num_context_per_center is not the same
        - generate the mask so that the padding part will not be included in the loss_func computation
        - use a mask-like label to distinguish the positive and the negative,1 -> positive and 0 ->negative
    '''
    max_len = max([len(c)+len(n) for _,c,n in data])
    centers,all_contexts_negatives,all_masks,all_labels = [],[],[],[]
    for i,item in enumerate(data):
        center,context,negative = item
        centers.append(center)
        cur_len = len(context) + len(negative)
        all_contexts_negatives.append(context+negative+[0]*(max_len - cur_len))
        all_masks.append([1]*cur_len + [0]*(max_len-cur_len))
        all_labels.append([1]*len(context) + [0] * (max_len - len(context)))
    
    centers = torch.tensor(centers).reshape(-1,1)
    all_contexts_negatives = torch.tensor(all_contexts_negatives)
    all_masks = torch.tensor(all_masks)
    all_labels = torch.tensor(all_labels).long()
    return centers,all_contexts_negatives,all_masks,all_labels


if __name__ == "__main__" :
    # Read Data
    print(FILE_LIST)
    text,sentences = read_data(DATA_DIR,FILE_LIST[2]) # train.txt
    print(f'# sentences num: {len(sentences)}')

    # Vocab Test
    vocab = Vocab(sentences,10,False)
    print(f'# vocab size: {len(vocab)}')

    # Subsample
    subsampled, counter = subsample(sentences, vocab)
    num_tokens = sum(counter.values())
    num_subsampled = sum(len(sub_sentence) for sub_sentence in subsampled)
    print(f'# num_subsampled: {num_subsampled}')
    # import matplotlib.pyplot as plt
    # color = ['red','pink']
    # plt.figure()
    # plt.bar([0,1],[num_tokens,num_subsampled],color=color)
    # plt.xticks([0,1], ['origin','subsampled'])  # give the x tick
    # plt.grid(True,linestyle=':',color='r',alpha=0.6)
    # plt.savefig('./subsampled.png')
    # print(compare_counts('the'))
    # print(compare_counts('join'))

    # transform token to idx
    corpus = [vocab[line] for line in subsampled]

    # test the center-context pair generator
    test_data = [list(range(7)), list(range(7, 10))]
    print('just_test: ', test_data)
    for center, context in zip(*get_centers_and_contexts(test_data, 2)): # set window_size = 2(left 2 + right 2)
        print('center: ', center, 'its context', context)

    # on real data (PTB dataset)
    all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
    print(f'# num of centers: {len(all_centers)}')
    print(f'# num of center-context pairs: {sum([len(contexts) for contexts in all_contexts])}')

    # test the negative_sampling func
    all_negatives = negtaive_sampling(all_contexts, vocab, counter, 5)
    print(f'# len(all_negatives) : {len(all_negatives)}')

    # test the batchify function
    x_1 = (1, [2, 2], [3, 3, 3, 3])
    x_2 = (1, [2, 2, 2], [3, 3])
    batch = batchify((x_1, x_2))

    names = ['centers', 'contexts_negatives', 'masks', 'labels']
    for name, data in zip(names, batch):
        print(name, '=', data)




