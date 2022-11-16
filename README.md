# PyTorch-Word2Vec & GloVe

This repo reproduces Word2Vec and GloVe to get good word embedding by PyTorch.

The word2vec code refers to the [d2l](https://zh-v2.d2l.ai/). And the GloVe theory knowledge is from there.



## Content

- [Data Preparation](##Data Preparation)
- [GloVe co-occurence](##GloVe co-occurence)

## Data Preparation

Just visit the URL [https://catalog.ldc.upenn.edu/LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42). This corpus is form Wall Street Journal, which has been splitted into transet、valset、testset.



You can use d2l open-source package to get the data, using the python code like following:

```python
import math
import os
import random
import torch
from d2l import torch as d2l

d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

def read_ptb():
    data_dir = d2l.download_extract('ptb')
    print(data_dir)
    
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
print(f'# sentences数: {len(sentences)}')
```

## GloVe co-occurence matrix

There is a memory-allocate problem in this part, **because the space memory-usage of glove co-occurence matrix is $O(N^2)$**, where $N$ refers to the length of our vocabulary. 

In my implementation, the size of vocabulary is 6-7k because I filter the token whose frequency is less than 10(min-freq-threshod in my code). If your vocabulary is much larger, this will cause a memory-allocate problem. **To solve this, you can use hashmap(dict in python such as  collections.Counter class)**