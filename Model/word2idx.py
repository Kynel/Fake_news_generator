from pprint import pprint
import numpy as np
import os
import pickle

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

sources, targets = [], []
with open(os.path.join(ROOT_DIR,'Data_Preprocess','sources.dat'), 'rb') as sources_stream:
    sources = pickle.load(sources_stream)
with open(os.path.join(ROOT_DIR,'Data_Preprocess','targets.dat'), 'rb') as targets_stream:
    targets = pickle.load(targets_stream)

# vocabulary for sources
s_vocab = list(set(sum(sources, [])))
s_vocab.sort()
s_vocab = ['<pad>'] + s_vocab
source2idx = {word: idx for idx, word in enumerate(s_vocab)}
idx2source = {idx: word for idx, word in enumerate(s_vocab)}

pprint(source2idx)

# vocabulary for tagrets
t_vocab = list(set(sum(targets, [])))
t_vocab.sort()
t_vocab = ['<pad>', '<bos>', '<eos>'] + t_vocab
target2idx = {word: idx for idx, word in enumerate(t_vocab)}
idx2target = {idx: word for idx, word in enumerate(t_vocab)}

pprint(target2idx)

print(source2idx.__len__())
print(target2idx.__len__())


with open(os.path.join(os.path.dirname(__file__),'source2idx.dat'), 'wb') as source2idx_stream:
    pickle.dump(source2idx, source2idx_stream)
with open(os.path.join(os.path.dirname(__file__),'idx2source.dat'), 'wb') as idx2source_stream:
    pickle.dump(idx2source, idx2source_stream)
with open(os.path.join(os.path.dirname(__file__),'target2idx.dat'), 'wb') as target2idx_stream:
    pickle.dump(target2idx, target2idx_stream)
with open(os.path.join(os.path.dirname(__file__),'idx2target.dat'), 'wb') as idx2target_stream:
    pickle.dump(idx2target, idx2target_stream)