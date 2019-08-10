# with deep learning zero to all season 2

# Import Tensorflow >= 1.10 and enable eager execution
import tensorflow as tf

# not needed anymore?!
# tf.enable_eager_execution()

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pprint import pprint
import numpy as np
import os

print(tf.__version__)

sources = [['한미훈련', '불만', '金친서'],
     ['문재인', '내년', '7월'],
     ['서울대', '이어', '고대도'],
     ['아베를', '오만하게', '만든']]
targets = [['공개', '…', '트럼프', '나도', '안', '좋아해'],
           ['끌어내리자', '막말', '난무', '우리공화당', '집회'],
           ['최악동문', '투표', '…', '1위', '장하성', '2위', '이명박'],
           ['박정희', '박근혜의', '원죄']]

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


def preprocess(sequences, max_len, dic, mode='source'):
    assert mode in ['source', 'target'], 'source와 target 중에 선택'

    if mode == 'source':
        # preprocessing for source (use in encoder)
        s_input = list(map(lambda sentence: [dic.get(token) for token in sentence], sequences))
        s_len = list(map(lambda sentence: len(sentence), s_input))
        # source의 길이는 3으로 고정될 것이므로 pad_sequences가 필요없어서 주석처리
        # s_input = pad_sequences(sequences=s_input, maxlen=max_len, padding='post', truncating='post')
        return s_len, s_input

    elif mode == 'target':
        # preprocessing for target (use in decoder)
        # decoder input
        # bos: beginning of sentence eos: end of sentence
        t_input = list(map(lambda sentence: ['<bos>'] + sentence + ['<eos>'], sequences))
        t_input = list(map(lambda sentence: [dic.get(token) for token in sentence], t_input))
        t_len = list(map(lambda sentence: len(sentence), t_input))
        t_input = pad_sequences(sequences=t_input, maxlen=max_len, padding='post', truncating='post') # truncating: 길이가 초과할경우 어디서부터 유효할지

        # decoder output
        t_output = list(map(lambda sentence: sentence + ['<eos>'], sequences))
        t_output = list(map(lambda sentence: [dic.get(token) for token in sentence], t_output))
        t_output = pad_sequences(sequences=t_output, maxlen=max_len, padding='post', truncating='post')

        return t_len, t_input, t_output


# sources preprocess test
s_max_len = 3 # not needed
s_len, s_input = preprocess(sequences=sources,
                            max_len=s_max_len, dic=source2idx, mode='source')
print(s_len, s_input)

# targets preprocess test
t_max_len = 20
t_len, t_input, t_output = preprocess(sequences=targets,
                                      max_len=t_max_len, dic=target2idx, mode='target')
print(t_len, t_input, t_output)
