# with deep learning zero to all season 2

# Import Tensorflow >= 1.10 and enable eager execution
import tensorflow as tf

# not needed anymore?!
# tf.enable_eager_execution()

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import os
import pickle

# 시간 측정
import time
start = time.time()

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

sources, targets = [], []
with open(os.path.join(ROOT_DIR,'Data_Preprocess','sources.dat'), 'rb') as sources_stream:
    sources = pickle.load(sources_stream)
with open(os.path.join(ROOT_DIR,'Data_Preprocess','targets.dat'), 'rb') as targets_stream:
    targets = pickle.load(targets_stream)

source2idx, idx2source, target2idx, idx2target = None, None, None, None

with open(os.path.join(os.path.dirname(__file__), 'source2idx.dat'), 'rb') as source2idx_stream:
    source2idx = pickle.load(source2idx_stream)
with open(os.path.join(os.path.dirname(__file__), 'idx2source.dat'), 'rb') as idx2source_stream:
    idx2source = pickle.load(idx2source_stream)
with open(os.path.join(os.path.dirname(__file__), 'target2idx.dat'), 'rb') as target2idx_stream:
    target2idx = pickle.load(target2idx_stream)
with open(os.path.join(os.path.dirname(__file__), 'idx2target.dat'), 'rb') as idx2target_stream:
    idx2target = pickle.load(idx2target_stream)


def preprocess(sequences, max_len, dic, mode='source'):
    assert mode in ['source', 'target'], 'source와 target 중에 선택'

    if mode == 'source':
        # preprocessing for source (use in encoder)
        s_input = list(map(lambda sentence: [dic.get(token) for token in sentence], sequences))
        s_len = list(map(lambda sentence: len(sentence), s_input))
        # source의 길이는 3으로 고정될 것이므로 pad_sequences가 필요없어서 주석처리
        s_input = pad_sequences(sequences=s_input, maxlen=max_len, padding='post', truncating='post')
        return s_len, s_input

    elif mode == 'target':
        # preprocessing for target (use in decoder)
        # decoder input
        # bos: beginning of sentence eos: end of sentence
        t_input = list(map(lambda sentence: ['<bos>'] + sentence + ['<eos>'], sequences))
        t_input = list(map(lambda sentence: [dic.get(token) for token in sentence], t_input))
        t_len = list(map(lambda sentence: len(sentence), t_input))
        t_input = pad_sequences(sequences=t_input, maxlen=max_len, padding='post',
                                truncating='post')  # truncating: 길이가 초과할경우 어디서부터 유효할지

        # decoder output
        t_output = list(map(lambda sentence: sentence + ['<eos>'], sequences))
        t_output = list(map(lambda sentence: [dic.get(token) for token in sentence], t_output))
        t_output = pad_sequences(sequences=t_output, maxlen=max_len, padding='post', truncating='post')

        return t_len, t_input, t_output


# sources preprocess test
s_max_len = 2  # not needed
s_len, s_input = preprocess(sequences=sources,
                            max_len=s_max_len, dic=source2idx, mode='source')
print(s_len, s_input)

# targets preprocess test
t_max_len = 20
t_len, t_input, t_output = preprocess(sequences=targets,
                                      max_len=t_max_len, dic=target2idx, mode='target')
print(t_len, t_input, t_output)

# hyper-parameters
epochs = 1
batch_size = 1000
learning_rate = .005
total_step = epochs / batch_size
buffer_size = 1000
n_batch = buffer_size // batch_size  # //: 몫
embedding_dim = 32
units = 128

# input
data = tf.data.Dataset.from_tensor_slices((s_len, s_input, t_len, t_input, t_output))
data = data.shuffle(buffer_size=buffer_size)
data = data.batch(batch_size=batch_size)


# iterator.get_next() -> pop (s_len, s_input, t_len, t_input, t_output)


# def gru(units):  # gru: 순환 RNN (https://www.tensorflow.org/api_docs/python/tf/keras/layers/CuDNNGRU)
# 2.0 알파에서는 CuDNNGRU에 문제가 있나봄 or 필요없게 통합
# if tf.test.is_gpu_available():
#    return tf.keras.layers.CuDNNGRU(units,
#                                    return_sequences=True,
#                                    return_state=True,
#                                    recurrent_initializer='glorot_uniform')
# else:
# return tf.keras.layers.GRU(units,
#                           return_sequences=True,
#                           return_state=True,
#                           recurrent_activation='sigmoid',
#                           recurrent_initializer='glorot_uniform'
#                           )


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_activation='sigmoid',
                                       recurrent_initializer='glorot_uniform'
                                       )

    def __call__(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_activation='sigmoid',
                                       recurrent_initializer='glorot_uniform'
                                       )
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, x, hidden, enc_output):  # 너무 어려운 개념이 많이쓰임... attention 눙물나쥬
        # enc_output shape == (batch_size, max_length, hidden_size)

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # * `score = FC(tanh(FC(EO) + FC(H)))`
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        # * `attention weights = softmax(score, axis = 1)`. Softmax by default is applied on the last axis but here we want to apply it on the *1st axis*, since the shape of score is *(batch_size, max_length, 1)*. `Max_length` is the length of our input. Since we are trying to assign a weight to each input, softmax should be applied on that axis.
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        # * `context vector = sum(attention weights * EO, axis = 1)`. Same reason as above for choosing axis as 1.
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # * `embedding output` = The input to the decoder X is passed through an embedding layer.
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # * `merged vector = concat(embedding output, context vector)`
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


encoder = Encoder(len(source2idx), embedding_dim, units, batch_size)
decoder = Decoder(len(target2idx), embedding_dim, units, batch_size)


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask

    return tf.reduce_mean(loss_)


# creating optimizer
# optimizer = tf.train.AdamOptimizer() -> in 2.0 make error
optimizer = tf.optimizers.Adam()

# 일단 지워둠
# creating check point (Object-based saving)
checkpoint_dir = './data_out/training_checkpoints/2words_input'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# create writer for tensorboard
# in 2.0 tf.contrib deleted....
# summary_writer = tf.contrib.summary.create_file_writer(logdir=checkpoint_dir)

EPOCHS = epochs

# 체크 포인트 사용할때 아래코드 사용 지울부분 지우기(임시)
# checkpoint.restore('./data_out/training_checkpoints/ckpt-1.index')

# and delete this part-----------------------------------------------------------------------------------------------
for epoch in range(EPOCHS):

    # initialize
    hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for i, (s_len, s_input, t_len, t_input, t_output) in enumerate(data):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(s_input, hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([target2idx['<bos>']] * batch_size, 1)

            # Teacher Forcing: feeding the target as the next input
            for t in range(1, t_input.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(t_input[:, t], predictions)

                dec_input = tf.expand_dims(t_input[:, t], 1)  # using teacher forcing

            batch_loss = (loss / int(t_input.shape[1]))

            total_loss += batch_loss

            variables = encoder.variables + decoder.variables

            gradient = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradient, variables))

        if epoch % 1 == 0:
            # save model every 10 epoch
            print('Epoch {} Loss {:.4f} Batch Loss {:.4f}'.format(epoch,
                                                                  total_loss / n_batch,
                                                                  batch_loss.numpy()))
    # 일단 지워둠
    # checkpoint.save(file_prefix=checkpoint_prefix)

# 학습된 모델 저장
checkpoint.save(file_prefix=checkpoint_prefix)


# and delete this part-----------------------------------------------------------------------------------------------

def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    # sentence = preprocess_sentence(sentence)

    inputs = [inp_lang[i] for i in sentence.split(' ')]
    # 필요하지않아서 주석처리
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang['<bos>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += idx2target[predicted_id] + ' '

        if idx2target.get(predicted_id) == '<eos>':
            return result, sentence

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


# result, sentence = evaluate(sentence, encoder, decoder, source2idx, target2idx,
#                                             s_max_len, t_max_len)


def test(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    result, sentence = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(sentence + ' ' + result))


# test!!!
sentence = '문재인 경제'

test(sentence, encoder, decoder, source2idx, target2idx, s_max_len, t_max_len)

# 시간 측정 종료
print("batch_size: {}, epoch: {}, time : {:.1f}min".format(batch_size, epochs, (time.time() - start) / 60))
