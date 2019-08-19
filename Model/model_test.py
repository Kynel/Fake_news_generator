# with deep learning zero to all season 2
# input으로 3개를 받는 버젼임.

# Import Tensorflow >= 1.10 and enable eager execution
import tensorflow as tf
from RNN_training import preprocess, Encoder, Decoder

import numpy as np
import os
import pickle

# hyper-parameters
epochs = 1
batch_size = 100
learning_rate = .005
buffer_size = 100
embedding_dim = 32
units = 128

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
total_step = epochs / batch_size
n_batch = buffer_size//batch_size  # //: 몫



sources, targets = [], []
with open(os.path.join(ROOT_DIR,'Data_Preprocess','Data','sources3.dat'), 'rb') as sources_stream:
    sources = pickle.load(sources_stream)
with open(os.path.join(ROOT_DIR,'Data_Preprocess','Data','targets3.dat'), 'rb') as targets_stream:
    targets = pickle.load(targets_stream)

source2idx, idx2source, target2idx, idx2target = None, None, None, None

with open(os.path.join(os.path.dirname(__file__), 'data_out', 'source2idx_3.dat'), 'rb') as source2idx_stream:
    source2idx = pickle.load(source2idx_stream)
with open(os.path.join(os.path.dirname(__file__), 'data_out', 'idx2source_3.dat'), 'rb') as idx2source_stream:
    idx2source = pickle.load(idx2source_stream)
with open(os.path.join(os.path.dirname(__file__), 'data_out', 'target2idx_3.dat'), 'rb') as target2idx_stream:
    target2idx = pickle.load(target2idx_stream)
with open(os.path.join(os.path.dirname(__file__), 'data_out', 'idx2target_3.dat'), 'rb') as idx2target_stream:
    idx2target = pickle.load(idx2target_stream)


# sources preprocess test
s_max_len = 3  # not needed
s_len, s_input = preprocess(sequences=sources,
                            max_len=s_max_len, dic=source2idx, mode='source')

# targets preprocess test
t_max_len = 20
t_len, t_input, t_output = preprocess(sequences=targets,
                                      max_len=t_max_len, dic=target2idx, mode='target')

# input
data = tf.data.Dataset.from_tensor_slices((s_len, s_input, t_len, t_input, t_output))
data = data.shuffle(buffer_size=buffer_size)
data = data.batch(batch_size=batch_size)
# iterator.get_next() -> pop (s_len, s_input, t_len, t_input, t_output)

encoder = Encoder(len(source2idx), embedding_dim, units, batch_size)
decoder = Decoder(len(target2idx), embedding_dim, units, batch_size)


# creating optimizer
optimizer = tf.optimizers.Adam()

# 일단 지워둠
# creating check point (Object-based saving)
# 새로운 방식의 체크포인트 불러오기 동작하는것같음 앙기모띠
# https://www.tensorflow.org/beta/guide/checkpoints
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, encoder=encoder, decoder=decoder)
manager = tf.train.CheckpointManager(ckpt, './data_out/training_checkpoints', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint).expect_partial()

#checkpoint.restore('./data_out/training_checkpoints')

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
sentence = '문재인 vs 이명박'

def run():
    args = input("Test input 파일을 통해 테스트할 시 1, 단어 직접 입력 후 테스트 할 시 2 입력하여 모델 테스트.\n")
    if(args == '1'):
        with open("./data_out/test_input.txt", "r", encoding='UTF8') as f:
            error_words= []
            lines = f.read().splitlines()
            for line in lines:
                try:
                    test(line, encoder, decoder, source2idx, target2idx, s_max_len, t_max_len)
                except KeyError as error_key:
                    error_words.append(error_key)
            print(error_words)

    elif args == '2':
        while(True):
            sentence = input("띄어쓰기로 구분된 세 단어 입력.\n")
            try:
                test(sentence, encoder, decoder, source2idx, target2idx, s_max_len, t_max_len)
            except KeyError as error_key:
                print("이 단어가 사전에 없음.")
                print(error_key)
    else:
        print("입력값 오류. 프로그램 종료")

run()



