# with deep learning zero to all season 2
# When the 3D-input gets

# Import Tensorflow >= 1.10 and enable eager execution
import tensorflow as tf
from RNN_training import preprocess, Encoder, Decoder, loss_function

# not needed anymore?!
# tf.enable_eager_execution()
import os
import pickle
# 시간 측정
import time
start = time.time()

# hyper-parameters
epochs = 25
batch_size = 250
learning_rate = .005
total_step = epochs / batch_size
buffer_size = 300
n_batch = buffer_size // batch_size  # //: 몫
embedding_dim = 32
units = 128

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

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
print(s_len, s_input)

# targets preprocess test
t_max_len = 20
t_len, t_input, t_output = preprocess(sequences=targets,
                                      max_len=t_max_len, dic=target2idx, mode='target')
print(t_len, t_input, t_output)


# input
data = tf.data.Dataset.from_tensor_slices((s_len, s_input, t_len, t_input, t_output))
data = data.shuffle(buffer_size=buffer_size)
data = data.batch(batch_size=batch_size)


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

encoder = Encoder(len(source2idx), embedding_dim, units, batch_size)
decoder = Decoder(len(target2idx), embedding_dim, units, batch_size)


# creating optimizer
# optimizer = tf.train.AdamOptimizer() -> in 2.0 make error
optimizer = tf.optimizers.Adam()

# 일단 지워둠
# creating check point (Object-based saving)
checkpoint_dir = './data_out/training_checkpoints/3words_input'
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
sentence = '문재인 경제 회생'

test(sentence, encoder, decoder, source2idx, target2idx, s_max_len, t_max_len)

# 시간 측정 종료
print("batch_size: {}, epoch: {}, time : {:.1f}min".format(batch_size, epochs, (time.time() - start) / 60))
