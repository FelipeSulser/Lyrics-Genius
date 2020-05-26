from __future__ import print_function
# Data manipulation
import pydot
import numpy as np
import pandas as pd

# Misc libraries
import json
import pickle
import sys
import io


# Deep Learning libraries
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, Embedding, InputLayer
from keras.layers import LSTM, Lambda, concatenate, Bidirectional, Concatenate, SpatialDropout1D
from keras.utils.vis_utils import plot_model
import keras
from keras.layers.merge import add
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.utils import to_categorical


def generate_output(text, model, tk, alphabet,Tx=40, prediction_length=400, static=False, temperature=0.1):
    generated = ''
    
    # zero pad the sentence to Tx characters.
    sentence = ('{0:0>' + str(Tx) + '}').format(text).lower()
    generated += text 

   
   
    for i in range(prediction_length):
        predict_sequence = tk.texts_to_sequences([sentence])
  
        # Padding
        predict_data = pad_sequences(predict_sequence, maxlen=Tx, padding='post')

        # Convert to numpy array
        x_pred = np.array(predict_data, dtype='float32')

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature = temperature)
        next_char = alphabet[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        

        if next_char == '\n':
            continue
    return generated



def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(len(chars)), p = probas.ravel())
    return out


def build_data(text, Tx = 40, stride = 3):
    """
    Create a training set by scanning a window of size Tx over the text corpus, with stride 3.
    
    Arguments:
    text -- string, corpus of Shakespearian poem
    Tx -- sequence length, number of time-steps (or characters) in one training example
    stride -- how much the window shifts itself while scanning
    
    Returns:
    X -- list of training examples
    Y -- list of training labels
    """
    
    X = []
    Y = []

    for i in range(0, len(text) - Tx, stride):
        X.append(text[i: i + Tx])
        Y.append(text[i + Tx])

    
    print('number of training examples:', len(X))
    
    return X, Y



def vectorize_data(X, Y, alphabet, alphabet_path="alphabet.json", tokenizer_path="tokenizer.pkl"):

	tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
	tk.fit_on_texts(X)
	
	#Store alphabet to make predictions
	with open(alphabet_path, 'w+') as fp:
	    json.dump(alphabet, fp)
	    
	char_dict = {}
	for i, char in enumerate(alphabet):
	    char_dict[char] = i + 1

	# Use char_dict to replace the tk.word_index
	tk.word_index = char_dict.copy()
	# Add 'UNK' to the vocabulary
	tk.word_index[tk.oov_token] = max(char_dict.values()) + 1


	# Save tokenizer to path to make predictions
	with open(tokenizer_path, 'wb') as handle:
	    pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Convert string to index
	train_sequences = tk.texts_to_sequences(X)

	# Padding
	train_data = pad_sequences(train_sequences, maxlen=Tx, padding='post')

	# Convert to numpy array
	train_data = np.array(train_data, dtype='float32')

	# =======================Get classes================
	train_classes = [elem[0] for elem in tk.texts_to_sequences(Y)]
	train_class_list = [x - 1 for x in train_classes]
	train_classes = to_categorical(train_class_list)

	x, y = train_data, train_classes
	return x, y



def new_lstm_cell(rnn_width, rnn_dropout, bidirectional=True, return_sequences=False):
    if bidirectional:
        return Bidirectional(LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout,return_sequences=return_sequences))
    else:
        return LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout,return_sequences=return_sequences)



def make_lstm_layers(input, rnn_width, rnn_depth, rnn_dropout, bidirectional=True):
    layer_list = []
    layer = input
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1
        prev_layer = input if i == 0 else layer_list[-1]
        layer = new_lstm_cell(rnn_width, rnn_dropout, bidirectional=bidirectional, return_sequences=return_sequences)
        
        layer_list.append(layer)
    return layer, layer_list
    


def make_residual_lstm_layers(input, rnn_width, rnn_depth, rnn_dropout, bidirectional=True):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    x = input
    layer_list = []
    for i in range(rnn_depth):
        return_sequences = i < rnn_depth - 1
        x_rnn = Bidirectional(LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=return_sequences))(x)
        if return_sequences:
            # Intermediate layers return sequences, input is also a sequence.
            if i > 0 or input.shape[-1] == rnn_width:
                x = add([x, x_rnn])
            else:
                # Note that the input size and RNN output has to match, due to the sum operation.
                # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
                x = x_rnn
        else:
            # Last layer does not return sequences, just the last element
            # so we select only the last element of the previous output.
            def slice_last(x):
                return x[..., -1, :]
            x = add([Lambda(slice_last)(x), x_rnn])
        layer_list.append(x_rnn)
    return x, layer_list


def create_model_residual(model_config):
    inputs = Input(shape=(Tx, ), name='sent_input', dtype='int64')
    
    embeddings = keras.layers.Embedding(len(chars) + 1, embedding_size, input_length=Tx)(inputs)
    embeddings = SpatialDropout1D(model_config['rnn_dropout'], name='spatial-dropout')(embeddings)
    lstm_layer, layer_list = make_residual_lstm_layers(embeddings, **model_config)
    
    dense_layer = keras.layers.Dense(len(chars), activation='softmax')(lstm_layer)
    model = keras.Model(inputs=inputs, outputs=dense_layer)
    optimizer = keras.optimizers.Adam(learning_rate=4e-3)
    model.compile( loss='categorical_crossentropy', optimizer=optimizer)
    return model

# Simple Deep LSTM Model without Residual Units
def create_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(Tx, len(chars))))
    model.add(LSTM(128, input_shape=(Tx, len(chars)), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(len(chars), activation='softmax'))

    optimizer = keras.optimizers.Adam(learning_rate=4e-3)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model