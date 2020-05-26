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
import warnings
warnings.filterwarnings('ignore')

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

from lyricsutils import (
	build_data,
	vectorize_data,
	create_model_residual,
	plot_history,
	generate_output
	)

class lyricsgenius:

	train_config = {
		'rnn_width': 64,
	    'rnn_depth': 4,
	    'rnn_dropout': 0.3,
	    'bidirectional': True,
	    
	    
	}

	predict_config = {
		'Tx': 40,
		'prediction_length': 300,
		"temperature": 0.2

	}

	embedding_size = 128
	network_path = ""
	alphabet_path = ""
	tokenizer_path = ""
	model = None
	Tx = 40
	history = []


	def __init__(self, network_path=None,
 					alphabet_path=None,
	  				tokenizer_path=None):
		if network_path is None:
			self.network_path = "model.h5"
		else:
			self.network_path = network_path

		if alphabet_path is None:
			self.alphabet_path = "alphabet.h5"
		else:
			self.alphabet_path = alphabet_path

		if tokenizer_path is None:
			self.tokenizer_path = "tokenizer.h5"
		else:
			self.tokenizer_path = tokenizer_path


	def load_corpus(self, corpus_path):
		self.text = io.open(corpus_path, encoding='utf-8').read().lower()
		chars = sorted(list(set(text)))
		X, Y = build_data(text, Tx=Tx, stride = 3)
		x, y = vectorize_data(X, Y, chars, self.alphabet_path, self.tokenizer_path)
		return x, y


	def create_model(self, train_config, continue_learning=False):
		if continue_learning:
			self.model = load_model(self.network_path)
			
		else:
			self.model = create_model_residual(self.train_config)
			

	def train(self, x, y, batch_size=128, epochs=30):
		self.history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=True)
		return history

	def plot_loss(self):
		plot_history(self.history)

	def save_model(self):
		self.model.save(network_path, overwrite=True)
		print("Model succesfully saved to disk")

	def predict(self, text):

		alphabet = None
		tk = None

		if self.model is None:
			self.create_model(self.train_config, continue_learning=True)


		# Load alphabet
		with open(self.alphabet_path, 'r') as fp:
		    alphabet = json.load(fp)
		    
		# Load tokenizer
		with open(self.tokenizer_path, 'rb') as fp:
		    tk = pickle.load(fp)


		# construct a new vocabulary
		char_dict = {}
		for i, char in enumerate(alphabet):
		    char_dict[char] = i + 1


		return generate_output(text, self.model, tk, alphabet, Tx=self.predict_config['Tx'], prediction_length=self.predict_config['prediction_length'], static=True, temperature=self.predict_config['temperature'])





