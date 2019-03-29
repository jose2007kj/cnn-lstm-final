from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout, Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
 
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# turn a doc into clean tokens
def clean_doc(doc, vocab):
	# split into tokens by white space
	# print(doc.split())
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens
 

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# read input
ip_text = input("Enter a sentance to check sentiment using cnn-lstm: ")
print("input sentance is " + ip_text + "!")
# pre-process input
tokens = clean_doc(ip_text, vocab)


# # create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
# print(*tokens, sep='\n')
tokenizer.fit_on_texts(tokens)
 
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(tokens)
# pad sequences
# print (encoded_docs)
max_length = 1209 #got while training
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1
print('------------------------Loading model---------------------------------------')
model = load_model("cnn-lstm_model.h5")
probability = (model.predict(Xtrain)[0][0] > 0.5).astype(int)
print('---------------Prediction value is...------------------')
print(probability)
