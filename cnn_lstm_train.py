import string
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dense,BatchNormalization
from keras.layers import Dropout, Activation
from keras import backend as K

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
	tokens = doc.split()
	# prepare regex for char filtering
	re_punc = re.compile('[%s]' % re.escape(string.punctuation))
	# remove punctuation from each word
	tokens = [re_punc.sub('', w) for w in tokens]
	# filter out tokens not in vocab
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_train):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc, vocab)
		# add to list
		documents.append(tokens)
	return documents

# load and clean a dataset
def load_clean_dataset(vocab, is_train):
	# load documents
	neg = process_docs('txt_sentoken/neg', vocab, is_train)
	# if is_train:
	# 	with open("negative_combined.txt", "w") as text_file:
	# 		for line in neg:
	# 			text_file.write(line+"\t"+str(0)+"\n")


	pos = process_docs('txt_sentoken/pos', vocab, is_train)
	# if is_train:
	# 	with open("positive_combined.txt", "w") as text_file:
	# 		for line in pos:
	# 			text_file.write(line+"\t"+str(1)+"\n")
	# for line in open(dataFile, 'r'):
	# 	d = line.split('\t')
	docs = neg + pos
	# combiiininf file and writing to a file
	# with open("combined_train.txt", "w") as text_file:
	# 	with open("positive_combined.txt") as file1, open("negative_combined.txt") as file2:
	# 		for line1, line2 in zip(file1, file2):
	# 			text_file.write(line2)
	# 			text_file.write(line1)

	# prepare labels
	labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
	train_text=list()
	train_senti=list()
	for line in open("rain.txt", 'r'):
		d = line.split('\t')
		text = d[0].strip()
		senti = d[1].strip()
		train_text.append(text)
		train_senti.append(senti)
	return train_text, train_senti

def get_train_data():
	train_text=list()
	train_senti=list()
	for line in open("rain.txt", 'r'):
		d = line.split('\t')
		text = d[0].strip()
		senti = d[1].strip()
		train_text.append(text)
		train_senti.append(senti)
	return train_text, train_senti
# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# integer encode and pad documents
def encode_docs(tokenizer, max_length, docs):
	# integer encode
	encoded = tokenizer.texts_to_sequences(docs)
	# pad sequences
	padded = pad_sequences(encoded, maxlen=max_length, padding='post')
	return padded

# define the model
def define_model(vocab_size, max_length):
	model = Sequential()
	model.add(Embedding(vocab_size, 100, input_length=max_length))
	model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(LSTM(100))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
	# model = Sequential()
	# model.add(Embedding(vocab_size, 100, input_length=max_length))
	# model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
	# model.add(MaxPooling1D(pool_size=2))
	# model.add(Flatten())
	# model.add(Dense(10, activation='relu'))
	# model.add(Dense(1, activation='sigmoid'))
	# # compile network
	# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize defined model
	model.summary()
	# plot_model(model, to_file='model.png', show_shapes=True)
	return model

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load training data
train_docs, ytrain = get_train_data()
# train_docs, ytrain = load_clean_dataset(vocab, True)
# test_docs, ytest = load_clean_dataset(vocab, False)
# create the tokenizer
tokenizer = create_tokenizer(train_docs)
# define vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
# calculate the maximum sequence length
max_length = max([len(s.split()) for s in train_docs])
print('Maximum length: %d' % max_length)
# encode data
Xtrain = encode_docs(tokenizer, max_length, train_docs)

# tokenizer = create_tokenizer(test_docs)
# Xtest = encode_docs(tokenizer, max_length, test_docs)
# define model
model = define_model(vocab_size, max_length)
# fit network
history = model.fit(Xtrain, ytrain, validation_split=0.25, epochs=10, verbose=2)
# save the model
model.save('worst_case.h5')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('acc_result_cnn_lstm.png')
plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss_result_cnn_lstm.png')
plt.clf()
K.clear_session()


