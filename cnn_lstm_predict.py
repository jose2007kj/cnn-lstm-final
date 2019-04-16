import string
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

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
	pos = process_docs('txt_sentoken/pos', vocab, is_train)
	docs = neg + pos
	# prepare labels
	labels = array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
	return docs, labels

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

# classify a review as negative or positive
def predict_sentiment(review, vocab, tokenizer, max_length, model):
	# clean review
	line = clean_doc(review, vocab)
	# encode and pad review
	padded = encode_docs(tokenizer, max_length, [line])
	# predict sentiment
	yhat = model.predict(padded, verbose=0)
	# retrieve predicted percentage and label
	percent_pos = yhat[0,0]
	if round(percent_pos) == 0:
		return (1-percent_pos), 'NEGATIVE'
	return percent_pos, 'POSITIVE'

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load all reviews
train_docs, ytrain = load_clean_dataset(vocab, True)
test_docs, ytest = load_clean_dataset(vocab, False)
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
Xtest = encode_docs(tokenizer, max_length, test_docs)
# load the model
model = load_model('worst_case.h5')
# evaluate model on training dataset
_, acc = model.evaluate(Xtrain, ytrain, verbose=0)
print('Train Accuracy: %.2f' % (acc*100))
# evaluate model on test dataset
_, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %.2f' % (acc*100))

# test positive text
text = input("Enter a sentance to check sentiment using cnn-lstm: ")
print("input sentance is " + text + "!")
# text = 'martin scorseses triumphant adaptation age innocence stunning film quintessential new york filmmaker man brought streets taxi driver mean streets life seems like odd choice scorsese period piece early fact pulls brilliantly wonder testament greatness scorsese filmmaker gorgeous visual experience surely one scorseses finest archer daylewis prestigious lawyer engaged may ryder somewhat empty shallow new yorker belongs prestigious family quite beautiful marriage one unite two prestigious families society nothing important opinions others day archer announce engagement may ellen pfeiffer cousin may walks life archer immediately captivated finds love ellen archer also bound limits new york society intrusive world archer finds secret love affair mind attempting keep mind trying lose social status films subject matter may seem alien scorsese theme definitely theme forbidden romance guilty pleasures consequences causes actions flawed hero choice life wants life destined truth film society audience doesnt know wants find much like society goodfellas even kundun performances absolutely breathtaking daylewis portrays mental anguish face one man forced take pfeiffer marvelous mix passion beauty audience would die well ryder probably gem group quiet presence plot slowly pushes daylewis closer closer eventual ending supporting cast also wonderful several characters singular indelible ones memory scorsese definitely passion filmmaking lavish sumptuous set design marvelous recreation new york wondrous sight literally transports viewer another world incredible imagery script also excellent slow buildup rapid conclusion fantastic ending seen believed difficult make period piece gripping scorsese however beautifully famous cameras legendary director also everywhere patient films everything anything remotely important cameras sweep pan track theyve ever done subtle one doesnt realize hes watching scorsese viewing central tracking shot probably longer complex famous goodfellas shot viewer doesnt notice want see gorgeous world deft touches filmmaking simply outstanding narration exquisite fast film like goodfellas shares common kundun anything else like kundun film truly shines given chance fully breathe end beautiful film director continuing challenge year year!'
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
# test negative text
# text = 'walt disney studios may finally met match lush animation twentieth century foxs anastasia judging latest efforts studios visuals thing fox brag disneys recent classics occasionally stretched credibility films pocahontas hunchback notre dame lesser extent hercules anastasia fox gone far throw facts completely window may say kids movie well young kids beware may noticeably frightened visuals zombie whose body parts continually fall real way consider warned nevertheless animation quite stunning times used computer animation throughout occasionally quality yet scenes material seems tv crowd leads wonder rushed market combat disney plot anyone read history knows concerns attempt return anastasia royal family lost overthrow anastasia much concerned really happened plot go rent disneys youll see anastasia'
text = input("Enter a sentance to check sentiment using cnn-lstm: ")
print("input sentance is " + text + "!")
percent, sentiment = predict_sentiment(text, vocab, tokenizer, max_length, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))

