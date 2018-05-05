import psutil
from ipywidgets import FloatProgress
from IPython.display import display
import time
from collections import Counter
import sys

from string import punctuation
from nltk.corpus import stopwords
import nltk
import string

stop_words = set(stopwords.words('english'))
table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
def clean(s, tag = True):
	nouns = 0
	propernouns = 0
	verbs = 0
	adverbs = 0
	adjectives = 0

    #remove punctuation
	s = s.translate(table)
    #lowercase
	s = s.lower()
    #split
	tokens = s.split()
	
	if tag == True:
		pos = nltk.pos_tag(tokens)
		for tag in pos:
			if tag[1] == 'JJ' or tag[1] == 'JJR' or tag[1] == 'JJS':
				adjectives += 1
			elif tag[1] == 'NN' or tag[1] == 'NNS':
				nouns += 1
			elif tag[1] == 'NNP' or tag[1] == 'NNPS':
				propernouns += 1
			elif tag[1] == 'RB' or tag[1] == 'RBR' or tag[1] == 'RBS':
				adverbs += 1
			elif tag[1] == 'VB' or tag[1] == 'VBD' or tag[1] == 'VBG' or \
					tag[1] == 'VBN' or tag[1] == 'VBP' or tag[1] == 'VBZ':
				verbs += 1

    # remove anything non-alphanumeric
	tokens = [word for word in tokens if word.isalpha()]
	numwords = len(tokens)
    #remove stop words (a, the, for, etc.)
	tokens = [word for word in tokens if not word in stop_words]
    #filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
    
	return tokens, numwords, adjectives, adverbs, nouns, propernouns, verbs

def dictionary_make(data, out, widgets, loc):
	dictionary = Counter()
	proc = psutil.Process()
    #pbar = pb.MakeProgressBar(loc)
    #pbar.start()
	widgets.put(("Process " + str(proc.pid) + " pipe is working"))
	print('Process PID: ' + str(proc.pid) + ', Affinity: ' +str(proc.cpu_affinity()))
	sys.stdout.flush()
	i = 0
	fiveperc = len(data) / 20
	for index, row in data.iterrows():
		i += 1
		if (i % fiveperc == 0):
			print("Process " + str(loc) + " is " + str(5 * i / fiveperc) + " percent done")
		tokens, length, adjectives, adverbs, nouns, propernouns, verbs = clean(row['text'], tag=True)
		dictionary.update(tokens)
		data.set_value(index, 'text', ' '.join(tokens))
		data.set_value(index, 'length', length)
		data.set_value(index, 'adjectives', adjectives)
		data.set_value(index, 'adverbs', adverbs)
		data.set_value(index, 'nouns', nouns)
		data.set_value(index, 'propernouns', propernouns)
		data.set_value(index, 'verbs', verbs)
        #pbar.update(int((i * 100) / len(data)))
        
    #pbar.finish()	
	out.put((data, dictionary))