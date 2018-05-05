# reference: https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

import MultiProcessing as mp
import psutil
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

def GetTopics(reviews, out, widgets):
	lemmer = WordNetLemmatizer()
	tokenizer = RegexpTokenizer(r'\w+')

	proc = psutil.Process()
	widgets.put(("Process " + str(proc.pid) + " pipe is working"))
	print('Process PID: ' + str(proc.pid) + ', Affinity: ' +str(proc.cpu_affinity()))
	sys.stdout.flush()

	reviews['topics'] = Series(0, index=reviews.index)

	fiveperc = len(reviews) / 20

	for index, row in reviews.iterrows():
		text = []
		for review in row['reviews']:
			tokens = tokenizer.tokenizer(reviews)
			tokens = [lemmer.lemmatize(w) for w in tokens]
			text.append(tokens)

		dictionary = corpora.Dictionary(text)
		# convert tokenized documents into a document-term matrix
		corpus = [dictionary.doc2bow(t) for t in text]
	
		# generate LDA model
		ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=7, id2word = dictionary, passes=50)
		topics = ldamodel.print_topics(num_topics = 7, 3)

		reviews.at[index, 'topics'] = topics

	out.put(reviews)
