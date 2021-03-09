import nltk
from nltk.sentiment.util import _show_plot
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import time
from collections import Counter
import pprint
from nltk import pos_tag
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import pandas as pd
import string
import matplotlib.animation as animation
# from pandas.plotting._matplotlib import style
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict
from nltk.corpus import opinion_lexicon
from nltk.tokenize import treebank
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import os
from os import path
import ast
import multidict as multidict
from preprocessing import text_to_dict, read, lowercase,  punctuation_and_spaces, lemming, stemming, \
    list_to_dataframe, dataframe_to_csv

if __name__ == '__main__':
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    reviews = read()
    small = lowercase(reviews)
    dictionary = text_to_dict(small)
    punct = punctuation_and_spaces(dictionary)
    # toknz = tokenize(small) # do not need it because all of te tokenization, and preprocessing is now done in the
    # function punctuation_and_spaces, the function tokenize is now obsolete
    # lemmz = lemming(toknz) #Tested this, found that it just added
    # more spaces into the list that contains the details of the reviews, not really needed. CURRENTLY
    # stemmz = stemming(toknz) #does not do much at the moment cant tell a difference
    dtaframe = list_to_dataframe(punct)
    dtaframe = dataframe_to_csv(dtaframe)
    #print(dtaframe)
