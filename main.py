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


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')






