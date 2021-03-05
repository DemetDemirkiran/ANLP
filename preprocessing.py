import nltk
import re
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
from nltk.stem import WordNetLemmatizer


def read():
    with open('/home/demet/Desktop/ANLP_Project/Electronics.txt', 'r') as f:
        electronics = f.read()
    f.close()
    return electronics


def read_pos():
    with open('/home/demet/Desktop/ANLP_Project/positive_words.txt', 'r') as f:
        positive = f.read()
    f.close()
    return positive


def read_neg():
    with open('/home/demet/Desktop/ANLP_Project/negative_words.txt', 'r') as f:
        negative = f.read()
    f.close()
    return negative


def text_to_dict(electronics):
    dictionary_for_reviews = list(electronics)
    return dictionary_for_reviews


def lowercase(texts_from_dict):
    lower_text = texts_from_dict.lower()
    return lower_text


def numbers(texts_from_dict):
    no_numbers = re.sub('[0-9]+', '', texts_from_dict)
    return no_numbers


def punctuation_and_spaces(texts_from_dict):
    # removes [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]

    texts_from_dict = texts_from_dict.split()
    no_pnc_space_text = texts_from_dict.translate(string.maketrans("", ""), string.punctuation)
    return no_pnc_space_text


def tokenize(texts_from_dict):
    stop_words = set(stopwords.words('english'))
    tokenize = word_tokenize(texts_from_dict.lower())
    # filtered_sentence = [w for w in tokenize if not w in stop_words]
    texts_from_dict = str.maketrans('', '', string.punctuation)
    stripped_texts_from_dict = [w.translate(texts_from_dict) for w in tokenize]
    words = [w for w in stripped_texts_from_dict if w.isalpha()]  # numbers
    filtered_stripped_texts_from_dict = [w for w in words if not w in stop_words]
    filtered_stripped_texts_from_dict = []

    for w in words:
        if w not in stop_words:
            filtered_stripped_texts_from_dict.append(w)

    return filtered_stripped_texts_from_dict


def positive_tagging():
    ...


def negative_tagging():
    ...


def pos_tagging(texts_from_dict):
    tagged_text = pos_tag(texts_from_dict)
    exclude = set(string.punctuation)
    stripped_dict = [w for w in tagged_text if w not in exclude]
    return stripped_dict


def stemming(texts_from_dict):
    ps = nltk.SnowballStemmer()
    stemmed_text = []
    for w in texts_from_dict:
        stemmed_text.append(ps.stem(w))
    return stemmed_text


def lemmanization(texts_from_dict):
    lemmatizer = WordNetLemmatizer()
    lemma_sentence = []
    for word in texts_from_dict:
        lemma_sentence.append(lemmatizer.lemmatize(word))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)


def summarize():
    ...


def chunking():
    ...


def ner():
    ...


def relation_extraction():
    ...


def getFrequencyDictForText(sentence):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    for text in sentence.split(" "):
        if re.match("a|the|an|to|in|for|of|or|by|with|is|on|that|be", text):
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])
    return fullTermsDict


def wordcloud(data):
    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    # text = ''.join(data)
    # tokenize = word_tokenize(data)
    # text = getdata.read()
    text = open(path.join(d, 'project eragon/char.txt')).read()
    alice_mask = np.array(Image.open(path.join(d, "open-book.png")))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white", max_words=75, mask=alice_mask,
                   stopwords=stopwords, contour_width=3, contour_color='darkblue')
    wc.generate(text)
    wc.to_file(path.join(d, "eragon.png"))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.figure()
    plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def frequency_char(distribution, data):
    tokenize = word_tokenize(data.lower())
    table = str.maketrans('', '', string.punctuation)
    stripped_book = [w.translate(table) for w in tokenize]
    words = [w for w in stripped_book if w.isalpha()]
    filtered_book = []

    for w in words:
        filtered_book.append(w)

    fdist = FreqDist(filtered_book)
    fdist.most_common(10)
    fdist.plot(20, cumulative=False)
    plt.show
    return fdist


def frequency(distribution):
    fdist = FreqDist(distribution)
    fdist.most_common(2)
    fdist.plot(20, cumulative=False)
    plt.show
    return fdist
