from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, state_union
from collections import Counter
from nltk import pos_tag
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.sentiment.util import *
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import string
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import os
from os import path
import multidict as multidict
from nltk.stem import WordNetLemmatizer
import re
from pandas import DataFrame


def read():
    with open('/home/demet/Desktop/ANLP_Project/Arts.txt', 'r') as f:
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


def text_to_dict(amazon_reviews):
    dictionary = dict()
    amazon_reviews = amazon_reviews.split('\n')

    matrix = list()
    for idx in np.arange(0, len(amazon_reviews), 11):
        reviews = amazon_reviews[idx:idx + 10]
        aux = list()

        for field in reviews:
            if not field == '':
                key, value, *rest = field.split(': ')
                aux.append(value)
        matrix.append(aux)

    return matrix


def lowercase(texts_from_dict):
    lower_text = texts_from_dict.lower()
    return lower_text



def punctuation_and_spaces(texts_from_dict):
    # removes [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]

    # texts_from_dict = texts_from_dict.split()
    # no_pnc_space_text = re.sub(rf"[{string.punctuation}]", "", texts_from_dict)

    # doesnt work because of the : needed to separate keys and values, so we maually remove all punctuations except for :

    # punctuations = '''!()-[]{};'"\, <>./?@#$%^&*_~'''

    # Removing punctuations in string
    # Using loop + punctuation string
    # for i in tqdm(texts_from_dict):
    #    if i in punctuations:
    #        no_pnc_space_text = texts_from_dict.replace(i, "")
    ## PREVIOUS WORKING CODE
    # remove = string.punctuation
    # remove = remove.replace(":", "")  # don't remove hyphens
    # pattern = r"[{}]".format(remove)  # create the pattern
    #
    # txt = ")*^%{}[]thi's - is - @@#!a !%%!!%- test."
    # new_texts_from_dict = re.sub(pattern, "", texts_from_dict)
    ## END OF PREVIOUS WORKING CODE

    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    new_dict = texts_from_dict.copy()
    for i, row in enumerate(new_dict):
        for j, item in enumerate(row):
            if j in [8, 9]:
                # tokens = word_tokenize(item)
                # tokens = [w for w in item if w.isalpha()]
                # tokens_no_stop = [i for i in tokens if not i in stop_words]
                # tokens_no_punct = [i for i in tokens_no_stop if not i in string.punctuation]
                # new_dict[i][j] = tokens_no_punct
                pass
    return new_dict


#def tokenize(texts_from_dict):
    # THIS FUNCTION IS NOW OBSOLETE BECAUSE OF THE FUNCTION punctuation_and_spaces THEREFORE I AM COMMENTING IT OUT

    #stop_words = set(stopwords.words('english'))

    # stripped_texts_from_dict = [w.translate(texts_from_dict) for w in texts_from_dict]  # tokenize
    # words = [w for w in stripped_texts_from_dict if w.isalpha()]  # numbers
    # filtered_stripped_texts_from_dict = [w for w in stripped_texts_from_dict if not w in stop_words]  # stopwords
    # filtered_stripped_texts_from_dict = []

    # for w in stripped_texts_from_dict:
    #    if w not in stop_words:
    #        filtered_stripped_texts_from_dict.append(w)

    #tokens = word_tokenize(texts_from_dict)
    #result = [i for i in tokens if not i in stop_words]

    #return result


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


def lemming(texts_from_dict):
    lemmas = WordNetLemmatizer()
    lemma_sentence = []
    for word in texts_from_dict:
        lemma_sentence.append(lemmas.lemmatize(word))
        lemma_sentence.append(" ")
    return "".join(lemma_sentence)


def find_proper_nouns(tagged_text):
    proper_nouns = []
    i = 0
    while i < len(tagged_text):
        if tagged_text[i][1] == 'NNP':
            if tagged_text[i + 1][1] == 'NNP':
                proper_nouns.append(tagged_text[i][0].lower() +
                                    " " + tagged_text[i + 1][0].lower())
                i += 1
            else:
                proper_nouns.append(tagged_text[i][0].lower())
        i += 1
    return proper_nouns


def summarize(proper_nouns, top_num):
    counts = dict(Counter(proper_nouns).most_common(top_num))
    return counts


def chunking():
    train_text = state_union.raw('/home/demet/Desktop/Eragon/char.txt')
    sample_text = state_union.raw('/home/demet/Desktop/Eragon/eldest.txt')
    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(sample_text)
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<NNP>*<PRP>*<WRB>*<WP>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)

            chunked.draw()
            print(chunked)
    except Exception as e:
        print(str(e))


def ner():
    train_text = state_union.raw('...')
    sample_text = state_union.raw('...')
    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(sample_text)
    # classifier = nltk.NaiveBayesClassifier.train(custom_sent_tokenizer)

    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()
            print(namedEnt)
    except Exception as e:
        print(str(e))
    # classifier.show_most_informative_features(15)


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


def list_to_dataframe(list):
    df = DataFrame(list, columns=['product/productID', 'product/title', 'product/price', 'review/userID',
                                  'review/profileName', 'review/helpfulness', 'review/score',
                                  'review/time', 'review/summary', 'review/text'])
    return df

def dataframe_to_csv(df):

    df.to_csv('/home/demet/Desktop/review_dataframe_notoken.csv', header=True)

    return df
