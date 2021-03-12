import nltk
import re
import time
from nltk.sentiment.util import *

from preprocessing import text_to_dict, read, lowercase,  punctuation_and_spaces, lemming, stemming, \
    list_to_dataframe, dataframe_to_csv
from senti_bert import Senti_Bert
from bert import bert_dataloader
from splitter import splitter, TextLoader
from torch.utils.data import DataLoader

def train(dataloader):
    model = Senti_Bert().cuda()

    for train, target in dataloader:
        train = train.cuda()
        target = target.cuda()

        model(train)








if __name__ == '__main__':
    #nltk.download('averaged_perceptron_tagger')
    #nltk.download('punkt')
    #nltk.download('stopwords')
    #nltk.download('wordnet')

    #reviews = read()
    #small = lowercase(reviews)
    #dictionary = text_to_dict(small)
    #punct = punctuation_and_spaces(dictionary)
    # toknz = tokenize(small) # do not need it because all of te tokenization, and preprocessing is now done in the
    # function punctuation_and_spaces, the function tokenize is now obsolete
    # lemmz = lemming(toknz) #Tested this, found that it just added
    # more spaces into the list that contains the details of the reviews, not really needed. CURRENTLY
    # stemmz = stemming(toknz) #does not do much at the moment cant tell a difference
    #dtaframe = list_to_dataframe(dictionary)
    #dtaframe = dataframe_to_csv(dtaframe)
    #print(dtaframe)

    reviews = read()
    small = lowercase(reviews)
    dictionary = text_to_dict(small)
    punct = punctuation_and_spaces(dictionary)
    dtaframe = list_to_dataframe(punct)

    tl = TextLoader('/home/demet/Desktop/review_dataframe_notoken.csv')
    dataloader = DataLoader(tl, batch_size=4, shuffle=True)
    train(dataloader)

