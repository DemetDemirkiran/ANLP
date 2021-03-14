from transformers import BertTokenizer, BertModel
from transformers import InputExample, InputFeatures
import torch
from nltk.sentiment.util import *
import nltk
from splitter import splitter
from preprocessing import text_to_dict, read, lowercase,  punctuation_and_spaces, lemming, stemming, \
    list_to_dataframe, dataframe_to_csv

def bert_dataloader(data_list, train, test, train_productID=None):
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data_list = data_list.to_numpy()
    train_data = train.to_numpy()
    test_data = test.to_numpy()
    #model.summary()

    for i in train_data:
        train_productID = i[1]
        train_helpfulness = i[6]
        train_score = i[7]
        train_summary = i[9]
        train_text = i[10]
        yield train_text

        #bert_tokens = tokenizer.batch_encode_plus([train_text])
        #outputs = model(torch.tensor(bert_tokens['input_ids']), attention_mask=torch.tensor(bert_tokens['attention_mask']))
        #embeddings = torch.mean(outputs.last_hidden_state, dim=1)






if __name__ == '__main__':

    reviews = read()
    small = lowercase(reviews)
    dictionary = text_to_dict(small)
    punct = punctuation_and_spaces(dictionary)
    dtaframe = list_to_dataframe(punct)

    train, test = splitter('/home/demet/Desktop/review_dataframe_notoken.csv')
    bert_dataloader(dtaframe, train, test)
