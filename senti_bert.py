from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import RobertaModel, RobertaTokenizer
from transformers import AlbertModel, AlbertTokenizer, AlbertForSequenceClassification
# from transformers import XLNetTokenizer, XLNetModel
import torch
from torch import nn

class Senti_Bert(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.score_fc = nn.Linear(768, 11)
        self.regression_fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        bert_tokens = self.tokenizer.batch_encode_plus(text,
                                                       padding='longest', return_tensors='pt')
        outputs = self.bert(bert_tokens['input_ids'][:, :256].cuda(),
                            attention_mask=bert_tokens['attention_mask'][:, :256].cuda())
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return self.score_fc(embeddings), self.sigmoid(self.regression_fc(embeddings))


class Senti_DistilBert(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.score_fc = nn.Linear(768, 11)
        self.regression_fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        bert_tokens = self.tokenizer.batch_encode_plus(text,
                                                       padding='longest', return_tensors='pt')
        outputs = self.bert(bert_tokens['input_ids'][:, :256].cuda(),
                            attention_mask=bert_tokens['attention_mask'][:, :256].cuda())
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return self.score_fc(embeddings), self.sigmoid(self.regression_fc(embeddings))


class Senti_Roberta(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.score_fc = nn.Linear(768, 11)
        self.regression_fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        bert_tokens = self.tokenizer.batch_encode_plus(text,
                                                       padding='longest', return_tensors='pt')
        outputs = self.bert(bert_tokens['input_ids'][:, :256].cuda(),
                            attention_mask=bert_tokens['attention_mask'][:, :256].cuda())
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return self.score_fc(embeddings), self.sigmoid(self.regression_fc(embeddings))

class Senti_Albert(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = AlbertModel.from_pretrained('albert-base-v2')
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.score_fc = nn.Linear(768, 11)
        self.regression_fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        bert_tokens = self.tokenizer.batch_encode_plus(text,
                                                       padding='longest', return_tensors='pt')
        outputs = self.bert(bert_tokens['input_ids'][:, :256].cuda(),
                            attention_mask=bert_tokens['attention_mask'][:, :256].cuda())
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return self.score_fc(embeddings), self.sigmoid(self.regression_fc(embeddings))

# class Senti_XLNet(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.bert = XLNetModel.from_pretrained('xlnet-base-cased')
#         self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
#         self.score_fc = nn.Linear(768, 11)
#         self.regression_fc = nn.Linear(768, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, text):
#         bert_tokens = self.tokenizer.batch_encode_plus(text,
#                                                        padding='longest', return_tensors='pt')
#         outputs = self.bert(bert_tokens['input_ids'][:, :256].cuda(),
#                             attention_mask=bert_tokens['attention_mask'][:, :256].cuda())
#         embeddings = torch.mean(outputs.last_hidden_state, dim=1)
#         return self.score_fc(embeddings), self.sigmoid(self.regression_fc(embeddings))




