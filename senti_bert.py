from transformers import BertTokenizer, BertModel
from transformers import InputExample, InputFeatures
import torch
from torch import nn

class Senti_Bert(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 11)

    def forward(self, text):
        bert_tokens = self.tokenizer.batch_encode_plus([text])
        outputs = self.bert(torch.tensor(bert_tokens['input_ids']), attention_mask=torch.tensor(bert_tokens['attention_mask']))
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        return self.fc(embeddings)




