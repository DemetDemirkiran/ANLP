from transformers import pipeline, BertModel, BertTokenizer, BertForSequenceClassification, \
    DistilBertForSequenceClassification, DistilBertTokenizer, \
    AlbertForSequenceClassification, AlbertTokenizer, \
    RobertaForSequenceClassification, RobertaTokenizer
from splitter import TextLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def sentiment_analysis(model_type, data_path):
    if model_type == 'albert':
        model = AlbertForSequenceClassification.from_pretrained("textattack/albert-base-v2-SST-2")
        tokenizer = AlbertTokenizer.from_pretrained("textattack/albert-base-v2-SST-2")

    elif model_type == 'bert':
        model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
        tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")

    elif model_type == 'distil':
        model = DistilBertForSequenceClassification.from_pretrained("textattack/distilbert-base-cased-SST-2")
        tokenizer = DistilBertTokenizer.from_pretrained("textattack/distilbert-base-cased-SST-2")

    elif model_type == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2")
        tokenizer = RobertaTokenizer.from_pretrained("textattack/roberta-base-SST-2")

    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    tl = TextLoader(data_path)
    ground_truth = list()
    review_predictions = list()
    label_dict = {'LABEL_0': 0, 'LABEL_1': 1}

    for data in tqdm(tl):
        text = data['text']
        score = data['score']
        score = score >= 2.5
        result = nlp(text, truncation=True)
        prediction = label_dict[result[0]['label']]
        ground_truth.append(score.cpu().numpy())
        review_predictions.append(prediction)

    accuracy = accuracy_score(ground_truth, review_predictions)
    print('ACCURACY: ', accuracy)

    return accuracy

    # result = nlp(List[tl])[0]
    # print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

    # result = nlp("I love you")[0]
    # print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


if __name__ == '__main__':
    sentiment_analysis('bert', '/home/demet/Desktop/review_dataframe_notoken.csv')
    ...
