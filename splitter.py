import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils import data
import torch

SCORE_TO_CLASS = {
    '0.0': 10,
    '0.5': 9,
    '1.0': 8,
    '1.5': 7,
    '2.0': 6,
    '2.5': 5,
    '3.0': 4,
    '3.5': 3,
    '4.0': 2,
    '4.5': 1,
    '5.0': 0
}

def splitter(dataset):

    data = pd.read_csv(dataset)
    with open(dataset, 'r') as f:
        text = f.readlines()

    text = [t.split(',') for t in text[1:]]
     #print(data.head())
    # split data into labels and features
    # Labels are the data which we want to predict and features are the data which are used to predict labels.

    #product_id = data.productID
    #X = data.drop('product_id', axis=1)

    X_train, X_test= train_test_split(data, test_size=0.2)
    # print("\nX_train:\n")
    # print(X_train.head())
    # print(X_train.shape)

    # print("\nX_test:\n")
    # print(X_test.head())
    # print(X_test.shape)

    return X_train, X_test

class TextLoader(data.Dataset):

    def __init__(self, path, mode='train'):
        super(TextLoader, self).__init__()
        self.mode = mode
        self.path = path
        self.train_data, self.test_data = splitter(self.path)
        self.train_data = self.train_data.to_numpy()
        self.test_data = self.test_data.to_numpy()

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data) #// 50
        elif self.mode == 'test':
            return len(self.test_data) #// 50
        else:
            raise ValueError('Wrong mode')

    def __getitem__(self, item):

        if self.mode == 'train':
            text_list = self.train_data
        elif self.mode == 'test':
            text_list = self.test_data
        else:
            raise ValueError('Wrong mode')

        text = text_list[item]
        text_review = text[10]
        text_score = text[7]

        return text_review[:512], torch.tensor(SCORE_TO_CLASS[str(text_score)])


if __name__ == '__main__':

    # splitter('/home/demet/Desktop/ANLP_Project/review_dataframe.csv')
    tl = TextLoader('/home/demet/Desktop/review_dataframe_notoken.csv')

    for data in tl:
        ...
