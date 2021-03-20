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
    # with open(dataset, 'r') as f:
    #    text = f.readlines()

    # text = [t.split(',') for t in text[1:]]
    # print(data.head())
    # split data into labels and features
    # Labels are the data which we want to predict and features are the data which are used to predict labels.

    # product_id = data.productID
    # X = data.drop('product_id', axis=1)

    X_train, X_test = train_test_split(data.to_numpy(), test_size=0.2, random_state=2021)

    return X_train, X_test


class TextLoader(data.Dataset):

    def __init__(self, path, mode='train'):
        super(TextLoader, self).__init__()
        self.mode = mode
        self.path = path
        self.train_data, self.test_data = splitter(self.path)
        self.train_data = self.train_data
        self.test_data = self.test_data

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'test':
            return len(self.test_data)
        else:
            raise ValueError('Wrong mode')

    def __getitem__(self, item):

        if self.mode == 'train':
            text_list = self.train_data
        elif self.mode == 'test':
            text_list = self.test_data
        else:
            raise ValueError('Wrong mode')

        text = text_list[item].copy()
        text_review = text[10]
        text_score = text[7]
        text_usefulness = text[6]
        text_usefulness = text_usefulness.split('/')
        text_usefulness[0] = float(text_usefulness[0])
        text_usefulness[1] = float(text_usefulness[1])
        # Avoid division by 0
        if text_usefulness[1] == 0:
            text_usefulness = 0
        else:
            text_usefulness = text_usefulness[0] / text_usefulness[1]

        return {'text': text_review,
                'score': torch.tensor(SCORE_TO_CLASS[str(text_score)]),
                'usefulness': torch.tensor(text_usefulness),
                'id': text[1],
                'name': text[2]}


if __name__ == '__main__':

    # splitter('/home/demet/Desktop/ANLP_Project/review_dataframe.csv')
    tl = TextLoader('/home/demet/Desktop/review_dataframe_notoken.csv')

    for data in tl:
        ...
