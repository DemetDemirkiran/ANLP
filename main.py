from senti_bert import Senti_Bert, Senti_DistilBert, Senti_Roberta, Senti_Albert
from splitter import TextLoader, SCORE_TO_CLASS
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from multitask_loss import MultitaskLoss
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import numpy as np
import os
import yaml
from glob import glob

MODEL_DICT = {'bert': Senti_Bert,
              'distil': Senti_DistilBert,
              'roberta': Senti_Roberta,
              'albert': Senti_Albert,
              }


def train(model, dataloader, writer, output_path, start_epoch=0, num_epochs=10, lr=1e-4):
    model = model.cuda()
    loss_function = MultitaskLoss().cuda()
    adam = Adam(model.parameters(), lr=lr)

    iter = 0
    for epoch in range(start_epoch, num_epochs):
        progress_bar = tqdm(total=len(dataloader), leave=False,
                            desc="Training Epoch {}".format(epoch),
                            mininterval=1, maxinterval=100)
        for data_dict in dataloader:
            train = data_dict['text']
            class_target = data_dict['score'].cuda()
            reg_target = data_dict['usefulness'].cuda()
            class_pred, reg_pred = model(train)

            adam.zero_grad()
            loss = loss_function(class_pred, class_target, reg_pred, reg_target)
            loss['total_loss'].backward()
            adam.step()

            writer.add_scalar('Total loss', loss['total_loss'].data.cpu(), iter)
            writer.add_scalar('Xent loss', loss['xent'].data.cpu(), iter)

            accuracy, l1 = evaluation_metrics(class_pred, class_target, reg_pred, reg_target)
            writer.add_scalar('Accuracy',
                              torch.tensor(accuracy).mean(),
                              iter)
            writer.add_scalar('L1 error',
                              torch.tensor(l1).mean(),
                              iter)

            iter += 1
            progress_bar.update()
        torch.save(model.state_dict(), os.path.join(output_path, 'ckpt{}.pth'.format(epoch)))


def test(model, dataloader):
    model = model.cuda()
    model.eval()
    final_acc = []
    final_sent = []
    final_l1 = []
    with torch.no_grad():
        for data_dict in tqdm(dataloader):
            train = data_dict['text']
            class_target = data_dict['score'].cuda()
            reg_target = data_dict['usefulness'].cuda()
            class_pred, reg_pred = model(train)
            accuracy, sentiment, l1_error = evaluation_metrics(class_pred, class_target, reg_pred, reg_target)
            final_acc.append(accuracy)
            final_sent.append(sentiment)
            final_l1.append(l1_error)

    final_acc = torch.mean(torch.tensor(final_acc))
    final_sent = torch.mean(torch.tensor(final_sent))
    final_l1 = torch.mean(torch.tensor([f2 for f1 in final_l1 for f2 in f1]))
    print('ACCURACY: {:.2f} SENTIMENT ACC: {:.2f} L1_ERROR: {:.2f}'.format(final_acc * 100.0, final_sent * 100.0,
                                                                           final_l1))

    return final_acc * 100.0, final_sent * 100.0, final_l1


def evaluation_metrics(class_pred, class_target, reg_pred, reg_target):
    with torch.no_grad():
        # Classes go in reverse order so 0 = best score 11 = worst

        softmax = torch.nn.Softmax(1)
        class_pred = torch.argmax(softmax(class_pred), dim=1)
        sentiment_pred = class_pred <= 5
        sentiment_target = class_target <= 5
        accuracy = accuracy_score(class_target.cpu().numpy(), class_pred.cpu().numpy())
        sentiment_accuracy = accuracy_score(sentiment_pred.cpu().numpy(), sentiment_target.cpu().numpy())
        l1_error = torch.abs(reg_pred.squeeze() - reg_target).cpu().numpy()

    return accuracy, sentiment_accuracy, l1_error


def trends(model, dataloader):
    class_score = {v: float(k) for k, v in SCORE_TO_CLASS.items()}
    product_dict = dict()
    model = model.cuda()
    model.eval()
    softmax = torch.nn.Softmax(1)

    with torch.no_grad():
        for data_dict in tqdm(dataloader):
            id = data_dict['id'][0]
            name = data_dict['name'][0]

            # Initialize dict key
            if id not in product_dict.keys():
                product_dict[id] = dict()
                product_dict[id]['name'] = name
                product_dict[id]['score_pred'] = []
                product_dict[id]['score_true'] = []

            train = data_dict['text']
            class_target = data_dict['score'].cuda()
            class_pred, reg_pred = model(train)
            score_pred = torch.argmax(softmax(class_pred))

            product_dict[id]['score_pred'].append(score_pred.cpu().numpy())
            product_dict[id]['score_true'].append(class_target[0].cpu().numpy())

    true_scores = []
    pred_scores = []

    for k in product_dict:
        true_scores.append(np.mean([class_score[int(sc)] for sc in product_dict[k]['score_true']]))
        pred_scores.append(np.mean([class_score[int(sc)] for sc in product_dict[k]['score_pred']]))

    # Sort in descending order
    best_rated_true = np.array(list(product_dict.keys()))[np.argsort(true_scores)[::-1]]
    best_rated_pred = np.array(list(product_dict.keys()))[np.argsort(pred_scores)[::-1]]

    best_rated_true = best_rated_true[:100]
    best_rated_pred = best_rated_pred[:100]
    overlap = set(best_rated_true).intersection(best_rated_pred)

    print("Overlap in 100-best products: {}".format(len(overlap)))
    for o in overlap:
        print(' ID: {}'.format(o))


if __name__ == '__main__':
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')

    # reviews = read()
    # small = lowercase(reviews)
    # dictionary = text_to_dict(small)
    # punct = punctuation_and_spaces(dictionary)
    # toknz = tokenize(small) # do not need it because all of te tokenization, and preprocessing is now done in the
    # function punctuation_and_spaces, the function tokenize is now obsolete
    # lemmz = lemming(toknz) #Tested this, found that it just added
    # more spaces into the list that contains the details of the reviews, not really needed. CURRENTLY
    # stemmz = stemming(toknz) #does not do much at the moment cant tell a difference
    # dtaframe = list_to_dataframe(dictionary)
    # dtaframe = dataframe_to_csv(dtaframe)
    # print(dtaframe)

    # reviews = read()
    # small = lowercase(reviews)
    # dictionary = text_to_dict(small)
    # punct = punctuation_and_spaces(dictionary)
    # dtaframe = list_to_dataframe(punct)
    config_path = '/home/demet/PycharmProjects/ANLP/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    batch_size = config['training']['batch_size']
    lr = float(config['training']['lr'])
    epochs = config['training']['epochs']

    mode = config['mode']
    model = MODEL_DICT[config['model']['name']]()

    if mode == 'train':

        resume = config['training']['resume']
        if resume:
            out_path = config['output_path']
            ckpts = glob(os.path.join(out_path, '*.pth'))
            if len(ckpts) > 0:
                ckpts = ckpts[-1]
                print('Restoring from: {}'.format(ckpts))
                model.load_state_dict(torch.load(ckpts))
                start_epoch = os.path.split(ckpts)[1]
                start_epoch = start_epoch.split('ckpt')[1]
                start_epoch = int(start_epoch.split('.')[0]) + 1
            else:
                print('No checkpoint found!')
                start_epoch = 0

        output_path = config['output_path']

        tl = TextLoader(config['path_to_data'])
        dataloader = DataLoader(tl, batch_size=batch_size, shuffle=True)
        writer = SummaryWriter(os.path.join(output_path, 'summary'))

        if not resume:
            train(model, dataloader, writer, output_path, num_epochs=epochs, lr=lr)
        else:
            train(model, dataloader, writer, output_path, start_epoch=start_epoch, num_epochs=epochs, lr=lr)
    elif mode == 'test':
        model.load_state_dict(torch.load(config['test']['restore_from']))
        tl = TextLoader(config['path_to_data'], mode='test')
        test_dataloader = DataLoader(tl, batch_size=batch_size, shuffle=False)
        acc, senti, l1_err = test(model, test_dataloader)
    elif 'trend':
        model.load_state_dict(torch.load(config['test']['restore_from']))
        tl = TextLoader(config['path_to_data'], mode='test')
        test_dataloader = DataLoader(tl, batch_size=1, shuffle=False)
        trends(model, test_dataloader)
