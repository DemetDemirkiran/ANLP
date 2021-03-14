from senti_bert import Senti_Bert
from splitter import TextLoader
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from multitask_loss import MultitaskLoss
from torch.optim import Adam
from sklearn.metrics import accuracy_score

import os
import yaml

MODEL_DICT = {'bert': Senti_Bert}


def train(model, dataloader, writer, output_path, num_epochs=10, lr=1e-4):
    model = model.cuda()
    loss_function = MultitaskLoss().cuda()
    adam = Adam(model.parameters(), lr=lr)

    iter = 0
    for epoch in range(num_epochs):
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

    return model


def test(model, dataloader):
    model = model.cuda()
    model.eval()
    final_acc = []
    final_l1 = []
    with torch.no_grad():
        for data_dict in tqdm(dataloader):
            train = data_dict['text']
            class_target = data_dict['score'].cuda()
            reg_target = data_dict['usefulness'].cuda()
            class_pred, reg_pred = model(train)
            accuracy, l1_error = evaluation_metrics(class_pred, class_target, reg_pred, reg_target)
            final_acc.append(accuracy)
            final_l1.append(l1_error)

    final_acc = torch.mean(torch.tensor(final_acc))
    final_l1 = torch.mean(torch.tensor([f2 for f1 in final_l1 for f2 in f1]))
    print('ACCURACY: {:.2f} L1_ERROR: {:.2f}'.format(final_acc * 100.0, final_l1))


def evaluation_metrics(class_pred, class_target, reg_pred, reg_target):
    with torch.no_grad():
        softmax = torch.nn.Softmax(1)
        class_pred = torch.argmax(softmax(class_pred), dim=1)
        accuracy = accuracy_score(class_target.cpu().numpy(), class_pred.cpu().numpy())

        l1_error = torch.abs(reg_pred.squeeze() - reg_target).cpu().numpy()

    return accuracy, l1_error


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

    config_path = 'D:\\PycharmProjects\\ANLP\\ANLP\\config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)

    mode = config['mode']
    model = MODEL_DICT[config['model']['name']]()

    batch_size = config['training']['batch_size']
    lr = float(config['training']['lr'])
    epochs = config['training']['epochs']

    if mode == 'train':
        output_path = config['output_path']

        tl = TextLoader(config['path_to_data'])
        dataloader = DataLoader(tl, batch_size=batch_size, shuffle=True)
        writer = SummaryWriter(os.path.join(output_path, 'summary'))

        train(model, dataloader, writer, output_path, num_epochs=epochs, lr=lr)

    else:
        model.load_state_dict(torch.load(config['test']['restore_from']))
        tl = TextLoader(config['path_to_data'], mode='test')
        test_dataloader = DataLoader(tl, batch_size=batch_size, shuffle=False)
        test(model, test_dataloader)
