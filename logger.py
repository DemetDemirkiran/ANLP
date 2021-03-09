import os
from datetime import datetime


class Logger:
    def __init__(self):
        self.cwd = os.getcwd()
        self.log_path = os.path.join(self.cwd, 'logs')
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        now = datetime.now()
        self.id = '{}-{}-{}_{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
        self.full_dir = os.path.join(self.log_path, self.id)
        if not os.path.exists(self.full_dir):
            os.makedirs(self.full_dir)
        self.full_path = os.path.join(self.full_dir, 'log.txt')

    def write_log(self, epoch, bsz, learning, total_loss, total_accuracy, val=False):
        with open(self.full_path, 'a') as file:
            if epoch == 0:
                file.write(
                    "Hyper-parameter: Epoch: {} Batch size: {} Learning rate: {}  \n".format(epoch, bsz, learning))

            elif val == True:
                file.write(
                    "Epoch: {} Total Loss: {} Validation Accuracy: {} \n".format(epoch, total_loss, total_accuracy))
            else:
                file.write("Epoch: {} Total Loss: {} Total Accuracy: {} \n".format(epoch, total_loss, total_accuracy))