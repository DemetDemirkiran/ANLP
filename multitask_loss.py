from torch import nn
from torch.nn import CrossEntropyLoss, L1Loss

class MultitaskLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.score_classifier = CrossEntropyLoss()
        self.regression = L1Loss()

    def forward(self, class_pred, class_true, use_pred, use_true):
        xent_loss = self.score_classifier(class_pred, class_true)
        reg_loss = self.regression(use_true.unsqueeze(1), use_pred)
        return {'total_loss': xent_loss+reg_loss,
                'xent': xent_loss,
                'regression': reg_loss}
