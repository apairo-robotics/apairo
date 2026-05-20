
from torch import nn


class RotError(Exception):
    def __init__(self) -> None:
        pass


class MSECoords:
    def __init__(self) -> None:
        self.mse = nn.MSELoss(reduction='mean')

    def __call__(self, pred, gt):
        return self.mse(pred['coord'], gt['coord'])


class Losses:
    def __init__(self, loss_list) -> None:
        self.loss_list = loss_list
        self.losses = self.build_losses()

    def build_losses(self):
        losses = []
        for loss, weight in self.loss_list:
            losses.append((loss(), weight))
        return losses

    def __call__(self, preds, gt):
        return sum([weight * loss(preds, gt) for loss, weight in self.losses])
