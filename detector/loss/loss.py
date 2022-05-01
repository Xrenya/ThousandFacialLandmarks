import torch.nn.functional as F


def MSE():
    def mse_loss(input, targets):
        return F.mse_loss(input, targets)
    return mse_loss
