from torch.optim.lr_scheduler import LambdaLR
import numpy as np


class CosineLR(LambdaLR):

    def __init__(self, optimizer, lr, num_epochs, offset=1):
        self.init_lr = lr
        fn = lambda epoch: lr * 0.5 * (1 + np.cos((epoch - offset) / num_epochs * np.pi))
        super().__init__(optimizer, lr_lambda=fn)

    def reset(self, epoch, num_epochs):
        self.__init__(self.optimizer, self.init_lr, num_epochs, offset=epoch)
