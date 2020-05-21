import math

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class PolyLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group according to poly learning rate policy
    """

    def __init__(self, optimizer, max_iter=90000, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * (1 - float(self.last_epoch) / self.max_iter) ** self.power
            for base_lr in self.base_lrs
        ]


func_zoo = {
    "cosine_decay": lambda epoch, step, len_epoch, total_epoch: 0.5
    * (math.cos(step * math.pi / (total_epoch * len_epoch)) + 1)
}


class CosineWarmRestart:
    def __init__(
        self,
        optimizer: Optimizer,
        func: str = "cosine_decay",
        warmup: bool = True,
        warmup_epoch: int = 1,
        period: int = 10,
        min_lr: float = 1e-5,
        low_epoch: int = 1,
    ):
        self.base_lrs = [x["lr"] for x in optimizer.param_groups][0]
        self.optimizer = optimizer
        self.warmup = warmup
        self.warmup_epoch = warmup_epoch
        self.period = period
        self.cos_period = period - low_epoch
        self.low_epoch = low_epoch
        self.lr_func = func_zoo[func]
        self.min_lr = min_lr

    def cosine_step(
        self, current_epoch: int, global_step: int, len_epoch: int
    ) -> float:
        if self.warmup and current_epoch < self.warmup_epoch:
            lr = (
                self.base_lrs * float(1 + global_step) / (self.warmup_epoch * len_epoch)
            )
        else:
            lr = self.base_lrs * self.lr_func(
                current_epoch, global_step, len_epoch, self.cos_period
            )
        lr = max(self.min_lr, lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def step(self, current_epoch: int, global_step: int, len_epoch: int) -> float:
        current_epoch = current_epoch % self.period
        if current_epoch >= self.period - self.low_epoch:
            global_step = len_epoch * self.cos_period
        else:
            global_step = global_step % (self.period * len_epoch)
        return self.cosine_step(current_epoch, global_step, len_epoch)
