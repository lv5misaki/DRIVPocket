import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-5, min_lr=1e-5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warm-up phase
            lr = (self.base_lrs[0] - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs + self.warmup_start_lr
            return [lr for _ in self.base_lrs]
        else:
            # Cosine Annealing phase
            cosine_epoch = self.last_epoch - self.warmup_epochs
            cosine_epochs = self.max_epochs - self.warmup_epochs
            lr = self.min_lr + (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * cosine_epoch / cosine_epochs)) / 2
            return [lr for _ in self.base_lrs]

def update_average(model, averaged_model, num_models):
    """
    更新平均模型的权重。
    :param model: 当前模型
    :param averaged_model: 平均模型
    :param num_models: 到目前为止考虑的模型数量
    """
    for param, avg_param in zip(model.parameters(), averaged_model.parameters()):
        avg_param.data.copy_(avg_param.data * num_models / (num_models + 1) + param.data / (num_models + 1))
