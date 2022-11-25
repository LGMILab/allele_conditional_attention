import math
import torch
from torch import nn
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# class LearningRateScheduler(object):
class LearningRateScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr):
        """
        Base class for learning rate schedulers

        :param optimizer: Pytorch optimizer. [Optimizer]
        :param lr: Learning rate. [float]
        """
        self.optimizer = optimizer
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']


class TriStageLRScheduler(LearningRateScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 init_lr: float,
                 peak_lr: float,
                 final_lr: float,
                 init_lr_scale: float,
                 final_lr_scale: float,
                 warmup_steps: int,
                 hold_steps: int,
                 decay_steps: int,
                 total_steps: int):
        """
        Tri-Stage Learning Rate Scheduler.
        :param optimizer: Optimizer. [Optimizer]
        :param init_lr: Initial learning rate. [float]
        :param peak_lr: Maximum learning rate. [float]
        :param final_lr: Final learning rate. [float]
        :param init_lr_scale: Initial learning rate scale. [float]
        :param final_lr_scale: Final learning rate scale. [float]
        :param warmup_steps: Warmup the learning rate linearly for the first N updates. [int]
        :param hold_steps: Hold the learning rate for the N updates. [int]
        :param decay_steps: Decay the learning rate linearly for the first N updates. [int]
        :param total_steps: Total steps in training. [int]
        """
        assert isinstance(warmup_steps, int), "warmup_steps should be integer type"
        assert isinstance(total_steps, int), "total_steps should be integer type"

        super(TriStageLRScheduler, self).__init__(optimizer, init_lr)
        self.init_lr = init_lr
        self.init_lr *= init_lr_scale
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.decay_steps = decay_steps

        self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_steps = 0

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        offset = self.warmup_steps

        if self.update_steps < offset + self.hold_steps:
            return 1, self.update_steps - offset

        offset += self.hold_steps

        if self.update_steps <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_steps - offset

        offset += self.decay_steps

        return 3, self.update_steps - offset

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_steps += 1

        return self.lr


class InverseSQRTLRScheduler(LearningRateScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 init_lr: float,
                 peak_lr: float,
                 final_lr: float,
                 init_lr_scale: float,
                 final_lr_scale: float,
                 warmup_steps: int,
                 decay_steps: int,
                 total_steps: int):
        """
        Tri-Stage Learning Rate Scheduler.
        :param optimizer: Optimizer. [Optimizer]
        :param init_lr: Initial learning rate. [float]
        :param peak_lr: Maximum learning rate. [float]
        :param final_lr: Final learning rate. [float]
        :param init_lr_scale: Initial learning rate scale. [float]
        :param final_lr_scale: Final learning rate scale. [float]
        :param warmup_steps: Warmup the learning rate linearly for the first N updates. [int]
        :param hold_steps: Hold the learning rate for the N updates. [int]
        :param decay_steps: Decay the learning rate linearly for the first N updates. [int]
        :param total_steps: Total steps in training. [int]
        """
        assert isinstance(warmup_steps, int), "warmup_steps should be integer type"
        assert isinstance(total_steps, int), "total_steps should be integer type"

        super(InverseSQRTLRScheduler, self).__init__(optimizer, init_lr)
        self.init_lr = init_lr
        self.init_lr *= init_lr_scale
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        # self.hold_steps = hold_steps
        self.decay_steps = decay_steps

        self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        # self.decay_factor = -math.log(final_lr_scale) / self.decay_steps
        self.decay_factor = self.peak_lr * self.warmup_steps ** 0.5

        self.lr = self.init_lr
        self.update_steps = 0

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        offset = self.warmup_steps

        # if self.update_steps < offset + self.hold_steps:
        #     return 1, self.update_steps - offset

        # offset += self.hold_steps

        if self.update_steps <= offset + self.decay_steps:
            # decay stage
            return 1, self.update_steps - offset

        offset += self.decay_steps

        return 2, self.update_steps - offset

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        # elif stage == 1:
        #     self.lr = self.peak_lr
        elif stage == 1:
            # self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
            self.lr = self.decay_factor * self.update_steps ** -0.5
        elif stage == 2:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_steps += 1

        return self.lr

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, 
    num_cycles: float = 0.5, last_epoch: int = -1, min_lr: float = 0.0
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

