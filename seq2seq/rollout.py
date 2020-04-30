import os
import random
import math
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from seq2seq.model import Model


class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model: Model, update_rate):
        self.lstm = model
        self.update_rate = update_rate

    def get_reward(self, x: torch.LongTensor, num, commands_input, commands_lengths, 
                        situations_input, target_batch, sos_idx, 
                        eos_idx,  reward_func):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        x = F.log_softmax(x, dim=-1).max(dim=-1)[1].detach()[:,1:]
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                data = x[:, 0:l]
                samples = self.lstm.sample(batch_size, seq_len, commands_input, commands_lengths, 
                                        situations_input, target_batch, sos_idx, 
                                        eos_idx, data)
                pred = reward_func(samples)
                # pred = pred.cpu().data[:,1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = reward_func(x)
            # pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
