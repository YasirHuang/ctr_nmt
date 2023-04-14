# coding=utf-8
'''
@author: Xin Huang
@contact: xin.huang@nlpr.ia.ac.cn
@file:loss_functions.py
@time:2020-07-30 
@desc:
'''
import torch
import torch.nn as nn

class CosineSimilarity(nn.Module):
    def __init__(self, margin, reduction="mean"):
        super(CosineSimilarity, self).__init__()
        self.margin = margin

        assert reduction in ["mean", "sum", "none", None]
        self.reduction = reduction
        if reduction == "mean":
            self._reduction_fn = torch.mean
        elif reduction == "sum":
            self._reduction_fn = torch.sum
        elif reduction == "none" or reduction is None:
            self._reduction_fn = lambda x:x

    def forward(self, logits, labels):
        def l2_normalize(vector, dim=-1):
            vec_norm = torch.norm(vector, dim=1, keepdim=True)
            return vector / vec_norm

        logits_norm = l2_normalize(logits, dim=1)
        labels_norm = l2_normalize(labels, dim=1)
        cos_mat = logits_norm.mm(labels_norm.t())
        diag_part = torch.diag(cos_mat)
        diag_mat = torch.diag(diag_part)

        cost_s = self.margin - cos_mat + diag_mat
        cost_s = cost_s.masked_fill(torch.le(cost_s, 0.0), value=0.0)

        cost_i = self.margin - cos_mat + diag_part
        cost_i = cost_i.masked_fill(torch.le(cost_i, 0.0), value=0.0)

        # cost_s = torch.max(torch.Tensor([0.0, self.margin - cos_mat + diag_mat]))
        # cost_i = torch.max(torch.Tensor([0.0, self.margin - cos_mat + diag_part]))
        cost_tot = cost_s + cost_i - diag_mat
        # loss = torch.mean(cost_tot)
        # return loss
        return self._reduction_fn(cost_tot)

class LabelSmoothing(nn.Module):
    '''

    '''

    def __init__(self, smoothing=0.0):
        '''

        :param smoothing: default 0.0 which is a pure KLDivLoss
        '''
        super(LabelSmoothing, self).__init__()

        self.smoothing = smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, output, label):
        '''

        :param output: (batch_size, sequence_length, class_num/vocab_size) or
                        a packed sequence(sum(sequence_length), class_num/vocab_size).
        :param label: (batch_size, sequence_length) or a packed sequence(sum(sequence_length))
        :return: KLDivLoss
        '''
        class_size = output.shape[-1]
        device = output.device

        confidence = 1.0 - self.smoothing
        smoothing_value = self.smoothing / (class_size - 2)

        true_dist = torch.full_like(output, smoothing_value).to(device)
        true_dist.scatter_(-1, label.unsqueeze(-1), confidence)
        return self.criterion(output, true_dist)
