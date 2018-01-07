# -*- coding: utf-8 -*-

import os
import random

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F

def cos_similarity(gamma, y_ref, y_sent):
    return gamma * cosine_similarity(y_ref, y_sent)

class Discriminator(nn.Module):
    """A CNN for text classification

    architecture: Embedding >> Convolution >> Max-pooling >> Highway >> Paper's 
    """

    def __init__(self, num_classes, vocab_size, emb_dim, filter_sizes, num_filters, dropout, batch = 32, ref = 16):
        super(Discriminator, self).__init__()
        self.batch = batch
        self.ref = ref
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.convs_ref = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.highway_ref = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), num_classes)
        self.softmax = nn.LogSoftmax()
        self.init_parameters()
    
    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        gamma = 1.0                     # gamma value as mentioned in the paper for ranking
        emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        convs_ref = [F.relu(conv(emb)).squeeze(3) for conv in self.convs_ref]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs] # [batch_size * num_filter]
        pools_ref = [F.max_pool1d(conv_ref, conv_ref.size(2)).squeeze(2) for conv_ref in convs_ref]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        pred_ref = torch.cat(pools_ref, 1)
        highway = self.highway(pred)
        highway_ref = self.highway_ref(pred_ref)
        pred = F.sigmoid(highway) *  F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.softmax(self.lin(self.dropout(pred)))
        pred_ref = F.sigmoid(highway_ref) *  F.relu(highway_ref) + (1. - F.sigmoid(highway_ref)) * pred_ref
        pred_ref = self.softmax(self.lin(self.dropout(pred_ref)))
        values = []
        for i in range(self.batch):
            value = 0
            for j in range(self.ref):
                value += cos_similarity(1.0, pred_ref[j], pred[i])
            values.append(value)
        total = sum(values)
        for i in range(self.batch):
            values[i] = values[i] / total
        pred = torch.from_numpy(np.log(np.asarray(values)))
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
