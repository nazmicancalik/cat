'''
Code taken from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py Official pytorch implementation of DGCNN.
Courtesy of Yue Wang.
'''

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.utils import farthest_point_sampling


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    # Returns the top k element of the given tensor in the given dimension. In this case -1 so the k values.
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, device='cpu', idx=None):
    batch_size, num_dims, num_points = x.size()

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature
    
class LightningDGCNNFeatureExtractor(pl.LightningModule):
    def __init__(self, config):
        super(LightningDGCNNFeatureExtractor, self).__init__()
        self.k = config.modules.DGCNN.k
        self.embed_dim = config.modules.DGCNN.embed_dim
        self.dropout = config.modules.DGCNN.dropout
        self.input_channels = config.datasets.scannet.input_channels
        self.num_points = config.modules.DGCNN.num_in_points
        self.seq_len = config.modules.transformer.seq_len

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.embed_dim)

        # This 6 has to change if we add the normals to the input.
        self.conv1 = nn.Sequential(nn.Conv2d(self.input_channels*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256*2, self.embed_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        coor = x
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k, device=self.device)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k, device=self.device)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k, device=self.device)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k, device=self.device)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        x = self.conv5(x)
        
        # Pick the furthest points' features to map it to seq_len
        fps_index = farthest_point_sampling(coor.transpose(-1,-2), self.seq_len)
        coor = coor.transpose(-1,-2)
        x = x.transpose(-1,-2).contiguous()
        coor = coor[np.arange(batch_size)[:, None], fps_index, :] # downsampled point locations
        x = x[np.arange(batch_size)[:, None], fps_index, :] # downsampled point features
        return coor.transpose(-1,-2), x.transpose(-1,-2).contiguous()
