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


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    # Returns the top k element of the given tensor in the given dimension. In this case -1 so the k values.
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def _fps(points,num_group):
    """
    Input:
        points: pointcloud data, [N, D]
        num_group: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = points.shape
    xyz = points[:, :3]
    centroids = np.zeros((num_group,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(num_group):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids.astype(np.int32)
def fps_downsample(coor, x, num_group):
    xyz = coor.transpose(1, 2).contiguous() # b, n, 3
    fps_idx = _fps(xyz, num_group)

    combined_x = torch.cat([coor, x], dim=1)

    new_combined_x = (
        pointnet2_utils.gather_operation(
            combined_x, fps_idx
        )
    )

    new_coor = new_combined_x[:, :3]
    new_x = new_combined_x[:, 3:]

    return new_coor, new_x

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
        #self.linear_projection_layer = nn.Conv2d(self.num_points,self.seq_len,kernel_size=1, bias=False)                  
        #self.linear_projection_layer = nn.Conv1d(self.embed_dim*self.num_points, self.embed_dim*self.seq_len, kernel_size=1)
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
        print(f"x is {x.size()}")
        x = self.conv5(x)

        x1_ = F.adaptive_max_pool1d(x, self.seq_len).view(batch_size, self.embed_dim,-1)
        x2_ = F.adaptive_avg_pool1d(x, self.seq_len).view(batch_size, self.embed_dim,-1)
        x = torch.cat((x1_, x2_), 1)
        x = F.adaptive_avg_pool1d(x, self.seq_len).view(batch_size, self.embed_dim,-1)
        """
        print("\n\n\n x.view size  shape : ", x.view(batch_size,-1).shape)
        x = self.linear_projection_layer(x.view(batch_size,-1).unsqueeze(-1))
        x = x.reshape(batch_size,self.embed_dim,self.seq_len)
        """
        #print("\n\n\n output of dgcnn shape : ", x.shape)
        # x = torch.cat((x1, x2, x3, x4), dim=1)

        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1) # Concatanate the result
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        return coor, x
