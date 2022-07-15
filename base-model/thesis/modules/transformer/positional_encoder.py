import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LightningPositionalEncoder(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_channels = config.datasets.scannet.input_channels
        self.hidden_dim = config.modules.transformer.pos_encoder.hidden_dim
        self.embed_dim = config.modules.transformer.embed_dim

        # Can actually make this network one layer shorter thereby 1 hidden dim.
        # TODO: Should there be bias in these linear layers?
        self.conv1 = nn.Conv1d(self.input_channels, self.hidden_dim, 1)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
        
        self.conv2 = nn.Conv1d(self.hidden_dim, self.embed_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batch_norm(self.conv1(x))
        x = self.conv2(self.relu(x))
        return x
