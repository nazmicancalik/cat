import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F 

from utils.config import get_config

from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer

class LightningTransformer(pl.LightningModule):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.modules.transformer.embed_dim
        self.depth = config.modules.transformer.encoder.depth

        self.encoder_layer = TransformerEncoderLayer(d_model = self.embed_dim, nhead=8)
        self.encoder = TransformerEncoder(self.encoder_layer,num_layers = self.depth)

    def forward(self,x,knn_index=None):
        batch_size = x.size(0)
        x = self.encoder(x)
        x_avg = F.adaptive_avg_pool1d(x,1).view(batch_size,-1)
        x_max = F.adaptive_max_pool1d(x,1).view(batch_size,-1)
        x = torch.cat((x_avg, x_max), 1)
        return x