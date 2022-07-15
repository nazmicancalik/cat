import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from utils.config import get_config
from modules.transformer.encoder import EncoderBlock


class LightningTransformer(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        # Config
        self.config = config
        self.num_knn_layer = config.modules.transformer.num_knn_layer
        self.embed_dim = config.modules.transformer.embed_dim
        self.depth = config.modules.transformer.encoder.depth
        self.seq_len = config.modules.transformer.seq_len

        # Submodule pieces
        self.block_1 = EncoderBlock(config)
        self.block_2 = EncoderBlock(config)
        self.block_3 = EncoderBlock(config)
        self.block_4 = EncoderBlock(config)
        #self.block_5 = EncoderBlock(config)
        #self.block_6 = EncoderBlock(config)

        self.fc = nn.Sequential(
            nn.Conv1d(self.embed_dim*4, self.embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.embed_dim),
            nn.LeakyReLU(negative_slope=0.2))
        
    def forward(self, x, knn_index=None):
        # Only encoder is used.
        batch_size = x.size(0)

        x1 = self.block_1(x,knn_index)
        x2 = self.block_2(x1,knn_index)
        x3 = self.block_3(x2)
        x4 = self.block_4(x3)
        #x5 = self.block_5(x4)
        #x6 = self.block_6(x5)

        x = torch.cat((x1,x2,x3,x4),dim=2)#x5,x6),dim=2)
        #print("\n--------------------------------------\nConcatted transformer encoder Output: \n",x.size(),"\n--------------------------------------\n")

        x = self.fc(x.transpose(-1,-2))
        #print("\n--------------------------------------\nAnd after fc output: \n",x.size(),"\n--------------------------------------\n")
        x_avg = F.adaptive_avg_pool1d(x,1).view(batch_size,-1)
        x_max = F.adaptive_max_pool1d(x,1).view(batch_size,-1)
        x = torch.cat((x_avg, x_max), 1) # Concatanate the result
        #print("\n--------------------------------------\nTransformer Output: \n",x.size(),"\n--------------------------------------\n")
        return x