import torch
import pytorch_lightning as pl
import torch.nn as nn

from utils.config import get_config
from modules.transformer.encoder import TransformerEncoder
from modules.transformer.decoder import TransformerDecoder


class LightningTransformer(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        # Config
        self.config = config
        self.num_knn_layer = config.modules.transformer.num_knn_layer
        self.embed_dim = config.modules.transformer.embed_dim

        # Submodule pieces
        self.encoder = TransformerEncoder(config).encoder
        self.decoder = TransformerDecoder(config).decoder

        # Used between encoder and decoder.
        self.encoder_to_decoder_map = nn.Sequential(
            nn.Conv1d(self.embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, self.embed_dim, 1)
        )

        # TODO: What to give as q to the decoder is a matter of question now.
        # Let's for now only give the output of the encoder together mapped with the original point cloud.
        # In pointr they deal with it differently
    def forward(self, x, knn_index=None):
        # Pass through the encoder
        #print('\n\n transformer input x :',x.size(), '\n\n')
        for i, blk in enumerate(self.encoder):
            if i < self.num_knn_layer:
                x = blk(x, knn_index)
            else:
                x = blk(x)
        #print('\n\n encoder output x :',x.size(), '\n\n')

        # Apply projection and get the global feature
        global_feature = self.encoder_to_decoder_map(
            x.transpose(1, 2))  # B 1024 N
        #print('\n\n global feature size after encoder to decoder map :',global_feature.size(), '\n\n')

        # global_feature = torch.max(global_feature, dim=-1)[0]  # B 1024
        #print('\n\n global feature size after torch max :',global_feature.size(), '\n\n')

        # Pass through the decoder
        q = self.mlp_query(global_feature).transpose(1, 2)  # B M C
        #print('\n\n query size after mlp_query:',q.size(), '\n\n')

        for i, blk in enumerate(self.decoder):
            if i < self.num_knn_layer:
                q = blk(q, x, self_knn_index=knn_index)
            else:
                q = blk(q, x)
        #print('\n\n q size after the decoder blocks :', q.size(), '\n\n')
        return q
