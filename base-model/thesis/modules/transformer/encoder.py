import torch
import torch.nn as nn
import pytorch_lightning as pl

# from modules.transformer.misc import ResidualConnectionModule
from modules.transformer.attention import LinearSelfAttention, SelfAttention
from modules.transformer.convolution import ConvolutionBlock

# This graph feature extraction for the transformer.


def get_graph_feature(x, k=20, knn_index=None, x_q=None):
    # x: bs, np, c, knn_index: bs*k*np
    batch_size, num_points, num_dims = x.size()
    num_query = x_q.size(1) if x_q is not None else num_points
    feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
    feature = feature.view(batch_size, k, num_query, num_dims)
    x = x_q if x_q is not None else x
    x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1)
    feature = torch.cat((feature - x, x), dim=-1)
    return feature  # b k np c


class TransformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.depth = config.modules.transformer.encoder.depth
        self.encoder = nn.ModuleList(
            [EncoderBlock(config) for i in range(self.depth)])

    def forward(self, x):
        for i, blk in enumerate(self.encoder):
            if i < self.knn_layer:
                x = blk(x, knn_index)   # B N C
            else:
                x = blk(x)

        return self.encoder(x)


class EncoderBlock(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.dim = config.modules.transformer.embed_dim
        self.seq_len = config.modules.transformer.seq_len
        self.attn_drop = config.modules.transformer.encoder.attn_drop
        self.dropout = config.modules.transformer.encoder.dropout
        self.k = config.modules.DGCNN.k
        self.num_heads = config.modules.transformer.num_heads
        self.dim_head = config.modules.transformer.dim_head
        self.one_kv_head = config.modules.transformer.encoder.one_kv_head
        self.share_kv = config.modules.transformer.encoder.share_kv
        self.fc_hidden_ratio = config.modules.transformer.encoder.fc_hidden_ratio
        self.expansion_factor = config.modules.transformer.convolution.expansion_factor

        self.layer_norm_1 = nn.LayerNorm(self.dim)
        self.attention = SelfAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            attn_drop=self.attn_drop,
            proj_drop=self.dropout
        )

        self.dropout_layer = nn.Dropout(p=self.dropout)
        # Because the convolution block upscales the features size by the expansion factor,
        # the layer norm should be updated accordingly to the new size
        if config.modules.transformer.convolution.enabled:
            self.layer_norm_2 = nn.LayerNorm(self.dim * self.expansion_factor)
        else:
            self.layer_norm_2 = nn.LayerNorm(self.dim)
            self.expansion_factor = 1  # needed for fc after conv
        
        self.knn_map = nn.Sequential(
            nn.Linear(self.dim*2, self.dim),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.merge_map = nn.Linear(self.dim*2, self.dim)
        fc_hidden_dim = int(self.dim*self.fc_hidden_ratio)
        self.convolution_block = ConvolutionBlock(config)
        self.fc = nn.Sequential(
            nn.Linear(self.dim * self.expansion_factor, fc_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(fc_hidden_dim, self.dim),
            nn.Dropout(p=self.dropout)
        )

    def forward(self, x, knn_index=None):
        normalized_x = self.layer_norm_1(x)
        # print('\n\n encoder after layer norm 1 x :', normalized_x.size(), '\n\n')
        x_1 = self.attention(normalized_x)
        #print('\n\n encoder after attention x :',x_1.size(), '\n\n')
        if knn_index is not None:
            knn_features = get_graph_feature(
                normalized_x, k=self.k, knn_index=knn_index)
            knn_features = self.knn_map(knn_features)
            knn_features = knn_features.max(dim=1, keepdim=False)[0]
            x_1 = torch.cat([x_1, knn_features], dim=2)
            x_1 = self.merge_map(x_1)
        #print('\n\n encoder after knn x :',x_1.size(), '\n\n')

        x = x + self.dropout_layer(x_1)
        #print('\n\n encoder after dropout and residual sum x :', x.size(), '\n\n')

        if self.config.modules.transformer.convolution.enabled:
            x = self.convolution_block(x).transpose(-1, -2)  # Conformer addition
        
        #print('\n\n encoder after first dropout block x :',x.size(), '\n\n')
        x = self.fc(self.layer_norm_2(x))
        x = x + self.dropout_layer(x)
        #print('\n\n encoder output x :',x.size(), '\n\n')
        return x
