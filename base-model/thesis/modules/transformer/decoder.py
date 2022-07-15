import torch
import torch.nn as nn
import pytorch_lightning as pl

from modules.transformer.attention import LinearSelfAttention, CrossAttention
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


class TransformerDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.depth = config.modules.transformer.decoder.depth

        self.decoder = nn.ModuleList(
            [DecoderBlock(config) for i in range(self.depth)])

    def forward(self, q, v):
        return self.decoder(q, v)


class DecoderBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.dim = config.modules.transformer.embed_dim
        self.seq_len = config.modules.transformer.seq_len
        self.dropout = config.modules.transformer.decoder.dropout
        self.depth = config.modules.transformer.decoder.depth
        self.k = config.modules.transformer.decoder.k
        self.num_heads = config.modules.transformer.num_heads
        self.dim_head = config.modules.transformer.dim_head
        self.one_kv_head = config.modules.transformer.decoder.one_kv_head
        self.share_kv = config.modules.transformer.decoder.share_kv
        self.fc_hidden_ratio = config.modules.transformer.decoder.fc_hidden_ratio
        self.attn_drop = config.modules.transformer.decoder.attn_drop
        self.expansion_factor = config.modules.transformer.convolution.expansion_factor

        self.layer_norm_1 = nn.LayerNorm(self.dim)
        self.self_attention = LinearSelfAttention(
            dim=self.dim,
            seq_len=self.seq_len,
            k=self.k,
            heads=self.num_heads,
            dim_head=self.dim_head,
            one_kv_head=self.one_kv_head,
            share_kv=self.share_kv,
            dropout=self.dropout
        )

        self.norm_q = nn.LayerNorm(self.dim)
        self.norm_v = nn.LayerNorm(self.dim)
        self.cross_atention = CrossAttention(
            dim=self.dim,
            out_dim=self.dim,
            num_heads=self.num_heads,
            attn_drop=self.attn_drop,
            proj_drop=self.dropout
        )

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.convolution_block = ConvolutionBlock(config)

        self.layer_norm_2 = nn.LayerNorm(self.dim * self.expansion_factor)

        self.knn_map = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(self.dim*2, self.dim)

        self.knn_map_cross = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map_cross = nn.Linear(self.dim*2, self.dim)

        fc_hidden_dim = int(self.dim*self.fc_hidden_ratio)
        self.fc = nn.Sequential(
            nn.Linear(self.dim * self.expansion_factor, fc_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(fc_hidden_dim, self.dim),
            nn.Dropout(p=self.dropout)
        )

    def forward(self, q, v, self_knn_index=None, cross_knn_index=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        norm_q = self.layer_norm_1(q)
        q_1 = self.self_attention(norm_q)

        if self_knn_index is not None:
            knn_f = get_graph_feature(
                norm_q, device=torch.device('cuda:0'), knn_index=self_knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_1 = torch.cat([q_1, knn_f], dim=-1)
            q_1 = self.merge_map(q_1)

        q = q + self.dropout_layer(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.cross_atention(norm_q, norm_v)

        # I am not using the cross attention knn for now.
        if cross_knn_index is not None:
            knn_f = get_graph_feature(norm_v, cross_knn_index, norm_q)
            knn_f = self.knn_map_cross(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_2 = torch.cat([q_2, knn_f], dim=-1)
            q_2 = self.merge_map_cross(q_2)

        q = q + self.dropout_layer(q_2)

        q = self.convolution_block(q).transpose(-1, -2)
        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        q = self.fc(self.layer_norm_2(q))
        q = q + self.dropout_layer(q)
        return q
