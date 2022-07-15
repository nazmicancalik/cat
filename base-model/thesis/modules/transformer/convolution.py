import torch
import torch.nn as nn


class ConvolutionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config.modules.transformer.convolution
        self.dim = config.modules.transformer.seq_len
        self.input_channels = config.modules.transformer.embed_dim
        self.kernel_size = self.config.kernel_size
        self.expansion_factor = self.config.expansion_factor
        self.dropout = self.config.dropout

        assert (self.kernel_size -
                1) % 2 == 0, "kernel size should be an odd numbe for 'SAME' "
        assert self.expansion_factor == 2, "Currently only 2 is supported as expansion factor"

        self.layer_norm = nn.LayerNorm(self.input_channels)
        self.pointwise_conv_1 = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.input_channels * self.expansion_factor,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.input_channels,
            kernel_size=self.kernel_size,
            groups=self.input_channels,
            stride=1,
            padding=(self.kernel_size-1) // 2,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(self.input_channels)
        self.pointwise_conv_2 = nn.Conv1d(
            in_channels=self.input_channels,
            out_channels=self.input_channels * self.expansion_factor,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1,2)
        x = self.pointwise_conv_1(x)
        # apply GLU activation
        outputs, gate = x.chunk(2, dim=1)
        x = outputs * gate.sigmoid()
        x = self.batch_norm(self.depthwise_conv(x))
        # apply swish
        x = x * x.sigmoid()
        x = self.dropout(self.pointwise_conv_2(x))
        return x
