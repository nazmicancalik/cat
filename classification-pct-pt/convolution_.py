import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
'''
class ConvolutionBlock(nn.Module):
    def __init__(self,dim,kernel_size,expansion_factor) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.expansion_factor= expansion_factor

        self.layer_norm = nn.LayerNorm(dim)

        self.pointwise_conv_1 = nn.Conv1d(
            in_channels = self.dim,
            out_channels = self.dim * self.expansion_factor,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = True
        )

        self.depthwise_conv = nn.Conv1d(
            in_channels=self.dim,
            out_channels = self.dim,
            kernel_size = self.kernel_size,
            groups=self.dim,
            stride=1,
            padding=(self.kernel_size-1) // 2,
            bias=False
        )

        self.batch_norm = nn.BatchNorm1d(self.dim)
        self.pointwise_conv_2 = nn.Conv1d(
            in_channels= self.dim,
            out_channels = self.dim * self.expansion_factor,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.dropout = nn.Dropout(p=0.5)
        self.glu = nn.GLU()

    def forward(self,x):
        x = self.layer_norm(x)
        x = self.pointwise_conv_1(x)
        
        # apply GLU activation
        outputs, gate = x.chunk(2, dim=1)
        x = outputs * gate.sigmoid()
        
        x = self.batch_norm(self.depthwise_conv(x))
        
        # apply swish
        x = x * x.sigmoid()
        
        #x = self.dropout(self.pointwise_conv_2(x))
        return x

        
import torch
import torch.nn as nn
'''

from torch import Tensor

class ConvolutionBlock(nn.Module):
    def __init__(self, input_channels, kernel_size, expansion_factor) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.dropout = 0.5

        assert (self.kernel_size -
                1) % 2 == 0, "kernel size should be an odd numbe for 'SAME' "

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
            out_channels=self.input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = self.layer_norm(x).transpose(1, 2)
        x = self.pointwise_conv_1(x)
        # apply GLU activation
        outputs, gate = x.chunk(2, dim=1)
        x = outputs * gate.sigmoid()
        x = self.batch_norm(self.depthwise_conv(x))
        
        # apply swish
        x = x * x.sigmoid()
        x = self.dropout(self.pointwise_conv_2(x))
        return x.transpose(1,2)

class FeedForwardModule(nn.Module):
    """
    Conformer Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.
    Args:
        encoder_dim (int): Dimension of conformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """
    def __init__(
            self,
            encoder_dim: int = 512,
            expansion_factor: int = 4,
            dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True),
            Swish(),
            nn.Dropout(p=dropout_p),
            nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)

class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()