"""
@Time    : 2021/10/3 11:37
@File    : transformer.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    adapt from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, feature_dim: int, dropout: float = 0.1, max_len: int = 5000):
        """

        Args:
            feature_dim ():
            dropout ():
            max_len (): determines how far the position can have an effect on a token (window)
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2) * (-np.log(10000.0) / feature_dim))
        pe = torch.zeros(max_len, 1, feature_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        Args:
            x (): shape [batch_size, seq_len, embedding_dim]

        Returns:

        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim: int, num_head: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = 'relu', num_layers: int = 4, norm: nn.Module = None,
                 use_pos_enc: bool = True, use_src_mask: bool = True, max_len: int = 5000):
        """
        A transformer encoder with positional encoding and source masking.

        Args:
            feature_dim ():
            num_head ():
            dim_feedforward ():
            dropout ():
            activation ():
            num_layers ():
            norm ():
            use_pos_enc ():
            use_src_mask ():
        """
        super(TransformerEncoder, self).__init__()

        self.feature_dim = feature_dim
        self.use_pos_enc = use_pos_enc
        self.use_src_mask = use_src_mask

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True
            ),
            num_layers=num_layers, norm=norm)

        if use_pos_enc:
            self.pos_encoder = PositionalEncoding(feature_dim=feature_dim, max_len=max_len, dropout=dropout)

    def forward(self, src):
        # src (batch_size, seq_len, feature_dim)

        if self.use_pos_enc:
            src = src * np.sqrt(self.feature_dim)
            src = self.pos_encoder(src)

        if self.use_src_mask:
            # EX for size=5:
            # [[0., -inf, -inf, -inf, -inf],
            #  [0.,   0., -inf, -inf, -inf],
            #  [0.,   0.,   0., -inf, -inf],
            #  [0.,   0.,   0.,   0., -inf],
            #  [0.,   0.,   0.,   0.,   0.]]
            mask = torch.triu(torch.ones(src.size(1), src.size(1)) * float('-inf'), diagonal=1)
        else:
            mask = None

        return self.encoder(src, mask)
