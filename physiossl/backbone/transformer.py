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
    def __init__(self, feature_dim: int, max_len: int, dropout: float):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, feature_dim)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(
            torch.arange(0, feature_dim, 2).float() * (-np.log(10000.0)) / feature_dim)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


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

        self.init_weights()

    def init_weights(self) -> None:
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        # src (batch_size, seq_len, feature_dim)

        if self.use_pos_enc:
            src = src * np.sqrt(self.feature_dim)
            src = self.pos_encoder(src)

        if self.use_src_mask:
            mask = torch.triu(torch.ones(src.size(1), src.size(1)) * float('-inf'), diagonal=1)
        else:
            mask = None

        return self.encoder(src, mask)


# class SequenceTransformer(nn.Module):
#     def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
#         super().__init__()
#         patch_dim = channels * patch_size
#         self.patch_to_embedding = nn.Linear(patch_dim, dim)
#         self.c_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.transformer = TransformerLayer(dim, depth, heads, mlp_dim, dropout)
#         self.to_c_token = nn.Identity()
# 
#     def forward(self, forward_seq):
#         x = self.patch_to_embedding(forward_seq)
#         b, n, _ = x.shape
#         c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
#         x = torch.cat((c_tokens, x), dim=1)
#         x = self.transformer(x)
#         c_t = self.to_c_token(x[:, 0])
#         return c_t

