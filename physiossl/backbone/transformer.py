"""
@Time    : 2021/10/3 11:37
@File    : transformer.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, feature_dim: int, num_head: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = 'relu', num_layers: int = 4, norm: nn.Module = None):
        super(Transformer, self).__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            ),
            num_layers=num_layers, norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        return self.transformer(src, mask, src_key_padding_mask)


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


class SequenceTransformer(nn.Module):
    def __init__(self, feature_dim: int, num_head: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = 'relu', num_layers: int = 4, norm: nn.Module = None):
        super(SequenceTransformer, self).__init__()
