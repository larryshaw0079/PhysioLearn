"""
@Time    : 2021/10/3 11:37
@File    : rnn.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.3):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None):
        # x:   (batch, seq_len,    input_size)
        # h_0: (num_layers, batch, hidden_size)

        batch_size, num_epoch, *_ = x.shape

        if h_0 is None:
            h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
            h_0 = h_0.cuda(x.device)

        out, h_n = self.gru(x, h_0)

        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        return out, h_n


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.3):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, x: torch.Tensor, h_0: torch.Tensor = None, c_0: torch.Tensor = None):
        """

        Args:
            x (): of shape (batch, seq_len,    input_size)
            h_0 (): of shape (num_layers, batch, hidden_size)
            c_0 ():

        Returns:

        """

        batch_size, num_epoch, *_ = x.shape

        if h_0 is None:
            h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
            h_0 = h_0.to(x.device)

        if c_0 is None:
            c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
            c_0 = c_0.to(x.device)

        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        return out, (h_n, c_n)

# class ConvGRUCell1D(nn.Module):
#     ''' Initialize ConvGRU cell '''
#
#     def __init__(self, input_size: int, hidden_size: int, kernel_size: int):
#         super(ConvGRUCell1D, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.kernel_size = kernel_size
#         padding = kernel_size // 2
#
#         self.reset_gate = nn.Conv1d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
#         self.update_gate = nn.Conv1d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
#         self.out_gate = nn.Conv1d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
#
#         self.initialize()
#
#     def initialize(self):
#         nn.init.orthogonal_(self.reset_gate.weight)
#         nn.init.orthogonal_(self.update_gate.weight)
#         nn.init.orthogonal_(self.out_gate.weight)
#         nn.init.constant_(self.reset_gate.bias, 0.)
#         nn.init.constant_(self.update_gate.bias, 0.)
#         nn.init.constant_(self.out_gate.bias, 0.)
#
#     def forward(self, input_tensor: torch.Tensor, hidden_state: torch.Tensor):
#         if hidden_state is None:
#             B, C, *spatial_dim = input_tensor.size()
#             hidden_state = torch.zeros([B, self.hidden_size, *spatial_dim]).cuda(input_tensor.device)
#
#         # [B, C, H, W]
#         combined = torch.cat([input_tensor, hidden_state], dim=1)  # concat in C
#         update = torch.sigmoid(self.update_gate(combined))
#         reset = torch.sigmoid(self.reset_gate(combined))
#         out = torch.tanh(self.out_gate(torch.cat([input_tensor, hidden_state * reset], dim=1)))
#         new_state = hidden_state * (1 - update) + out * update
#
#         return new_state
#
#
# class ConvGRU(nn.Module):
#     ''' Initialize a multi-layer Conv GRU '''
#
#     def __init__(self, input_size: int, hidden_size: int, kernel_size: int, num_layers: int, dropout: float = 0.1):
#         super(ConvGRU, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.kernel_size = kernel_size
#         self.num_layers = num_layers
#
#         cell_list = []
#         for i in range(self.num_layers):
#             if i == 0:
#                 input_dim = self.input_size
#             else:
#                 input_dim = self.hidden_size
#             cell = ConvGRUCell1D(input_dim, self.hidden_size, self.kernel_size)
#             name = 'ConvGRUCell1d_' + str(i).zfill(2)
#
#             setattr(self, name, cell)
#             cell_list.append(getattr(self, name))
#
#         self.cell_list = nn.ModuleList(cell_list)
#         self.dropout_layer = nn.Dropout(p=dropout)
#
#     def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None):
#         [B, seq_len, *_] = x.size()
#
#         if hidden_state is None:
#             hidden_state = [None] * self.num_layers
#         # input: image sequences [B, T, C, H, W]
#         current_layer_input = x
#         del x
#
#         last_state_list = []
#
#         for idx in range(self.num_layers):
#             cell_hidden = hidden_state[idx]
#             output_inner = []
#             for t in range(seq_len):
#                 cell_hidden = self.cell_list[idx](current_layer_input[:, t, :], cell_hidden)
#                 cell_hidden = self.dropout_layer(cell_hidden)  # dropout in each time step
#                 output_inner.append(cell_hidden)
#
#             layer_output = torch.stack(output_inner, dim=1)
#             current_layer_input = layer_output
#
#             last_state_list.append(cell_hidden)
#
#         last_state_list = torch.stack(last_state_list, dim=1)
#
#         return layer_output, last_state_list
