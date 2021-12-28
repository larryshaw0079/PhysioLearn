"""
@Time    : 2021/10/16 15:12
@File    : dcc.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DisContrasCoding(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, use_temperature, temperature,
                 device):
        super(DisContrasCoding, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.device = device
        self.encoder = R1DNet(input_channels, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                              final_fc=True)

        self.targets = None

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, channel, time_len = x.shape
        x = x.view(batch_size * num_epoch, channel, time_len)
        feature = self.encoder(x)
        feature = F.normalize(feature, p=2, dim=1)
        feature = feature.view(batch_size, num_epoch, self.feature_dim)

        #################################################################
        #                       Multi-InfoNCE Loss                      #
        #################################################################
        mask = torch.zeros(batch_size, num_epoch, num_epoch, batch_size, dtype=bool)
        for i in range(batch_size):
            for j in range(num_epoch):
                mask[i, j, :, i] = 1
        mask = mask.cuda(self.device)

        logits = torch.einsum('ijk,mnk->ijnm', [feature, feature])
        # if self.use_temperature:
        #     logits /= self.temperature

        pos = torch.exp(logits.masked_select(mask).view(batch_size, num_epoch, num_epoch)).sum(-1)
        neg = torch.exp(logits.masked_select(torch.logical_not(mask)).view(batch_size, num_epoch,
                                                                           batch_size * num_epoch - num_epoch)).sum(-1)

        loss = (-torch.log(pos / (pos + neg))).mean()

        return loss

        # Compute scores
        # logits = torch.einsum('ijk,kmn->ijmn', [pred, feature])  # (batch, pred_step, num_seq, batch)
        # logits = logits.view(batch_size * self.pred_steps, num_epoch * batch_size)

        # logits = torch.einsum('ijk,mnk->ijnm', [feature, feature])
        # # print('3. Logits: ', logits.shape)
        # logits = logits.view(batch_size * num_epoch, num_epoch * batch_size)
        # if self.use_temperature:
        #     logits /= self.temperature
        #
        # if self.targets is None:
        #     targets = torch.zeros(batch_size, num_epoch, num_epoch, batch_size)
        #     for i in range(batch_size):
        #         for j in range(num_epoch):
        #             targets[i, j, :, i] = 1
        #     targets = targets.view(batch_size * num_epoch, num_epoch * batch_size)
        #     targets = targets.argmax(dim=1)
        #     targets = targets.cuda(device=self.device)
        #     self.targets = targets
        #
        # return logits, self.targets

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
