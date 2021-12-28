"""
@Time    : 2021/10/16 15:14
@File    : dpcm.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPCMemory(nn.Module):
    def __init__(self, network, input_channels, hidden_channels, feature_dim, pred_steps, use_temperature, temperature,
                 use_memory_pool=False, stop_memory=False, m=None, K=None, agg='rnn', device='cuda'):
        super(DPCMemory, self).__init__()

        self.network = network
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.use_memory_pool = use_memory_pool
        self.stop_memory = stop_memory
        self.m = m
        self.K = K
        self.device = device

        if use_memory_pool:
            assert m is not None
            assert K is not None

        if network == 'r1d':
            self.encoder_q = R1DNet(input_channels, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                                    final_fc=True)
            if use_memory_pool:
                self.encoder_k = R1DNet(input_channels, hidden_channels, feature_dim, stride=2,
                                        kernel_size=[7, 11, 11, 7],
                                        final_fc=True)
            feature_size = self.encoder_q.feature_size
            self.feature_size = feature_size
            if agg == 'rnn':
                self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
            elif agg == 'transformer':
                self.agg = Transformer(feature_dim=feature_dim, num_layers=4, dim_feedforward=64, heads=4, dropout=0.1)
            else:
                raise ValueError
            self.predictor = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim)
            )
        elif network == 'r2d':
            self.encoder_q = ResNet(input_channels=input_channels, num_classes=feature_dim)
            if use_memory_pool:
                self.encoder_k = ResNet(input_channels=input_channels, num_classes=feature_dim)
            self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
            self.predictor = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim)
            )
        else:
            raise ValueError
        # self.gru = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)

        if use_memory_pool:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        self.relu = nn.ReLU(inplace=True)
        self.targets = None

        if use_memory_pool:
            self.register_buffer("queue", torch.randn(feature_dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # self._initialize_weights(self.agg)
        self._initialize_weights(self.predictor)

    def start_memory(self):
        assert self.stop_memory
        print('[INFO] Start using memory...')
        self.stop_memory = False
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.targets = None

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = keys.view(-1, self.feature_dim)
        batch_size, *_ = keys.shape

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f'{self.K}, {batch_size}'  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, *x_shape = x.shape
        x = x.view(batch_size * num_epoch, *x_shape)
        feature_q = self.encoder_q(x)  # (batch_size, num_epoch, feature_size)
        feature_q = feature_q.view(batch_size, num_epoch, self.feature_dim)
        feature_relu = self.relu(feature_q)

        if self.use_memory_pool and not self.stop_memory:
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                feature_k = self.encoder_k(x)
                feature_k = feature_k.view(batch_size, num_epoch, self.feature_dim)

        out, h_n = self.agg(feature_relu[:, :-self.pred_steps, :].contiguous())

        # Get predictions
        pred = []
        h_next = h_n
        c_next = out[:, -1, :].squeeze(1)
        for i in range(self.pred_steps):
            z_pred = self.predictor(c_next)
            pred.append(z_pred)
            c_next, h_next = self.agg(z_pred.unsqueeze(1), h_next)
            c_next = c_next[:, -1, :].squeeze(1)
        pred = torch.stack(pred, 1)  # (batch, pred_step, feature_dim)
        # Compute scores
        pred = pred.contiguous()

        feature_q = F.normalize(feature_q, p=2, dim=-1)
        pred = F.normalize(pred, p=2, dim=-1)
        if self.use_memory_pool and not self.stop_memory:
            feature_k = F.normalize(feature_k, p=2, dim=-1)

        # feature (batch_size, num_epoch, feature_size)
        # pred (batch_size, pred_steps, feature_size)
        if self.use_memory_pool and not self.stop_memory:
            logits_pos = torch.einsum('ijk,ijk->ij', [pred, feature_k[:, -self.pred_steps:, :]])
            logits_pos = logits_pos.view(batch_size * self.pred_steps, 1)

            logits_neg = torch.einsum('ijk,km->ijm', [pred, self.queue.clone().detach()])
            logits_neg = logits_neg.view(batch_size * self.pred_steps, self.K)

            logits = torch.cat([logits_pos, logits_neg], dim=-1)
        else:
            logits = torch.einsum('ijk,mnk->ijnm', [pred, feature_q])
            logits = logits.view(batch_size * self.pred_steps, num_epoch * batch_size)

        if self.use_temperature:
            logits /= self.temperature

        if self.targets is None:
            if self.use_memory_pool and not self.stop_memory:
                targets = torch.zeros(logits.shape[0]).long().cuda(self.device)
                self.targets = targets
            else:
                targets = torch.zeros(batch_size, num_epoch, self.pred_steps, batch_size)
                for i in range(batch_size):
                    for j in range(self.pred_steps):
                        targets[i, num_epoch - self.pred_steps + j, j, i] = 1
                targets = targets.view(batch_size * num_epoch, self.pred_steps * batch_size)
                targets = targets.t()
                targets = targets.argmax(dim=1)
                targets = targets.cuda(device=self.device)
                self.targets = targets

        if self.use_memory_pool and not self.stop_memory:
            self._dequeue_and_enqueue(feature_k)

        return logits, self.targets

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
