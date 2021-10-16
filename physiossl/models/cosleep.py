"""
@Time    : 2021/10/16 15:11
@File    : cosleep.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoSleep(nn.Module):
    def __init__(self, first_network, second_network, first_channels, second_channels, hidden_channels, feature_dim,
                 pred_steps, use_temperature, temperature, m, K, num_prop, lam, lower_bound, upper_bound, device):
        super(CoSleep, self).__init__()

        self.first_network = first_network
        self.second_network = second_network
        self.first_channels = first_channels
        self.second_channels = second_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.m = m
        self.K = K
        self.num_prop = num_prop
        self.device = device

        self.lam = lam
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if first_network == 'r1d':
            self.encoder_q = R1DNet(first_channels, hidden_channels, feature_dim, stride=2,
                                    kernel_size=[7, 11, 11, 7],
                                    final_fc=True)
            self.encoder_k = R1DNet(first_channels, hidden_channels, feature_dim, stride=2,
                                    kernel_size=[7, 11, 11, 7],
                                    final_fc=True)
        elif first_network == 'r2d':
            self.encoder_q = ResNet(input_channels=first_channels, num_classes=feature_dim)
            self.encoder_k = ResNet(input_channels=first_channels, num_classes=feature_dim)
        else:
            raise ValueError

        if second_network == 'r1d':
            self.sampler = R1DNet(second_channels, hidden_channels, feature_dim, stride=2,
                                  kernel_size=[7, 11, 11, 7],
                                  final_fc=True)
        elif second_network == 'r2d':
            self.sampler = ResNet(input_channels=second_channels, num_classes=feature_dim)
        else:
            raise ValueError

        for param_s in self.sampler.parameters():
            param_s.requires_grad = False
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

        self.relu = nn.ReLU(inplace=True)
        # self.targets_pred = None

        self.register_buffer("queue_first", torch.randn(feature_dim, K))
        self.queue_first = F.normalize(self.queue_first, dim=0)

        self.register_buffer("queue_second", torch.randn(feature_dim, K))
        self.queue_second = F.normalize(self.queue_second, dim=0)

        self.register_buffer("queue_idx", torch.ones(K, dtype=torch.long) * -1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_is_full = False

        self._initialize_weights(self.predictor)

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, feature_q, feature_k, idx):
        batch_size = feature_q.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_first[:, ptr:ptr + batch_size] = feature_q.T
        self.queue_second[:, ptr:ptr + batch_size] = feature_k.T
        self.queue_idx[ptr:ptr + batch_size] = idx
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x1, x2, idx):
        assert x1.shape[:2] == x2.shape[:2] and x1.shape[:2] == idx.shape[:2], \
            f'x1: {x1.shape}, x2: {x2.shape}, idx: {idx.shape}'

        (B1, num_epoch, *epoch_shape1) = x1.shape
        (B2, num_epoch, *epoch_shape2) = x2.shape

        # Compute query features for the first view
        x1 = x1.view(B1 * num_epoch, *epoch_shape1)
        feature_q = self.encoder_q(x1)
        feature_q = F.normalize(feature_q, p=2, dim=-1)
        feature_q = feature_q.view(B1, num_epoch, self.feature_dim)

        # Get predictions
        feature_relu = self.relu(feature_q)
        out, h_n = self.agg(feature_relu[:, :-self.pred_steps, :].contiguous())
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
        pred = F.normalize(pred, p=2, dim=-1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            # Compute key features for the first view
            feature_k = self.encoder_k(x1)
            feature_k = F.normalize(feature_k, p=2, dim=-1)
            feature_k = feature_k.view(B1, num_epoch, self.feature_dim)

            # Compute key features for the second view
            x2 = x2.view(B2 * num_epoch, *epoch_shape2)
            feature_kf = self.sampler(x2)
            feature_kf = F.normalize(feature_kf, p=2, dim=-1)
            feature_kf = feature_kf.view(B2, num_epoch, self.feature_dim)

        logits_pos = torch.einsum('ijk,ijk->ij', [pred, feature_k[:, -self.pred_steps:, :]]).unsqueeze(-1)
        logits_neg = torch.einsum('ijk,km->ijm', [pred, self.queue_first.clone().detach()])

        if not self.queue_is_full:
            self.queue_is_full = torch.all(self.queue_idx != -1)
            if self.queue_is_full:
                print('[INFO] ===== Queue is full now =====')

        # targets_mem = idx.unsqueeze(-1)[:, -self.pred_steps:] == self.queue_idx.unsqueeze(0) Notice!
        targets_mem = torch.zeros(B1, self.pred_steps, self.K).cuda(self.device)

        if self.queue_is_full:
            sim_v1 = torch.einsum('ijk,km->ijm',
                                  [feature_k[:, -self.pred_steps:, :], self.queue_first.clone().detach()])
            sim_v2 = torch.einsum('ijk,km->ijm',
                                  [feature_kf[:, -self.pred_steps:, :], self.queue_second.clone().detach()])

            sim_profile = self.lam * sim_v1 + (1 - self.lam) * sim_v2
            _, topk_idx = torch.topk(sim_profile, k=self.num_prop, dim=-1)
            # topk_mat = torch.zeros(B1, self.pred_steps, self.K).cuda(self.device)
            targets_mem.scatter_(-1, topk_idx, 1)

            low_th = int(self.lower_bound * self.K)
            high_th = int(self.upper_bound * self.K)
            semi_hard_idx = torch.argsort(sim_profile, dim=-1)[:, :, low_th: high_th]
            semi_hard_onehot = torch.zeros_like(targets_mem, dtype=torch.bool).cuda(self.device)
            semi_hard_onehot.scatter_(-1, semi_hard_idx, 1)

            targets_mem = targets_mem[semi_hard_onehot]
            targets_mem = targets_mem.view(self.pred_steps * B1, semi_hard_idx.shape[-1])
            targets = torch.cat([torch.ones(targets_mem.shape[0], 1).long().cuda(self.device), targets_mem], dim=-1)

            logits_neg = logits_neg[semi_hard_onehot]
            logits_neg = logits_neg.view(B1, self.pred_steps, semi_hard_idx.shape[-1])
            logits = torch.cat([logits_pos, logits_neg], dim=-1)
            logits = logits.view(B1 * self.pred_steps, semi_hard_idx.shape[-1] + 1)
            if self.use_temperature:
                logits /= self.temperature
        else:
            targets_mem = targets_mem.view(self.pred_steps * B1, self.K)
            targets = torch.cat([torch.ones(targets_mem.shape[0], 1).long().cuda(self.device), targets_mem], dim=-1)

            logits = torch.cat([logits_pos, logits_neg], dim=-1)
            logits = logits.view(B1 * self.pred_steps, logits.shape[-1])
            if self.use_temperature:
                logits /= self.temperature

        assert targets.shape == logits.shape

        self._dequeue_and_enqueue(feature_k.view(B1 * num_epoch, self.feature_dim),
                                  feature_kf.view(B1 * num_epoch, self.feature_dim),
                                  idx.view(B1 * num_epoch))

        return logits, targets
