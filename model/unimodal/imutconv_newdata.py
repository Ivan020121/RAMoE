import torch
import torch.nn as nn


class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
                bias=False,
            ),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, batch):
        return self.net(batch)


class TemporalConvEncoder(nn.Module):
    def __init__(self, input_dim=64, size_embeddings: int = 256, imu_channels: int = 9):
        super().__init__()
        self.name = "TemporalConvEncoder"
        self.net = nn.Sequential(
            nn.GroupNorm(3, imu_channels),
            TCN(imu_channels, input_dim, 5),
            TCN(input_dim, input_dim, 3),
            TCN(input_dim, input_dim, 3),
            nn.GroupNorm(4, input_dim),
            # TCN(input_dim, input_dim, 3),
            # nn.GroupNorm(4, input_dim),
        )
        self.rnn = nn.GRU(
            batch_first=True,
            num_layers=2,
            input_size=input_dim,
            hidden_size=size_embeddings,
            bidirectional=True,
            dropout=0.3,
        )

    def forward(self, batch):
        # B, T, C -> B, C, T
        batch = torch.permute(batch, (0, 2, 1))
        x = self.net(batch)
        # B, C, T -> B, T, C
        x = torch.permute(x, (0, 2, 1))
        _, h_n = self.rnn(x)
        return torch.mean(h_n, dim=0)
