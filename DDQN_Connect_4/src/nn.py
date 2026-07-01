import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(64 * 6 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        ch_pos = (x == 1).float()
        ch_neg = (x == -1).float()
        x = torch.stack([ch_pos, ch_neg], dim=1)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.head(x)
        return x
