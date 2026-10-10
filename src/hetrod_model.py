import torch
import torch.nn as nn

class TrajectoryLSTM(nn.Module):
    """BiLSTM encoder over the whole window, LSTM decoder -> (x, y) per step."""
    def __init__(self, in_dim=14, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.enc = nn.LSTM(hidden, hidden, layers, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.dec = nn.LSTM(2 * hidden, hidden, layers, batch_first=True, dropout=dropout)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):                 # x: (B, W, in_dim)
        h, _ = self.enc(torch.relu(self.inp(x)))
        d, _ = self.dec(h)
        return self.out(d)                # (B, W, 2) normalized units