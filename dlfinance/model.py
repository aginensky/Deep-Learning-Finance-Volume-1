"""PyTorch models for next-day return-direction classification."""

from __future__ import annotations

import torch
from torch import nn


class DirectionLSTM(nn.Module):
    """A compact LSTM that maps a window of features to an up/down logit."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features)
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(last).squeeze(-1)
