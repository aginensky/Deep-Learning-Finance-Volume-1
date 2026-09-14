"""Deep-Learning-Finance-Volume-1.

A small, CPU-friendly deep learning toolkit for financial time-series. It
provides utilities to build datasets of price windows, an LSTM classifier that
predicts next-day return direction, and a training/backtesting entry point.
"""

from .data import PriceDataset, load_prices, make_sequences
from .model import DirectionLSTM

__all__ = [
    "PriceDataset",
    "load_prices",
    "make_sequences",
    "DirectionLSTM",
]

__version__ = "0.1.0"
