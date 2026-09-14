# Deep-Learning-Finance-Volume-1

A small, CPU-friendly deep learning toolkit for financial time-series. It trains
an LSTM to predict the **direction of the next day's return** and backtests a
simple long/flat strategy against buy-and-hold.

The project runs fully offline by default using reproducible synthetic price
data, and can optionally pull real market data via `yfinance`.

## Project layout

```
dlfinance/
  data.py     # price loading, feature engineering, sequence windows, splits
  model.py    # DirectionLSTM classifier (PyTorch)
  train.py    # training loop, evaluation, backtest, CLI entry point
notebooks/
  demo.ipynb  # end-to-end walkthrough notebook
requirements.txt
```

## Setup

The project targets Python 3.12 on Linux and uses a virtual environment. PyTorch
is installed as the CPU-only build (see `requirements.txt`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On a fresh Ubuntu machine you may first need the venv/build packages:

```bash
sudo apt-get update && sudo apt-get install -y python3.12-venv python3-dev build-essential
```

## Usage

Train on synthetic data (default, offline):

```bash
python -m dlfinance.train --epochs 40
```

Train on real market data:

```bash
python -m dlfinance.train --source yfinance --ticker MSFT --period 5y --epochs 40
```

Outputs (metrics, saved model, and a `training_report.png` with the training
curve and backtest) are written to `outputs/`.

### Useful flags

| Flag | Default | Description |
| --- | --- | --- |
| `--source` | `synthetic` | `synthetic` or `yfinance` |
| `--ticker` | `AAPL` | Ticker when `--source yfinance` |
| `--window` | `20` | Look-back window (days) per sample |
| `--epochs` | `40` | Training epochs |
| `--hidden-size` | `32` | LSTM hidden units |
| `--lr` | `1e-3` | Learning rate |
| `--seed` | `7` | Random seed |

## Notes

This is an educational example, not investment advice. The synthetic data has a
deliberately mild, learnable momentum signal so the model has something to fit;
real markets are far noisier.
