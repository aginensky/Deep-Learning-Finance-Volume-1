"""Train and evaluate the next-day return-direction model.

Run it as a module:

    python -m dlfinance.train --epochs 40

By default it uses reproducible synthetic data so it works offline. Pass
``--source yfinance --ticker MSFT`` to train on real market data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import PriceDataset, load_prices
from .model import DirectionLSTM


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    model.eval()
    logits = model(torch.from_numpy(x))
    probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(np.float32)
    acc = float((preds == y).mean())
    return acc, probs


def backtest(probs: np.ndarray, next_log_ret: np.ndarray) -> dict:
    """Go long when the model predicts 'up', flat otherwise; compare to buy&hold."""
    signal = (probs > 0.5).astype(np.float64)
    strat_returns = signal * next_log_ret
    strat_curve = np.exp(np.cumsum(strat_returns))
    hold_curve = np.exp(np.cumsum(next_log_ret))
    return {
        "strategy_total_return": float(strat_curve[-1] - 1.0),
        "buy_hold_total_return": float(hold_curve[-1] - 1.0),
        "strategy_curve": strat_curve.tolist(),
        "buy_hold_curve": hold_curve.tolist(),
    }


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)

    prices = load_prices(
        source=args.source, ticker=args.ticker, period=args.period, seed=args.seed
    )
    data = PriceDataset.from_prices(prices, window=args.window)
    print(
        f"Loaded {len(prices)} prices from '{args.source}'. "
        f"Train/Val/Test sequences: "
        f"{len(data.x_train)}/{len(data.x_val)}/{len(data.x_test)} "
        f"({data.n_features} features, window={args.window})"
    )

    model = DirectionLSTM(n_features=data.n_features, hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = _loader(data.x_train, data.y_train, args.batch_size, shuffle=True)

    history = {"train_loss": [], "val_acc": []}
    best_val = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(data.x_train)

        val_acc, _ = evaluate(model, data.x_val, data.y_val)
        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | train_loss {epoch_loss:.4f} | val_acc {val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_probs = evaluate(model, data.x_test, data.y_test)
    bt = backtest(test_probs, data.r_test)
    print(
        f"\nBest val_acc {best_val:.3f} | test_acc {test_acc:.3f} | "
        f"strategy {bt['strategy_total_return']*100:+.1f}% vs "
        f"buy&hold {bt['buy_hold_total_return']*100:+.1f}%"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_plots(history, bt, out_dir)
    torch.save(model.state_dict(), out_dir / "model.pt")

    metrics = {
        "source": args.source,
        "ticker": args.ticker if args.source == "yfinance" else None,
        "epochs": args.epochs,
        "window": args.window,
        "best_val_acc": best_val,
        "test_acc": test_acc,
        "strategy_total_return": bt["strategy_total_return"],
        "buy_hold_total_return": bt["buy_hold_total_return"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model, metrics, and plots to {out_dir.resolve()}")
    return metrics


def _save_plots(history: dict, bt: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(history["train_loss"], label="train loss", color="tab:red")
    ax1b = ax1.twinx()
    ax1b.plot(history["val_acc"], label="val acc", color="tab:blue")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:red")
    ax1b.set_ylabel("val accuracy", color="tab:blue")
    ax1.set_title("Training curve")

    ax2.plot(bt["strategy_curve"], label="LSTM long/flat", color="tab:green")
    ax2.plot(bt["buy_hold_curve"], label="buy & hold", color="tab:gray")
    ax2.set_xlabel("test day")
    ax2.set_ylabel("growth of $1")
    ax2.set_title("Backtest on held-out test set")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_report.png", dpi=120)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train an LSTM to predict return direction.")
    p.add_argument("--source", choices=["synthetic", "yfinance"], default="synthetic")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--period", default="5y")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-dir", default="outputs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
