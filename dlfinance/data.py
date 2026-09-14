"""Data loading and feature engineering for financial time-series.

The module deliberately supports two price sources:

* ``synthetic`` (default) generates a reproducible geometric-Brownian-motion
  price path with a mild momentum regime. It requires no network access, which
  makes it ideal for CI, Cloud Agent builds, and quick experiments.
* ``yfinance`` downloads real daily OHLCV data for a ticker. It needs outbound
  network access and is best for interactive exploration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def generate_synthetic_prices(
    n_days: int = 1500,
    start_price: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.2,
    phi: float = 0.35,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a reproducible synthetic daily price series.

    Daily log returns follow an AR(1) process (momentum): each day's return is
    partly predictable from the previous day's return via the ``phi``
    autocorrelation coefficient. This gives a sequence model a genuine, if
    noisy, signal to learn, rather than a pure random walk which would be
    unlearnable by construction. ``phi`` is kept well below 1 so the series
    stays stationary and the signal remains realistically subtle.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    daily_drift = (mu - 0.5 * sigma**2) * dt
    daily_vol = sigma * np.sqrt(dt)
    shocks = rng.normal(0.0, daily_vol, size=n_days)

    log_returns = np.empty(n_days, dtype=np.float64)
    prev = 0.0
    for t in range(n_days):
        # AR(1) in the mean-zero component, plus a constant drift.
        centered = phi * prev + shocks[t]
        log_returns[t] = daily_drift + centered
        prev = centered

    prices = start_price * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    return pd.DataFrame({"Close": prices}, index=dates)


def load_prices(
    source: str = "synthetic",
    ticker: str = "AAPL",
    period: str = "5y",
    seed: int = 7,
) -> pd.DataFrame:
    """Return a DataFrame with a ``Close`` column indexed by date."""
    if source == "synthetic":
        return generate_synthetic_prices(seed=seed)
    if source == "yfinance":
        import yfinance as yf

        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError(
                f"yfinance returned no data for {ticker!r}; "
                "check network access or the ticker symbol."
            )
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return pd.DataFrame({"Close": close.astype(float)})
    raise ValueError(f"Unknown source: {source!r}. Use 'synthetic' or 'yfinance'.")


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Turn a price series into a small set of stationary features."""
    df = prices.copy()
    df["log_ret"] = np.log(df["Close"]).diff()
    df["ret_5"] = df["log_ret"].rolling(5).mean()
    df["ret_10"] = df["log_ret"].rolling(10).mean()
    df["vol_10"] = df["log_ret"].rolling(10).std()
    df["mom_20"] = df["Close"].pct_change(20)
    df = df.dropna()
    # Target: 1 if the next day's log return is positive, else 0.
    df["target"] = (df["log_ret"].shift(-1) > 0).astype(np.int64)
    return df.dropna()


FEATURE_COLUMNS = ["log_ret", "ret_5", "ret_10", "vol_10", "mom_20"]


def make_sequences(
    features: pd.DataFrame,
    window: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows of features and their next-day direction labels.

    Returns ``(X, y, next_log_ret)`` where ``X`` has shape
    ``(n_samples, window, n_features)`` and ``next_log_ret`` holds the realized
    next-day log return aligned with each label (used for backtesting).
    """
    feats = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = features["target"].to_numpy(dtype=np.float32)
    next_ret = features["log_ret"].shift(-1).to_numpy(dtype=np.float32)

    x_list, y_list, r_list = [], [], []
    for end in range(window, len(feats)):
        x_list.append(feats[end - window : end])
        y_list.append(targets[end - 1])
        r_list.append(next_ret[end - 1])

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    r = np.asarray(r_list, dtype=np.float32)
    return x, y, r


@dataclass
class PriceDataset:
    """A chronologically split, standardized dataset of price-window sequences."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    r_test: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray

    @property
    def n_features(self) -> int:
        return self.x_train.shape[-1]

    @classmethod
    def from_prices(
        cls,
        prices: pd.DataFrame,
        window: int = 20,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
    ) -> "PriceDataset":
        features = build_features(prices)
        x, y, r = make_sequences(features, window=window)

        n = len(x)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError("Not enough data for the requested split/window.")

        # Standardize using train statistics only, to avoid look-ahead leakage.
        train_x = x[:n_train]
        mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
        std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0) + 1e-8

        def norm(arr: np.ndarray) -> np.ndarray:
            return ((arr - mean) / std).astype(np.float32)

        return cls(
            x_train=norm(x[:n_train]),
            y_train=y[:n_train],
            x_val=norm(x[n_train : n_train + n_val]),
            y_val=y[n_train : n_train + n_val],
            x_test=norm(x[n_train + n_val :]),
            y_test=y[n_train + n_val :],
            r_test=r[n_train + n_val :],
            feature_mean=mean,
            feature_std=std,
        )
