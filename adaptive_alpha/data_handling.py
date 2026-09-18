"""Consolidated module for the adaptive alpha project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_FIELDS = ("open", "high", "low", "close", "return", "volume", "dollar_volume")


@dataclass(frozen=True)
class PanelData:
    """Wide date x asset data used by the factor engine.

    Fields are pandas DataFrames sharing the same DateTimeIndex and asset columns.
    Metadata is indexed by asset and may contain industry, exchange, share_code,
    and shares_outstanding.
    """

    fields: dict[str, pd.DataFrame]
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        if "close" not in self.fields:
            raise ValueError("PanelData requires a 'close' field.")
        base_index = self.fields["close"].index
        base_columns = self.fields["close"].columns
        normalized: dict[str, pd.DataFrame] = {}
        for name, frame in self.fields.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"Field {name!r} must be a pandas DataFrame.")
            normalized[name] = frame.reindex(index=base_index, columns=base_columns)
        object.__setattr__(self, "fields", normalized)
        object.__setattr__(self, "metadata", self.metadata.reindex(base_columns))

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.fields["close"].index)

    @property
    def assets(self) -> pd.Index:
        return self.fields["close"].columns

    def field(self, name: str) -> pd.DataFrame:
        if name not in self.fields:
            raise KeyError(f"Unknown panel field: {name}")
        return self.fields[name]

    def slice_dates(self, start: str | pd.Timestamp | None, end: str | pd.Timestamp | None) -> "PanelData":
        mask = pd.Series(True, index=self.dates)
        if start is not None:
            mask &= self.dates >= pd.Timestamp(start)
        if end is not None:
            mask &= self.dates <= pd.Timestamp(end)
        selected = self.dates[mask.to_numpy()]
        return PanelData({k: v.loc[selected] for k, v in self.fields.items()}, self.metadata.copy())

    def take_date_positions(self, start: int, end: int) -> "PanelData":
        selected = self.dates[start:end]
        return PanelData({k: v.loc[selected] for k, v in self.fields.items()}, self.metadata.copy())

    def with_field(self, name: str, frame: pd.DataFrame) -> "PanelData":
        fields = dict(self.fields)
        fields[name] = frame
        return PanelData(fields, self.metadata.copy())

    def future_returns(self, horizon: int = 5, execution_delay: int = 1) -> pd.DataFrame:
        if horizon <= 0 or execution_delay < 0:
            raise ValueError("horizon must be positive and execution_delay must be non-negative.")
        close = self.field("close")
        start_px = close.shift(-execution_delay)
        end_px = close.shift(-(execution_delay + horizon))
        return end_px / start_px - 1.0

    def market_cap(self) -> pd.DataFrame:
        if "shares_outstanding" in self.fields:
            return self.field("close") * self.field("shares_outstanding")
        if "shares_outstanding" in self.metadata.columns:
            shares = self.metadata["shares_outstanding"].astype(float).reindex(self.assets)
            return self.field("close").mul(shares, axis=1)
        return self.field("dollar_volume").rolling(21, min_periods=5).mean()

    def observable_universe_mask(
        self,
        min_price: float = 5.0,
        min_dollar_volume: float | None = None,
        allowed_exchanges: Iterable[str] | None = None,
        allowed_share_codes: Iterable[int] | None = None,
    ) -> pd.DataFrame:
        close = self.field("close")
        mask = close >= min_price
        if min_dollar_volume is not None:
            mask &= self.field("dollar_volume") >= min_dollar_volume
        if allowed_exchanges is not None and "exchange" in self.metadata:
            allowed = set(allowed_exchanges)
            cols = [asset for asset in self.assets if self.metadata.loc[asset, "exchange"] in allowed]
            mask = mask.where(mask.columns.isin(cols), False)
        if allowed_share_codes is not None and "share_code" in self.metadata:
            allowed_codes = set(allowed_share_codes)
            cols = [asset for asset in self.assets if self.metadata.loc[asset, "share_code"] in allowed_codes]
            mask = mask.where(mask.columns.isin(cols), False)
        return mask.fillna(False)

    def mask_assets(self, mask: pd.DataFrame | pd.Series) -> "PanelData":
        if isinstance(mask, pd.Series):
            asset_mask = mask.reindex(self.assets).fillna(False).astype(bool)
            cols = self.assets[asset_mask.to_numpy()]
            return PanelData({k: v.loc[:, cols] for k, v in self.fields.items()}, self.metadata.loc[cols].copy())
        aligned = mask.reindex(index=self.dates, columns=self.assets).fillna(False)
        fields = {k: v.where(aligned) for k, v in self.fields.items()}
        return PanelData(fields, self.metadata.copy())

    def summary(self) -> dict[str, float | int | str]:
        close = self.field("close")
        return {
            "start": str(self.dates.min().date()) if len(self.dates) else "",
            "end": str(self.dates.max().date()) if len(self.dates) else "",
            "n_dates": int(close.shape[0]),
            "n_assets": int(close.shape[1]),
            "coverage": float(close.notna().mean().mean()),
        }


def from_wide_fields(fields: dict[str, pd.DataFrame], metadata: pd.DataFrame | None = None) -> PanelData:
    close = fields["close"].sort_index()
    prepared = {name: frame.reindex(index=close.index, columns=close.columns).astype(float) for name, frame in fields.items()}
    if "return" not in prepared:
        prepared["return"] = prepared["close"].pct_change(fill_method=None)
    if "dollar_volume" not in prepared:
        prepared["dollar_volume"] = prepared["close"] * prepared.get("volume", prepared["close"] * np.nan)
    if metadata is None:
        metadata = pd.DataFrame(index=close.columns)
    return PanelData(prepared, metadata)

from pathlib import Path

import pandas as pd



def load_price_volume_csv(path: str | Path) -> PanelData:
    """Load a long CSV into PanelData.

    Required columns: date, asset, open, high, low, close, volume.
    Optional columns include shares_outstanding, industry, exchange, share_code.
    """

    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "asset", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    df = df.sort_values(["date", "asset"])
    fields: dict[str, pd.DataFrame] = {}
    for column in ["open", "high", "low", "close", "volume", "shares_outstanding"]:
        if column in df.columns:
            fields[column] = df.pivot(index="date", columns="asset", values=column)
    fields["return"] = fields["close"].pct_change(fill_method=None)
    fields["dollar_volume"] = fields["close"] * fields["volume"]

    meta_cols = [c for c in ["industry", "exchange", "share_code", "shares_outstanding"] if c in df.columns]
    if meta_cols:
        metadata = df.sort_values("date").groupby("asset")[meta_cols].last()
    else:
        metadata = pd.DataFrame(index=fields["close"].columns)
    return from_wide_fields(fields, metadata)

import numpy as np
import pandas as pd



def make_synthetic_panel(
    n_assets: int = 120,
    n_days: int = 720,
    seed: int = 7,
    start: str = "2018-01-01",
    n_industries: int = 10,
) -> PanelData:
    """Create a price-volume panel with weak latent price-volume alpha.

    The process is intentionally simple but contains enough structure for
    falsification policies to learn: short-term reversal, medium momentum,
    liquidity shocks, and volatility stress effects.
    """

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    assets = pd.Index([f"S{i:04d}" for i in range(n_assets)], name="asset")
    industries = np.array([f"ind_{i % n_industries:02d}" for i in range(n_assets)])

    market = rng.normal(0.00025, 0.009, size=n_days)
    industry_shocks = rng.normal(0.0, 0.006, size=(n_days, n_industries))
    idio = rng.normal(0.0, 0.018, size=(n_days, n_assets))
    beta = rng.normal(1.0, 0.18, size=n_assets)
    industry_idx = np.array([i % n_industries for i in range(n_assets)])

    raw_returns = (
        market[:, None] * beta[None, :]
        + industry_shocks[:, industry_idx]
        + idio
    )

    close = 30.0 * np.exp(np.cumsum(raw_returns, axis=0))
    close = np.clip(close, 2.0, None)
    base_volume = rng.lognormal(mean=12.4, sigma=0.7, size=n_assets)
    volume_noise = rng.lognormal(mean=0.0, sigma=0.35, size=(n_days, n_assets))
    volatility_proxy = pd.DataFrame(raw_returns).rolling(10, min_periods=1).std().to_numpy()
    volume = base_volume[None, :] * volume_noise * (1.0 + 12.0 * np.nan_to_num(volatility_proxy))

    close_df = pd.DataFrame(close, index=dates, columns=assets)
    volume_df = pd.DataFrame(volume, index=dates, columns=assets)
    returns_df = close_df.pct_change(fill_method=None)

    reversal = -returns_df.shift(1).fillna(0.0)
    momentum = close_df.pct_change(20, fill_method=None).shift(1).fillna(0.0)
    liquidity = -np.log1p(close_df * volume_df).rank(axis=1, pct=True).sub(0.5)
    vol_penalty = -returns_df.rolling(15, min_periods=5).std().fillna(0.0)
    latent_alpha = 0.05 * reversal + 0.025 * momentum + 0.015 * liquidity + 0.04 * vol_penalty
    latent_alpha = latent_alpha.clip(-0.015, 0.015)

    adjusted_returns = returns_df.fillna(0.0) + latent_alpha.shift(1).fillna(0.0)
    close_df = 30.0 * (1.0 + adjusted_returns).cumprod()
    close_df = close_df.clip(lower=2.0)
    open_df = close_df.shift(1).fillna(close_df.iloc[0]).mul(1.0 + rng.normal(0, 0.002, size=close_df.shape))
    high_df = pd.concat([open_df, close_df]).groupby(level=0).max() * (1.0 + rng.uniform(0.0, 0.01, size=close_df.shape))
    low_df = pd.concat([open_df, close_df]).groupby(level=0).min() * (1.0 - rng.uniform(0.0, 0.01, size=close_df.shape))

    shares = pd.Series(rng.lognormal(mean=17.5, sigma=0.55, size=n_assets), index=assets, name="shares_outstanding")
    shares_df = pd.DataFrame(np.tile(shares.to_numpy(), (n_days, 1)), index=dates, columns=assets)
    metadata = pd.DataFrame(
        {
            "industry": industries,
            "exchange": np.where(np.arange(n_assets) % 3 == 0, "NYSE", np.where(np.arange(n_assets) % 3 == 1, "NASDAQ", "AMEX")),
            "share_code": 10,
            "shares_outstanding": shares,
        },
        index=assets,
    )

    return from_wide_fields(
        {
            "open": open_df,
            "high": high_df,
            "low": low_df,
            "close": close_df,
            "return": close_df.pct_change(fill_method=None),
            "volume": volume_df,
            "shares_outstanding": shares_df,
            "dollar_volume": close_df * volume_df,
        },
        metadata,
    )
