"""Degraded drift: every [q10, q90] band inflated 5x around q50 (deterministic Gate-3 failure)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Forecaster:
    min_samples = 20

    def predict_origin(self, pack_slice: dict, origin_date: str, horizons, external_signals=None) -> pd.DataFrame:
        hist = pack_slice["series_history"]
        hist = hist[hist["series_id"].isin(pack_slice["targets"]["series_id"])]
        rows = []
        for sid, g in hist.groupby("series_id", sort=True):
            days = pd.to_datetime(g["date"]).map(pd.Timestamp.toordinal).to_numpy()
            vals = g["value"].to_numpy(float)
            last, use_log = vals[-1], bool((vals > 0).all())
            for h in horizons:
                j = np.searchsorted(days, days + h)
                ok = (j < len(days)) & (days[np.minimum(j, len(days) - 1)] == days + h)
                a, b = vals[ok], vals[j[ok]]
                ch = np.log(b / a) if use_log else b - a
                if len(ch) >= self.min_samples:
                    q = np.quantile(ch, [0.1, 0.5, 0.9])
                    q10, q50, q90 = last * np.exp(q) if use_log else last + q
                else:
                    q10 = q50 = q90 = last
                rows.append((sid, origin_date, h, min(q10, q50), q50, max(q90, q50)))
        df = pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
        df["q10"] = np.maximum(df["q50"] - 5 * (df["q50"] - df["q10"]), 0.01 * df["q50"])
        df["q90"] = df["q50"] + 5 * (df["q90"] - df["q50"])
        return df
