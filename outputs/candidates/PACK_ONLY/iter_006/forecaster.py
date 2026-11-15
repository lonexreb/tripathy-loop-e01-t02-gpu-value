"""Multi-regime first-principles forecaster (deterministic Task-4 candidate).

Regime A, AWS hyperscaler_spot (administered, smoothed spot price; ~96% of the pooled loss):
  H0 momentum: AWS spot moves in trends, so the h-day log change is regressed (pooled OLS over every
     benchmark AZ series, trailing 540 days) on the trailing 1/3/7-day log changes.
  H1 stickiness: residuals are scaled by local volatility (14-day mean |1-day log change|), so the
     interval collapses toward the last value on zones whose price has stopped moving.
  H3 conformal: q10/q50/q90 = forecast + local vol x empirical quantiles of the pooled standardised
     in-sample residuals (split-conformal calibration of the 80% band).
Regime B, daily-statistics rate cards (neocloud_rate_card / hyperscaler_list; menu-cost pricing):
  empirical h-day log-change quantiles (drift); H1: q50 pinned to the last value when >= 50% of the
  trailing 60 one-day changes are zero; H3: band width scaled by the IQR dispersion (p75 - p25) / median
  relative to its trailing 60-day median (clipped to [0.6, 1.6]).
Regime C, daily-statistics marketplace_ask / spot series:
  drift. H2 cross-collector error correction (q50 pulled, ramped by h/7, toward the median of the other
  collectors' same-family, same-tier, same-class daily medians over the last 7 days) is kept as a switch
  but disabled (kappa = 0): on Discovery a pull of 0.25 cut the non-AWS pooled skill from +0.06 to -0.32,
  and a cross-AZ consensus term cut the AWS skill from +0.44 to +0.33 (falsified, see theory_of_market.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QS = np.array([0.1, 0.5, 0.9])
EPOCH_ORDINAL = 719163


def _days(dates: pd.Series) -> np.ndarray:
    return pd.to_datetime(dates).to_numpy().astype("datetime64[D]").astype(np.int64) + EPOCH_ORDINAL


def _shift(days: np.ndarray, x: np.ndarray, k: int) -> np.ndarray:
    """x at day t + k aligned to t (NaN when that calendar day is absent); k < 0 looks back."""
    j = np.searchsorted(days, days + k)
    jc = np.minimum(j, len(days) - 1)
    ok = days[jc] == days + k
    out = np.full(len(x), np.nan)
    out[ok] = x[jc[ok]]
    return out


class Forecaster:
    lags = (1, 3, 7)
    train_days = 540
    vol_win = 14
    min_samples = 20
    rec_win = 60  # regime-C recent-window change quantiles (regime_change): provider repricing
    rec_min = 30  # is non-stationary; full-history quantiles under-cover (Discovery cov 0.58-0.63)
    tail_scale = {1: 0.95, 3: 1.0, 7: 1.0}  # h1 tail narrowing around q50 (confirmed, iter 5)
    sticky_share = 0.5
    kappa = 0.0

    # ---------------------------------------------------------------- regime A: AWS spot ----
    def _aws_features(self, days, lv):
        X = np.column_stack([lv - _shift(days, lv, -k) for k in self.lags])
        vol = pd.Series(np.abs(lv - _shift(days, lv, -1))).rolling(self.vol_win, min_periods=3).mean().to_numpy()
        return X, vol + 1e-4

    def _aws(self, series, origin, od, horizons):
        feats = {sid: (days, lv, *self._aws_features(days, lv)) for sid, (days, lv) in series.items()}
        rows = []
        for h in horizons:
            Xs, ys, ss = [], [], []
            for days, lv, X, vol in feats.values():
                y = _shift(days, lv, h) - lv
                m = (days + h <= od) & (days > od - self.train_days) & np.isfinite(y) & np.isfinite(X).all(1) & np.isfinite(vol)
                Xs.append(X[m]); ys.append(y[m]); ss.append(vol[m])
            X, y, s = np.vstack(Xs), np.concatenate(ys), np.concatenate(ss)
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            z = np.quantile((y - X @ beta) / s, QS)
            # h1 tail narrowing (horizon_structure, confirmed in iteration 5): the in-sample
            # standardized tails at h=1 are too wide relative to the true 1-day-ahead
            # distribution (Discovery h1 coverage 0.840 vs 0.80 target); scale both tails
            # around the unchanged q50 by 0.95 at h=1 only. q50, h3 and h7 unchanged.
            f = self.tail_scale.get(h, 1.0)
            z = z[1] + f * (z - z[1])
            for sid, (days, lv, Xa, vol) in feats.items():
                v = vol[-1] if np.isfinite(vol[-1]) else 1e-4
                q = lv[-1] + np.nan_to_num(Xa[-1]) @ beta + v * z
                rows.append((sid, origin, h, *np.exp(np.sort(q))))
        return rows

    # ------------------------------------------------------- regimes B and C: daily statistics ----
    def _daily(self, sid, meta, days, lv, ds, origin, od, horizons):
        sub = ds[(ds["source_id"] == meta.source_id) & (ds["gpu_model"] == meta.gpu_model)
                 & (ds["pricing_tier"] == meta.pricing_tier) & (ds["instrument_class"] == meta.instrument_class)]
        recent = np.diff(lv[days > od - 60])
        zero_share = float((recent == 0).mean()) if len(recent) else 0.0
        card = meta.instrument_class in ("neocloud_rate_card", "hyperscaler_list")
        width = 1.0
        if card and len(sub) > 10:
            rel = ((sub["p75_usd_per_gpu_hour"] - sub["p25_usd_per_gpu_hour"]) / sub["median_usd_per_gpu_hour"]).to_numpy()
            width = float(np.clip(rel[-1] / max(np.median(rel[-60:]), 1e-6), 0.6, 1.6))
        pull = 0.0
        if not card and self.kappa:
            fam = str(meta.gpu_model).split(" ")[0]
            c = ds[(ds["source_id"] != meta.source_id) & ds["gpu_model"].str.startswith(fam)
                   & (ds["pricing_tier"] == meta.pricing_tier) & (ds["instrument_class"] == meta.instrument_class)]
            c = c[c["date"] > (pd.Timestamp(origin) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")]
            if len(c):
                pull = float(np.log(c["median_usd_per_gpu_hour"].median()) - lv[-1])
        rows = []
        for h in horizons:
            ch = _shift(days, lv, h) - lv
            ch = ch[np.isfinite(ch)]
            # Regime-C recent-window change quantiles (regime_change): the marketplace_ask /
            # non-card spot series reprice in bursts (provider entries/exits), so the
            # full-history h-day change distribution under-covers the forward window
            # (Discovery coverage 0.58-0.63, q90 breach 25%). Use a trailing rec_win-day
            # window when it has enough samples; fall back to the full history otherwise.
            if not card and len(ch) >= self.rec_min:
                ch_r = ch[-self.rec_win:]
                q = np.quantile(ch_r, QS) if len(ch_r) >= self.min_samples else np.quantile(ch, QS)
            else:
                q = np.quantile(ch, QS) if len(ch) >= self.min_samples else np.zeros(3)
            if card and zero_share >= self.sticky_share:
                q[1] = 0.0
            q = q[1] + (q - q[1]) * np.array([width, 1.0, width])
            q = q + self.kappa * min(1.0, h / 7) * pull
            rows.append((sid, origin, h, *np.exp(lv[-1] + np.sort(q))))
        return rows

    def predict_origin(self, pack_slice: dict, origin_date: str, horizons, external_signals=None) -> pd.DataFrame:
        hist, targets, ds = pack_slice["series_history"], pack_slice["targets"], pack_slice["daily_statistics"]
        od = int(_days(pd.Series([origin_date]))[0])
        hist = hist[hist["series_id"].isin(targets["series_id"])]
        series = {sid: (_days(g["date"]), np.log(g["value"].to_numpy(float))) for sid, g in hist.groupby("series_id")}
        aws = {s: v for s, v in series.items() if s.startswith("aws_spot.")}
        rows = self._aws(aws, origin_date, od, horizons) if aws else []
        for meta in targets.itertuples():
            if meta.series_id in series and meta.series_id not in aws:
                days, lv = series[meta.series_id]
                rows += self._daily(meta.series_id, meta, days, lv, ds, origin_date, od, horizons)
        return pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
