"""MODEL-CLASS EXPLORATION (PACK_ONLY iter 10): model_combination (GBM x pooled-linear-momentum).

Regime A (AWS hyperscaler_spot, ~95% of pooled loss) is a from-scratch QUANTILE AVERAGE of two
forecasters from DIFFERENT classes, both fit point-in-time on the pooled AWS benchmark series
(trailing 540 days):
  1. gradient_boosting: sklearn HistGradientBoostingRegressor(loss='quantile') per quantile,
     features = trailing 1/3/7-day log changes + 7/14/28-day local vol. Nonlinear state->
     quantile map; alone it is +0.039 Discovery pooled vs the champion but -0.039 median
     (tree q50 drifts on calm sticky zones -> Gate-2 risk).
  2. pooled_linear_momentum (the champion's class, embedded): pooled OLS of the h-day log
     change on trailing 1/3/7-day log changes, vol-window ensemble {7,14,28} at h3/h7,
     single 14-day window + 0.95 tail narrowing at h1, vol-scaled conformal z bands.
The two quantile sets are averaged 50/50 in LOG space (q10/q50/q90 each), then clipped to
log(last) +/- 1.0. Mechanism: the tree carries the nonlinear/pooled gain (h1 +0.070, h3/h7
+0.036 Discovery skill), the linear member anchors the calm-zone center (median) and the
calibrated band; quantile averaging is the one width lever with a confirmed-positive record
(iters 8/9). Regimes B (rate cards) and C (marketplace / non-AWS spot) are the champion's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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
    clip = 1.0            # log-band clip around the last value
    w_gbm = 0.35           # blend weight on the GBM quantile set (1-w on the linear member)
    # GBM hyperparameters (fixed seed, deterministic)
    gb = dict(max_iter=200, learning_rate=0.06, max_leaf_nodes=31,
              min_samples_leaf=40, random_state=0, early_stopping=False)
    gbm_min_rows = 500    # pooled rows needed to trust a GBM fit
    per_series_min = 20   # per-series empirical fallback threshold
    # linear member (champion class) parameters
    vol_wins = (7, 14, 28)
    vol_win = 14
    ens_horizons = (3, 7)
    tail_scale = {1: 0.95, 3: 1.0, 7: 1.0}

    # ------------------------------------------------- regime A member 1: gradient boosting ----
    def _gbm_features(self, days, lv):
        ch = [lv - _shift(days, lv, -k) for k in self.lags]
        vols = [pd.Series(np.abs(lv - _shift(days, lv, -1))).rolling(vw, min_periods=3).mean().to_numpy() + 1e-4
                for vw in self.vol_wins]
        return np.column_stack(ch + vols)

    def _gbm_quantiles(self, feats, od, h):
        """{sid: q3} from pooled quantile-loss gradient boosting, plus fallback info."""
        Xs, ys, per = [], [], {}
        for sid, (days, lv, X, _vols) in feats.items():
            y = _shift(days, lv, h) - lv
            m = (days + h <= od) & (days > od - self.train_days) & np.isfinite(y) & np.isfinite(X).all(1)
            if m.any():
                Xs.append(X[m]); ys.append(y[m])
            per[sid] = (lv, X, y, (days + h <= od) & np.isfinite(y))
        models = None
        if Xs and int(sum(len(a) for a in ys)) >= self.gbm_min_rows:
            Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
            models = []
            for q in QS:
                g = HistGradientBoostingRegressor(loss="quantile", quantile=float(q), **self.gb)
                g.fit(Xtr, ytr)
                models.append(g)
        out = {}
        for sid, (lv, X, y, m2) in per.items():
            x0 = X[-1] if len(X) else np.full(len(self.lags) + len(self.vol_wins), np.nan)
            if models is not None and np.isfinite(x0).all():
                q = np.array([float(g.predict(x0.reshape(1, -1))[0]) for g in models])
            elif int(m2.sum()) >= self.per_series_min:
                q = np.quantile(y[m2], QS)
            else:
                q = np.zeros(3)
            out[sid] = np.sort(np.clip(q, -self.clip, self.clip))
        return out

    # ------------------------------- regime A member 2: pooled linear momentum (champion) ----
    def _lin_quantiles(self, feats, od, h):
        """{sid: q3} replicating the champion's regime-A machinery for horizon h."""
        Xs, ys, ss = [], [], []
        for days, lv, X, vols in feats.values():
            vol = vols[self.vol_wins.index(self.vol_win)]
            y = _shift(days, lv, h) - lv
            m = (days + h <= od) & (days > od - self.train_days) & np.isfinite(y) & np.isfinite(X).all(1) & np.isfinite(vol)
            Xs.append(X[m]); ys.append(y[m]); ss.append(vol[m])
        out = {}
        if not Xs or not np.concatenate(ss).size:
            return {sid: np.zeros(3) for sid in feats}
        X, y, s = np.vstack(Xs), np.concatenate(ys), np.concatenate(ss)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        z = np.quantile((y - X @ beta) / s, QS)
        f = self.tail_scale.get(h, 1.0)
        z = z[1] + f * (z - z[1])
        members = [(beta, z, self.vol_win)]
        if h in self.ens_horizons:
            members = []
            for vw in self.vol_wins:
                Xs2, ys2, ss2 = [], [], []
                for days, lv, Xa, vols in feats.values():
                    vol = vols[self.vol_wins.index(vw)]
                    y2 = _shift(days, lv, h) - lv
                    m = (days + h <= od) & (days > od - self.train_days) & np.isfinite(y2) & np.isfinite(Xa).all(1) & np.isfinite(vol)
                    Xs2.append(Xa[m]); ys2.append(y2[m]); ss2.append(vol[m])
                if not Xs2 or not np.concatenate(ss2).size:
                    continue
                X2, y2, s2 = np.vstack(Xs2), np.concatenate(ys2), np.concatenate(ss2)
                b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                z2 = np.quantile((y2 - X2 @ b2) / s2, QS)
                z2 = z2[1] + f * (z2 - z2[1])
                members.append((b2, z2, vw))
        for sid, (days, lv, X, vols) in feats.items():
            qs = []
            for beta, z, vw in members:
                v = vols[self.vol_wins.index(vw)][-1]
                v = v if np.isfinite(v) else 1e-4
                qs.append(np.nan_to_num(X[-1]) @ beta + v * z)
            out[sid] = np.mean(qs, axis=0)
        return out

    # ---------------------------------------------------------------- regime A: blend ----
    def _aws(self, series, origin, od, horizons):
        feats = {sid: (days, lv, self._gbm_features(days, lv),
                       [pd.Series(np.abs(lv - _shift(days, lv, -1))).rolling(vw, min_periods=3).mean().to_numpy() + 1e-4
                        for vw in self.vol_wins])
                 for sid, (days, lv) in series.items()}
        lin_feats = {sid: (days, lv, np.column_stack([lv - _shift(days, lv, -k) for k in self.lags]), vols)
                     for sid, (days, lv, _, vols) in feats.items()}
        rows = []
        for h in horizons:
            qg = self._gbm_quantiles(feats, od, h)
            ql = self._lin_quantiles(lin_feats, od, h)
            for sid, (days, lv, _, _) in feats.items():
                q = self.w_gbm * qg[sid] + (1.0 - self.w_gbm) * ql[sid]
                q = lv[-1] + np.sort(q)
                q = np.clip(q, lv[-1] - self.clip, lv[-1] + self.clip)
                rows.append((sid, origin, h, *np.exp(q)))
        return rows

    # ------------------------------------------------------- regimes B and C: daily statistics ----
    rec_win = 60
    rec_min = 30
    min_samples = 20
    sticky_share = 0.5
    kappa = 0.0

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
        rows = []
        for h in horizons:
            ch = _shift(days, lv, h) - lv
            ch = ch[np.isfinite(ch)]
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
