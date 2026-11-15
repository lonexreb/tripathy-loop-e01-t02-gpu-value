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
    vol_wins = (7, 14, 28)  # ensemble (iter 8): average quantiles across vol-window members;
    vol_win = 14  # single-window fallback if a member has no valid rows
    ens_horizons = (1, 3, 7)  # ensemble at every horizon: h3/h7 since iter 8; h1 since the
    # iteration-9 analysis (raw h1 ensemble widens coverage 0.803 -> 0.829, re-centered by
    # tail_scale 0.90; OOS h1 +0.0080 avg, 5/5 origins, holdout h1 +0.0134 measured)
    chain_horizons = (3, 7)  # iterated 1-day forecasts (horizon_structure): h3/h7 q50 = h chained
    tw_horizons = (1,)  # 270-day z window (ensemble/interval_calibration): at h1 ONLY the
    z_train_days = 270  # pooled standardised-residual quantiles are estimated on a 270-day
    # trailing window (beta stays 540-day; avgbeta was a wash, iter 8 analysis). Pseudo-origin
    # OOS (fit at origin-28d, score next 28 days, 6 Discovery origins, on top of this stack):
    # h1 6/6 origins ~-1.8% loss, CONFIRMED on holdout (+0.0058 h1 skill, iter 2); h3 was 6/6
    # OOS-positive but FALSIFIED on holdout (-0.0022 h3 skill, iter 2) -> tw_horizons = (1,);
    # h7 0/6 OOS (270-day fit over-weights the 1-day lag at long horizons) -> h7 keeps 540-day z.
    min_samples = 20
    rec_win = 60  # regime-C recent-window change quantiles (regime_change): provider repricing
    rec_min = 30  # is non-stationary; full-history quantiles under-cover (Discovery cov 0.58-0.63)
    tail_scale = {1: 0.90, 3: 0.93, 7: 1.0}  # h1 tail narrowing around q50 (0.95 confirmed iter 5;
    # 0.90 re-centers the h1 ensemble widening, OOS-optimal composition per the iter-9 analysis).
    # h3 both-tail narrowing 0.93 (interval_calibration, iter-3 analysis): the champion's h3
    # in-sample tails are too wide relative to the true 3-day-ahead distribution (Discovery h3
    # coverage 0.808 vs 0.80 target); pseudo-origin OOS (fit at origin-28d, score next 28 days,
    # 6 Discovery origins, 540-day z, ensemble members) shows f=0.93 cuts h3 loss ~1.3% at 6/6
    # origins (f=0.95 6/6, f=0.97 6/6, monotone toward 0.93); q50 unchanged (pure width lever).
    sticky_share = 0.5
    kappa = 0.0

    # ---------------------------------------------------------------- regime A: AWS spot ----
    def _aws_features(self, days, lv):
        X = np.column_stack([lv - _shift(days, lv, -k) for k in self.lags])
        vols = [pd.Series(np.abs(lv - _shift(days, lv, -1))).rolling(vw, min_periods=3).mean().to_numpy() + 1e-4
                for vw in self.vol_wins]
        return X, vols

    def _aws(self, series, origin, od, horizons):
        feats = {sid: (days, lv, *self._aws_features(days, lv)) for sid, (days, lv) in series.items()}
        # Pooled h1 OLS beta for the iterated (chained) h3/h7 point forecasts: the 1-day-ahead
        # regression is the best-calibrated machinery in the pack (h1 coverage 0.803, skill 0.435),
        # and chaining it h times compounds the momentum trend instead of extrapolating the
        # single h-day pooled regression (which over-weights the 1-day lag at long horizons).
        beta1 = self._aws_beta(feats, 1, od)
        rows = []
        for h in horizons:
            # One pooled-OLS member per vol window (ensemble, iter 8): the local-vol scale is
            # the only window-sensitive part of the machinery, and the pseudo-origin OOS
            # analysis (fit at origin-28d, score the next 28 days, 5 Discovery origins) shows
            # averaging the three quantile sets across vol_wins beats the single 14-day
            # member at every horizon (h3/h7 5/5 origins, +0.006/+0.006 avg skill) while
            # moving coverage toward 0.80. train_days stays 540 (the 270-day member is
            # mixed-to-negative at h7: beta270 over-weights the 1-day lag).
            members = []
            for vw in self.vol_wins:
                Xs, ys, ss = [], [], []
                for days, lv, X, vols in feats.values():
                    vol = vols[self.vol_wins.index(vw)]
                    y = _shift(days, lv, h) - lv
                    m = (days + h <= od) & (days > od - self.train_days) & np.isfinite(y) & np.isfinite(X).all(1) & np.isfinite(vol)
                    Xs.append(X[m]); ys.append(y[m]); ss.append(vol[m])
                if not Xs or not np.concatenate(ss).size:
                    continue
                X, y, s = np.vstack(Xs), np.concatenate(ys), np.concatenate(ss)
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                z = np.quantile((y - X @ beta) / s, QS)
                # 270-day z window at h1/h3 (ensemble/interval_calibration): the pooled
                # standardised-residual quantiles are estimated on a 270-day trailing window
                # (beta stays 540-day). OOS 6/6 origins at h1/h3; 90-day and {270,90} are worse
                # (optimum ~270); h7 keeps the 540-day z (270-day fit over-weights the 1-day
                # lag at long horizons, OOS-negative 6/6).
                if h in self.tw_horizons:
                    Xs2, ys2, ss2 = [], [], []
                    for days, lv, X, vols in feats.values():
                        vol = vols[self.vol_wins.index(vw)]
                        y2 = _shift(days, lv, h) - lv
                        m2 = (days + h <= od) & (days > od - self.z_train_days) & np.isfinite(y2) & np.isfinite(X).all(1) & np.isfinite(vol)
                        Xs2.append(X[m2]); ys2.append(y2[m2]); ss2.append(vol[m2])
                    if Xs2 and np.concatenate(ss2).size:
                        X2, y2, s2 = np.vstack(Xs2), np.concatenate(ys2), np.concatenate(ss2)
                        b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                        z = np.quantile((y2 - X2 @ b2) / s2, QS)
                # h1 tail narrowing (horizon_structure, confirmed in iteration 5): the in-sample
                # standardized tails at h=1 are too wide relative to the true 1-day-ahead
                # distribution (Discovery h1 coverage 0.840 vs 0.80 target); scale both tails
                # around the unchanged q50 by 0.95 at h=1 only. q50, h3 and h7 unchanged.
                f = self.tail_scale.get(h, 1.0)
                z = z[1] + f * (z - z[1])
                members.append((beta, z, vw))
            if not members or h not in self.ens_horizons:
                beta, z = self._aws_single(h, feats, od)
                members = [(beta, z, self.vol_win)]
            # Chained (iterated) point forecast for h in chain_horizons: g[0] = last log value,
            # g[j+1] = g[j] + beta1 . (g[j]-g[j-1], g[j]-g[j-3], g[j]-g[j-7]) with actual history
            # for negative j. The pseudo-origin OOS analysis (fit at origin-28d, score the next
            # 28 days, 5 Discovery origins) shows the chained q50 beats the direct h-day pooled
            # OLS q50 at h3 (+0.0015 avg, 5/5 origins) and h7 (+0.0011 avg, 5/5 origins) with
            # coverage and bias unchanged (pure centering gain); blends are dominated by the
            # pure chain. Falls back to the direct member q50 when the chain is not finite.
            for sid, (days, lv, Xa, vols) in feats.items():
                qs = []
                for beta, z, vw in members:
                    v = vols[self.vol_wins.index(vw)][-1]
                    v = v if np.isfinite(v) else 1e-4
                    q = lv[-1] + np.nan_to_num(Xa[-1]) @ beta + v * z
                    if h in self.chain_horizons:
                        cp = self._chain_point(days, lv, h, beta1)
                        if np.isfinite(cp):
                            q = q + (cp - (lv[-1] + np.nan_to_num(Xa[-1]) @ beta))
                    qs.append(q)
                q = np.mean(qs, axis=0)
                rows.append((sid, origin, h, *np.exp(np.sort(q))))
        return rows

    def _aws_beta(self, feats, h, od):
        """Pooled OLS beta for horizon h (all series pooled, trailing train_days)."""
        Xs, ys = [], []
        for days, lv, X, vols in feats.values():
            y = _shift(days, lv, h) - lv
            m = (days + h <= od) & (days > od - self.train_days) & np.isfinite(y) & np.isfinite(X).all(1)
            Xs.append(X[m]); ys.append(y[m])
        if not Xs or not np.concatenate(ys).size:
            return None
        return np.linalg.lstsq(np.vstack(Xs), np.concatenate(ys), rcond=None)[0]

    def _chain_point(self, days, lv, h, beta1):
        """Iterated 1-day forecast: g[0]=lv[-1], g[j+1]=g[j]+beta1.(d1,d3,d7) ending at g[j]."""
        if beta1 is None:
            return np.nan
        g = {0: float(lv[-1])}
        # actual log values at day t-7..t-1 (calendar-day lookup, NaN when absent)
        H = {}
        for off in range(-7, 0):
            target = days[-1] + off
            j = np.searchsorted(days, target)
            if j < len(days) and days[j] == target:
                H[off] = float(lv[j])
            else:
                H[off] = np.nan
        for j in range(h):
            cur = g[j]
            l1 = g[j-1] if j-1 >= 0 else H.get(j-1, np.nan)
            l3 = g[j-3] if j-3 >= 0 else H.get(j-3, np.nan)
            l7 = g[j-7] if j-7 >= 0 else H.get(j-7, np.nan)
            if not all(np.isfinite([cur, l1, l3, l7])):
                return np.nan
            step = beta1[0]*(cur-l1) + beta1[1]*(cur-l3) + beta1[2]*(cur-l7)
            g[j+1] = cur + step
        return g[h]

    def _aws_single(self, h, feats, od):
        Xs, ys, ss = [], [], []
        for days, lv, X, vols in feats.values():
            vol = vols[self.vol_wins.index(self.vol_win)]
            y = _shift(days, lv, h) - lv
            m = (days + h <= od) & (days > od - self.train_days) & np.isfinite(y) & np.isfinite(X).all(1) & np.isfinite(vol)
            Xs.append(X[m]); ys.append(y[m]); ss.append(vol[m])
        X, y, s = np.vstack(Xs), np.concatenate(ys), np.concatenate(ss)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        z = np.quantile((y - X @ beta) / s, QS)
        f = self.tail_scale.get(h, 1.0)
        z = z[1] + f * (z - z[1])
        return beta, z

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
