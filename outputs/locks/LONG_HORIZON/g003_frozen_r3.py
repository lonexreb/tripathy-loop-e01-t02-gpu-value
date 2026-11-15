"""Long-horizon AWS GPU spot forecaster: iter-1 champion (pooled per-instance-family 730-day
empirical h-day log-change quantiles + cross-sectional below-median up-shift at h>=90) plus two
pack-only levers measured out-of-sample on Discovery pseudo-origins (2023-10..2024-08):
(1) per-series x family change-quantile blend at long horizons (w=0.3 at h90, w=0.5 at h180,
    only when the series itself has >= 30 change pairs in the 730-day window, else family):
    each AZ's own administered price path is more informative than the family pool at long
    horizons (measured +0.012 h90 / +0.009 h180 skill);
(2) family recent-trend drift shift at short/mid horizons only (h30/h90): the family median
    90-day log change, clipped to +/- 0.5 log and scaled by 0.15, is added to all three
    quantiles. Family-level drift (p4d rising, p5 falling through 2023-24) is the dominant q50
    error (44.9% of champion loss; p4d h180 q90 breach 66%, p5 h90 q10 breach 48%); at h180 the
    same shift measured loss-negative (trend extrapolation overshoots) and is gated off.
Exogenous news signals: falsified twice in prior episodes (direction-mixed) and this episode's
point-in-time search returned nothing usable; no external signal is used."""
import numpy as np
import pandas as pd

QS = (0.1, 0.5, 0.9)


def _days(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s, format="%Y-%m-%d").to_numpy("datetime64[D]").astype(np.int64))


class Forecaster:
    window_days = 730
    min_pairs = 30
    cs_weight = 0.5        # fraction of the (positive) gap to the family median that is applied
    cs_clip = 0.5           # max magnitude of the log shift
    cs_min_group = 3        # need >= 3 series in the family to form a median
    cs_min_horizon = 90     # apply the cs shift only at long horizons (h30 measured loss-negative)
    ps_w = {90: 0.3, 180: 0.5}   # per-series x family blend weight by horizon (0 elsewhere)
    trend_w = 0.15               # family 90d-trend shift weight (h30/h90 only)
    trend_W = 90                 # trailing window for the family trend
    trend_clip = 0.5             # max magnitude of the trend shift (log)
    trend_horizons = (90,)      # h30 measured Discovery-negative (-0.017 skill), h180 loss-negative

    def _changes(self, series: dict, od: int, h: int) -> np.ndarray:
        out = []
        for days, lv in series.values():
            m = days >= od - self.window_days
            d, v = days[m], lv[m]
            idx = np.searchsorted(d, d + h)
            ok = (idx < len(d)) & (d[np.minimum(idx, len(d) - 1)] == d + h)
            out.append(v[idx[ok]] - v[ok])
        return np.concatenate(out) if out else np.empty(0)

    def _famtrend(self, sids: list, series: dict, od: int) -> float:
        """Median over the family of (last log value) - (log value ~trend_W days before origin)."""
        ch = []
        for s in sids:
            if s not in series:
                continue
            d, v = series[s]
            m0 = d <= od - self.trend_W
            m1 = d <= od
            if not m0.any() or not m1.any():
                continue
            ch.append(float(v[m1][-1]) - float(v[m0][-1]))
        return float(np.median(ch)) if len(ch) >= self.cs_min_group else 0.0

    def predict_origin(self, pack_slice: dict, origin_date: str, horizons, external_signals=None) -> pd.DataFrame:
        hist, targets = pack_slice["series_history"], pack_slice["targets"]
        od = int(_days(pd.Series([origin_date]))[0])
        hist = hist[hist["series_id"].str.startswith("aws_spot.")]
        series = {sid: (_days(g["date"]), np.log(g["value"].to_numpy(float)))
                  for sid, g in hist.groupby("series_id")}
        groups: dict[str, list] = {}
        for sid in series:
            groups.setdefault(sid.split(".")[1], []).append(sid)
        # point-in-time last log value per series (hist is truncated at the origin)
        last = {}
        for sid, (d, v) in series.items():
            m = d <= od
            if m.any():
                last[sid] = float(v[m][-1])
        # cross-sectional median log level and 90-day trend per instance family
        csmed, trend = {}, {}
        for inst, sids in groups.items():
            lev = [last[s] for s in sids if s in last]
            csmed[inst] = float(np.median(lev)) if len(lev) >= self.cs_min_group else None
            trend[inst] = self._famtrend(sids, series, od)
        cache, rows = {}, []
        for sid in targets["series_id"]:
            if sid not in last:
                continue
            inst = sid.split(".")[1]
            lv = last[sid]
            for h in horizons:
                key = (inst, h)
                if key not in cache:
                    ch = self._changes({s: series[s] for s in groups.get(inst, [])}, od, h)
                    if len(ch) < self.min_pairs:
                        ch = self._changes(series, od, h)
                    cache[key] = np.quantile(ch, QS) if len(ch) else np.zeros(3)
                q = np.array(cache[key], dtype=float)
                # (1) per-series x family blend at long horizons (own path >= min_pairs pairs)
                w = self.ps_w.get(h, 0.0)
                if w > 0.0 and sid in series:
                    chs = self._changes({sid: series[sid]}, od, h)
                    if len(chs) >= self.min_pairs:
                        q = (1.0 - w) * q + w * np.quantile(chs, QS)
                # (2) cross-sectional mean reversion: UP-only, long horizons (iter-1 champion)
                if h >= self.cs_min_horizon and csmed.get(inst) is not None:
                    diff = csmed[inst] - lv
                    if diff > 0.0:  # priced below the family median -> expected to revert up
                        wcs = 0.7 if h >= 180 else 0.5  # reversion completes further out
                        q = q + np.clip(diff, 0.0, self.cs_clip) * wcs
                # (3) family recent-trend drift shift, short/mid horizons only
                if h in self.trend_horizons:
                    q = q + np.clip(trend.get(inst, 0.0), -self.trend_clip, self.trend_clip) * self.trend_w
                q = np.clip(np.sort(q), -1.5, 1.5)
                rows.append((sid, origin_date, h, *np.exp(lv + q)))
        return pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
