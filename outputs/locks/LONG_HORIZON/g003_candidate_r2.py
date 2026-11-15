"""Long-horizon AWS GPU spot forecaster: seed empirical h-day log-change quantiles (pooled per
instance family over the trailing 730 days) plus a cross-sectional mean-reversion shift at long
horizons: AZ series priced ABOVE their instance-family cross-sectional median level are pulled
DOWN toward that median (down-only, capped), because AZ-level administered spot prices revert to
the family level over 180 days. Below-median series are left untouched (up-shift measured
loss-negative). Short horizons are gated off (h30/h90 measured negative/mixed)."""
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
    cs_min_horizon = 90     # apply the shift only at long horizons (h30 measured loss-negative)
    ps_weight = {90: 0.3, 180: 0.5}   # per-series x family change-quantile blend weight by horizon (0 at h30)
    ps_min_pairs = 30                 # min own change pairs before per-series quantiles are trusted
    ft_k = 1.0                        # family-trend q90 widening factor (up-side only, h>=90)
    ft_clip = 0.4                     # max family 90d trend (log) entering the widening
    ft_lookback = 90                  # trend lookback in days

    def _changes(self, series: dict, od: int, h: int) -> np.ndarray:
        out = []
        for days, lv in series.values():
            m = days >= od - self.window_days
            d, v = days[m], lv[m]
            idx = np.searchsorted(d, d + h)
            ok = (idx < len(d)) & (d[np.minimum(idx, len(d) - 1)] == d + h)
            out.append(v[idx[ok]] - v[ok])
        return np.concatenate(out) if out else np.empty(0)

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
        last = {sid: v[-1] for sid, (d, v) in series.items() if len(v)}
        # cross-sectional median log level per instance family
        csmed = {}
        for inst, sids in groups.items():
            lev = [last[s] for s in sids if s in last]
            csmed[inst] = float(np.median(lev)) if len(lev) >= self.cs_min_group else None
        # family recent 90d trend (median of per-series 90d log changes, point-in-time)
        famtr = {}
        for inst, sids in groups.items():
            tr = []
            for s in sids:
                if s not in last:
                    continue
                d, v = series[s]
                m = (d <= od) & (d >= od - self.ft_lookback - 30)
                dd, vv = d[m], v[m]
                if len(dd) == 0:
                    continue
                j = np.searchsorted(dd, od - self.ft_lookback)
                if j >= len(dd):
                    j = len(dd) - 1
                while j > 0 and dd[j] > od - self.ft_lookback:
                    j -= 1
                tr.append(vv[-1] - vv[j])
            famtr[inst] = float(np.median(tr)) if len(tr) >= self.cs_min_group else 0.0
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
                q = np.sort(np.asarray(cache[key], dtype=float))
                # per-series x family blend (h90/h180 only): each AZ's own 730d change
                # distribution is more informative than the family-pooled one; w=0 at h30
                w = self.ps_weight.get(h, 0.0)
                if w > 0.0:
                    chs = self._changes({sid: series[sid]}, od, h)
                    if len(chs) >= self.ps_min_pairs:
                        qps = np.sort(np.quantile(chs, QS))
                        q = np.sort(w * qps + (1.0 - w) * q)
                # cross-sectional mean reversion: UP-only (below-median AZs revert up), long horizons
                if h >= self.cs_min_horizon and csmed.get(inst) is not None:
                    diff = csmed[inst] - lv
                    if diff > 0.0:  # priced below the family median -> expected to revert up
                        w = 0.7 if h >= 180 else 0.5  # reversion completes further out (pseudo-origin scan)
                        q = q + np.clip(diff, 0.0, self.cs_clip) * w
                # family-trend q90 widening: rising families breach q90 at h90/h180
                # (administered repricing continues in the trend direction); up-side only,
                # q10 byte-identical (down-side measured loss-negative)
                if h >= 90:
                    tr = famtr.get(inst, 0.0)
                    if tr > 0.0:
                        q = q.copy()
                        q[2] = q[2] + self.ft_k * np.clip(tr, 0.0, self.ft_clip) * (h / 180.0)
                q = np.clip(np.sort(q), -1.5, 1.5)
                rows.append((sid, origin_date, h, *np.exp(lv + q)))
        return pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
