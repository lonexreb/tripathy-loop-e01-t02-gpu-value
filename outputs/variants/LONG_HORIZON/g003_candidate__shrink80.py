"""Long-horizon AWS GPU spot forecaster: champion logic (pooled per-instance-family empirical
730-day h-day log-change quantiles + cross-sectional up-only mean-reversion shift at h>=90)
with ONE change: at h90/h180 the family q50's deviation from zero change is shrunk by 20%
(q50_new = 0.8 * q50_old), pulling the center toward persistence, because the 730-day family
change distribution over-forecasts the center in drawdown cells (champion h180 bias_rel -0.23).
q10/q90 are left byte-identical (no width change). h30 untouched."""
import numpy as np
import pandas as pd

QS = (0.1, 0.5, 0.9)


def _days(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s, format="%Y-%m-%d").to_numpy("datetime64[D]").astype(np.int64))


class Forecaster:
    window_days = 730
    min_pairs = 30
    cs_clip = 0.5           # max magnitude of the log reversion shift
    cs_min_group = 3        # need >= 3 series in the family to form a median
    cs_min_horizon = 90     # apply the shift only at long horizons (h30 measured loss-negative)
    shrink_min_horizon = 90 # apply the q50 shrink only at long horizons
    shrink = 0.8            # q50_new = shrink * q50_old (0.8 = pull 20% toward persistence)

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
        cache, rows = {}, []
        for sid in targets["series_id"]:
            if sid not in last:
                continue
            inst = sid.split(".")[1]
            lv = last[sid]
            for h in horizons:
                h = int(h)
                key = (inst, h)
                if key not in cache:
                    ch = self._changes({s: series[s] for s in groups.get(inst, [])}, od, h)
                    if len(ch) < self.min_pairs:
                        ch = self._changes(series, od, h)
                    cache[key] = np.quantile(ch, QS) if len(ch) else np.zeros(3)
                q = np.sort(np.asarray(cache[key], dtype=float))
                # q50 shrink toward persistence at long horizons (center fix, width untouched)
                if h >= self.shrink_min_horizon:
                    q = q.copy()
                    q[1] = self.shrink * q[1]
                # cross-sectional mean reversion: UP-only (below-median AZs revert up), long horizons
                if h >= self.cs_min_horizon and csmed.get(inst) is not None:
                    diff = csmed[inst] - lv
                    if diff > 0.0:
                        ww = 0.7 if h >= 180 else 0.5  # reversion completes further out
                        q = q + np.clip(diff, 0.0, self.cs_clip) * ww
                q = np.clip(np.sort(q), -1.5, 1.5)
                rows.append((sid, origin_date, h, *np.exp(lv + q)))
        return pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
