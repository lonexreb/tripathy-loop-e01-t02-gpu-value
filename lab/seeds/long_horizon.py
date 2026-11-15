"""Seed forecaster of the long-horizon track: last value scaled by empirical h-day log-change quantiles, pooled per
AWS instance type over the trailing two years (falls back to all AWS series). No momentum extrapolation."""
import numpy as np
import pandas as pd

QS = (0.1, 0.5, 0.9)


def _days(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s, format="%Y-%m-%d").to_numpy("datetime64[D]").astype(np.int64))


class Forecaster:
    window_days = 730
    min_pairs = 30

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
        series = {sid: (_days(g["date"]), np.log(g["value"].to_numpy(float))) for sid, g in hist.groupby("series_id")}
        groups: dict[str, dict] = {}
        for sid, v in series.items():
            groups.setdefault(sid.split(".")[1], {})[sid] = v
        cache, rows = {}, []
        for sid in targets["series_id"]:
            if sid not in series:
                continue
            inst = sid.split(".")[1]
            for h in horizons:
                key = (inst, h)
                if key not in cache:
                    ch = self._changes(groups[inst], od, h)
                    if len(ch) < self.min_pairs:
                        ch = self._changes(series, od, h)
                    cache[key] = np.quantile(ch, QS) if len(ch) else np.zeros(3)
                q = np.clip(np.sort(cache[key]), -1.5, 1.5)
                rows.append((sid, origin_date, h, *np.exp(series[sid][1][-1] + q)))
        return pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
