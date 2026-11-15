"""Long-horizon AWS GPU spot forecaster: champion logic (pooled per-instance-family empirical
h-day log-change quantiles + cross-sectional up-only mean-reversion shift at h>=90) with ONE
change: at h90/h180 the family pooled change quantiles use a trailing 365-day window instead of
730, so the quantile center tracks the family's recent regime instead of embedding the 2023 rise
(the 730-day window over-forecasts q50 in drawdown cells: champion h180 bias_rel -0.23).
h30 keeps the 730-day window (single-lever change). Fallback chain unchanged."""
import numpy as np
import pandas as pd

QS = (0.1, 0.5, 0.9)


def _days(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s, format="%Y-%m-%d").to_numpy("datetime64[D]").astype(np.int64))


class Forecaster:
    window_days = 730
    fam_window_long = 365   # family change-quantile window at h >= 90 (recent regime)
    min_pairs = 30
    cs_clip = 0.5
    cs_min_group = 3
    cs_min_horizon = 90

    def _changes(self, series: dict, od: int, h: int, window: int = None) -> np.ndarray:
        w = self.window_days if window is None else window
        out = []
        for days, lv in series.values():
            m = days >= od - w
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
        last = {sid: v[-1] for sid, (d, v) in series.items() if len(v)}
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
                    # recent-regime family window at long horizons, champion window at h30
                    fw = self.fam_window_long if h >= 90 else self.window_days
                    ch = self._changes({s: series[s] for s in groups.get(inst, [])}, od, h, window=fw)
                    if len(ch) < self.min_pairs:
                        ch = self._changes(series, od, h, window=fw)
                    cache[key] = np.quantile(ch, QS) if len(ch) else np.zeros(3)
                q = np.sort(np.asarray(cache[key], dtype=float))
                if h >= self.cs_min_horizon and csmed.get(inst) is not None:
                    diff = csmed[inst] - lv
                    if diff > 0.0:
                        ww = 0.7 if h >= 180 else 0.5
                        q = q + np.clip(diff, 0.0, self.cs_clip) * ww
                q = np.clip(np.sort(q), -1.5, 1.5)
                rows.append((sid, origin_date, h, *np.exp(lv + q)))
        return pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
