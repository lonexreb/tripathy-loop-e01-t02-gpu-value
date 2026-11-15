"""Long-horizon AWS GPU spot forecaster: pooled per-instance-family 730-day empirical h-day
log-change quantiles + up-only cross-sectional mean-reversion shift at h>=90 (champion), plus
asymmetric upper-tail widening at long horizons: q90 <- q50 + f*(q90 - q50) with f=1.3 at h90
and 1.5 at h180, q10/q50 byte-identical to the champion. Rationale: the champion's pooled
730-day change quantiles under-cover the fat right tail of administered spot repricing at
90-180 days (Discovery q90 breach 24.6% overall / 32.4% at h180, coverage_80 0.664); the q90
breach penalty is 21% of the h180 loss, so widening only the upper quantile around the
unchanged center trades a small q90-width penalty for a large breach reduction."""
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
    tail_factor = {90: 1.3, 180: 1.5}  # upper-tail widening per horizon (absent = 1.0 = off)

    def _changes(self, series: dict, od: int, h: int, window: int = 0) -> np.ndarray:
        w = self.window_days if not window else window
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
                key = (inst, h)
                if key not in cache:
                    fam = {s: series[s] for s in groups.get(inst, [])}
                    ch = self._changes(fam, od, h)
                    if len(ch) < self.min_pairs:
                        ch = self._changes(series, od, h)
                    cache[key] = np.quantile(ch, QS) if len(ch) else np.zeros(3)
                q = np.sort(np.asarray(cache[key], dtype=float))
                # cross-sectional mean reversion: UP-only (below-median AZs revert up), long horizons
                if h >= self.cs_min_horizon and csmed.get(inst) is not None:
                    diff = csmed[inst] - lv
                    if diff > 0.0:  # priced below the family median -> expected to revert up
                        w = 0.7 if h >= 180 else 0.5  # reversion completes further out
                        q = q + np.clip(diff, 0.0, self.cs_clip) * w
                # asymmetric upper-tail widening: q10/q50 untouched, q90 stretched around q50.
                # Applied ONLY to below-family-median series (the same state the CS reversion
                # shift keys on): their upside tail is fat (reversion risk), while widening
                # above-median (premium p5) series measured loss-positive (dense backtest).
                f = self.tail_factor.get(int(h), 1.0)
                if f != 1.0 and h >= self.cs_min_horizon and csmed.get(inst) is not None:
                    diff0 = csmed[inst] - lv
                    if diff0 > 0.0:
                        q = np.array(q, dtype=float)
                        q[2] = q[1] + f * (q[2] - q[1])
                q = np.clip(np.sort(q), -1.5, 1.5)
                rows.append((sid, origin_date, h, *np.exp(lv + q)))
        return pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
