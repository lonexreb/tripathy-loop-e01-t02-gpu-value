"""Long-horizon AWS GPU spot forecaster: per-series x family BLENDED empirical h-day log-change
quantiles (730-day trailing window) plus the champion's cross-sectional up-only mean-reversion
shift at long horizons. Mechanism: each AZ's administered spot price has its own change
distribution (sticky/quiet AZs move less than the family pooled distribution), so blending the
per-series quantiles with the family-pooled ones (w=0.5 at h180, 0.3 at h90, 0 at h30) narrows and
re-centres the long-horizon band where a series' own history supports it; series with < 30 own
change pairs fall back to the family distribution. Derived az_median series are excluded from
family statistics (cross-sectional medians and pooled change quantiles) to avoid double-counting
the median AZ; az_median targets are forecast with family quantiles only (w=0)."""
import numpy as np
import pandas as pd

QS = (0.1, 0.5, 0.9)


def _days(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s, format="%Y-%m-%d").to_numpy("datetime64[D]").astype(np.int64))


class Forecaster:
    window_days = 730
    min_pairs = 30            # min family change pairs before falling back to the global pool
    ps_min_pairs = 30         # min OWN change pairs before per-series quantiles are trusted
    ps_w = {30: 0.0, 90: 0.3, 180: 0.3}   # per-series blend weight by horizon
    cs_clip = 0.5              # max magnitude of the log reversion shift
    cs_min_group = 3           # need >= 3 series in the family to form a median
    cs_min_horizon = 90        # apply the shift only at long horizons (h30 measured loss-negative)

    def _changes(self, pairs, od: int, h: int) -> np.ndarray:
        out = []
        for days, lv in pairs:
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
        # family statistics exclude derived az_median series (double-counts the median AZ)
        stat = {sid: sv for sid, sv in series.items() if ".az_median" not in sid}
        groups: dict[str, list] = {}
        for sid in stat:
            groups.setdefault(sid.split(".")[1], []).append(sid)
        # point-in-time last log value per series (hist is truncated at the origin)
        last = {sid: v[-1] for sid, (d, v) in series.items() if len(v)}
        # cross-sectional median log level per instance family (non-median members only)
        csmed = {}
        for inst, sids in groups.items():
            lev = [last[s] for s in sids if s in last]
            csmed[inst] = float(np.median(lev)) if len(lev) >= self.cs_min_group else None
        fam_cache, ps_cache, rows = {}, {}, []
        for sid in targets["series_id"]:
            if sid not in last:
                continue
            inst = sid.split(".")[1]
            lv = last[sid]
            is_med = ".az_median" in sid
            for h in horizons:
                key = (inst, int(h))
                if key not in fam_cache:
                    ch = self._changes([stat[s] for s in groups.get(inst, [])], od, int(h))
                    if len(ch) < self.min_pairs:
                        ch = self._changes(list(stat.values()), od, int(h))
                    fam_cache[key] = np.quantile(ch, QS) if len(ch) else np.zeros(3)
                qf = np.sort(np.asarray(fam_cache[key], dtype=float))
                q = qf
                # per-series x family blend (log space, whole quantile triple)
                w = self.ps_w.get(int(h), 0.0)
                if w > 0.0 and not is_med and sid in series:
                    if (sid, int(h)) not in ps_cache:
                        chs = self._changes([series[sid]], od, int(h))
                        ps_cache[(sid, int(h))] = (np.quantile(chs, QS)
                                                  if len(chs) >= self.ps_min_pairs else None)
                    qs = ps_cache[(sid, int(h))]
                    if qs is not None:
                        q = w * np.sort(np.asarray(qs, dtype=float)) + (1.0 - w) * qf
                q = np.sort(np.asarray(q, dtype=float))
                # cross-sectional mean reversion: UP-only (below-median AZs revert up), long horizons
                if h >= self.cs_min_horizon and csmed.get(inst) is not None:
                    diff = csmed[inst] - lv
                    if diff > 0.0:
                        wcs = 0.7 if h >= 180 else 0.5  # reversion completes further out
                        q = q + np.clip(diff, 0.0, self.cs_clip) * wcs
                q = np.clip(np.sort(q), -1.5, 1.5)
                rows.append((sid, origin_date, h, *np.exp(lv + q)))
        return pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
