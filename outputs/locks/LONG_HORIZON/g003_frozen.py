"""Long-horizon AWS GPU spot forecaster.

Base (iter-1 champion): pooled per-instance-family empirical h-day log-change
quantiles over the trailing 730 days, plus a cross-sectional mean-reversion
shift at long horizons (below-family-median AZs revert UP toward the median).

New this iteration (g003):
1. Per-series x family change-quantile blend (hierarchical pooling): each AZ's
   own 730-day h-day change quantiles are blended with its family's pooled
   quantiles (w=0.3 at h90, 0.5 at h180, 0 at h30; needs >= 30 own change
   pairs, else pure family). Mechanism: administered AZ-level price paths are
   persistent enough that own-history quantiles carry information the family
   pool averages away.
2. Upper-band widening of the blended quantiles: q90 += f * (q90 - q50) with
   f = 3.0 at h180 and 0.3 at h30/h90. Mechanism: the pooled 730-day family
   change quantiles under-disperse the family-heterogeneous drift realized
   over 30-180 days (premium families rise while older capacity falls), so
   the upper band breaches ~40% at h180; widening toward the realized
   dispersion cuts pinball loss and moves coverage toward 0.8. q10 is left
   byte-identical (lower-side widening measured loss-negative).
Fallbacks: family pool falls back to all-series pool under 30 pairs; per-series
blend falls back to pure family under 30 own pairs; series without history are
skipped (baseline fill). All outputs clipped to +/-1.5 log of the last value.
"""
import numpy as np
import pandas as pd

QS = (0.1, 0.5, 0.9)


def _days(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s, format="%Y-%m-%d").to_numpy("datetime64[D]").astype(np.int64))


class Forecaster:
    window_days = 730
    min_pairs = 30          # min pooled change pairs for a family quantile set
    cs_weight = 0.5        # fraction of the (positive) gap to the family median applied
    cs_clip = 0.5           # max magnitude of the log shift
    cs_min_group = 3        # need >= 3 series in the family to form a median
    cs_min_horizon = 90     # apply the shift only at long horizons (h30 measured loss-negative)
    ps_min_pairs = 30       # min OWN change pairs before per-series quantiles are blended in
    # per-series blend weight by horizon (0.0 = pure family; h30 gated off)
    ps_w = {30: 0.0, 90: 0.3, 180: 0.5}
    # upper-band widening factor (fraction of the q90-q50 gap added to q90), by horizon.
    # h180 widening measured strongly loss-NEGATIVE on the full 16-origin Discovery window
    # (dry run 1: h180 skill -0.115, mean_width_rel 2.93 at f=3.0) - the pseudo-origin scan
    # that suggested it sampled only 2023-09..2024-03. h30/h90 at f=0.3 measured positive
    # (slice deltas +0.0035 / +0.0115) and are kept.
    up_widen = {30: 0.3, 90: 0.3, 180: 0.0}
    clip_log = 1.5          # clip final log-change quantiles

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
        fam_cache, ser_cache, rows = {}, {}, []
        for sid in targets["series_id"]:
            if sid not in last:
                continue
            inst = sid.split(".")[1]
            lv = last[sid]
            for h in horizons:
                # --- family pooled h-day change quantiles (fallback: all series) ---
                if (inst, h) not in fam_cache:
                    ch = self._changes({s: series[s] for s in groups.get(inst, []) if s in series}, od, h)
                    if len(ch) < self.min_pairs:
                        ch = self._changes(series, od, h)
                    fam_cache[(inst, h)] = np.quantile(ch, QS) if len(ch) else np.zeros(3)
                qf = np.sort(np.asarray(fam_cache[(inst, h)], dtype=float))
                # --- per-series own change quantiles (blend in if enough own pairs) ---
                q = qf.copy()
                w = float(self.ps_w.get(int(h), 0.0))
                if w > 0.0 and sid in series:
                    if (sid, h) not in ser_cache:
                        c = self._changes({sid: series[sid]}, od, h)
                        ser_cache[(sid, h)] = np.quantile(c, QS) if len(c) >= self.ps_min_pairs else None
                    sq = ser_cache[(sid, h)]
                    if sq is not None:
                        sq = np.sort(np.asarray(sq, dtype=float))
                        q = w * sq + (1.0 - w) * qf
                # --- cross-sectional mean reversion: UP-only, long horizons ---
                if h >= self.cs_min_horizon and csmed.get(inst) is not None:
                    diff = csmed[inst] - lv
                    if diff > 0.0:  # priced below the family median -> expected to revert up
                        wcs = 0.7 if h >= 180 else 0.5  # reversion completes further out
                        q = q + np.clip(diff, 0.0, self.cs_clip) * wcs
                # --- upper-band widening toward realized drift dispersion (q10 untouched) ---
                f = float(self.up_widen.get(int(h), 0.0))
                q = np.asarray(q, dtype=float).copy()
                if f > 0.0 and q[2] > q[1]:
                    q[2] = q[2] + f * (q[2] - q[1])
                q = np.clip(np.sort(q), -self.clip_log, self.clip_log)
                rows.append((sid, origin_date, h, *np.exp(lv + q)))
        return pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
