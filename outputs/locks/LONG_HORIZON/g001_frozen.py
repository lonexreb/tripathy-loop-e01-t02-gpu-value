"""Long-horizon candidate A: regime-adaptive change quantiles.

Seed champion used fixed 730-day pooled h-day log-change quantiles (q50 ~ persistence).
This candidate blends the long (730d) and recent (180d) empirical change quantiles per
AWS instance type (w_recent = 0.5) so the center/band track the prevailing price regime,
and adds a capped cross-sectional mean-reversion shift of the whole quantile triple toward
the instance-type cross-sectional median level (AZs below the pooled median revert up).
Point-in-time: only rows dated on or before the origin are used; no hard-coded dates.
"""
import numpy as np
import pandas as pd

QS = (0.1, 0.5, 0.9)


def _days(s: pd.Series) -> np.ndarray:
    return (pd.to_datetime(s, format="%Y-%m-%d").to_numpy("datetime64[D]").astype(np.int64))


class Forecaster:
    window_long = 730
    window_recent = 180
    w_recent = 0.5
    min_pairs = 30
    beta_cs = 0.25   # cross-sectional reversion coefficient (log units)
    cap_cs = 0.25    # cap on the reversion shift (log units)
    # horizon gating: regime-adaptive levers only at long horizons (h >= 90);
    # h30 keeps the seed champion's long-window quantiles (backtest: blend -0.012,
    # cs-shift hurts h30; dry-run A h30 delta was -0.055 with both levers on).
    min_h_adaptive = 90

    def _changes(self, series: dict, od: int, h: int, window: int) -> np.ndarray:
        out = []
        for days, lv in series.values():
            m = days >= od - window
            d, v = days[m], lv[m]
            if len(d) == 0:
                continue
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
        groups: dict[str, dict] = {}
        for sid, v in series.items():
            groups.setdefault(sid.split(".")[1], {})[sid] = v
        # last log-levels and cross-sectional median level per instance type
        last_lvl = {sid: v[1][-1] for sid, v in series.items() if len(v[1])}
        csmed: dict[str, float] = {}
        for inst, g in groups.items():
            lv = [v[1][-1] for v in g.values() if len(v[1])]
            if lv:
                csmed[inst] = float(np.median(lv))
        cache, rows = [], []
        for sid in targets["series_id"]:
            if sid not in series or sid not in last_lvl:
                continue
            inst = sid.split(".")[1]
            lv_last = last_lvl[sid]
            # cross-sectional reversion shift (log units), capped; long horizons only
            for h in horizons:
                key = (inst, h)
                adaptive = h >= self.min_h_adaptive
                if key not in cache:
                    ch_l = self._changes(groups[inst], od, h, self.window_long)
                    if len(ch_l) < self.min_pairs:
                        ch_l = self._changes(series, od, h, self.window_long)
                    q_l = np.quantile(ch_l, QS) if len(ch_l) else np.zeros(3)
                    if adaptive:
                        ch_r = self._changes(groups[inst], od, h, self.window_recent)
                        if len(ch_r) < self.min_pairs:
                            ch_r = self._changes(series, od, h, self.window_recent)
                        q_r = np.quantile(ch_r, QS) if len(ch_r) else q_l
                        q = self.w_recent * q_r + (1.0 - self.w_recent) * q_l
                    else:
                        q = q_l
                    cache[key] = q
                shift = 0.0
                if adaptive and inst in csmed:
                    shift = float(np.clip(self.beta_cs * (csmed[inst] - lv_last), -self.cap_cs, self.cap_cs))
                q = np.clip(np.sort(cache[key]) + shift, -1.5, 1.5)
                rows.append((sid, origin_date, h, *np.exp(lv_last + q)))
        return pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
