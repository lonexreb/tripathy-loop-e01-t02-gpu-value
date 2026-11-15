"""2-3 year (2027-2029) rental and secondary-resale valuation of H100 SXM 80GB and A100 SXM 80GB from the pack.

Inputs, all read from the pack:
  * current rental levels at the 2026-11-06 cutoff (daily_statistics: neocloud_rate_card on-demand and
    marketplace_ask medians across collectors);
  * the MLPerf capability ladder (capability_units: measured A100 -> H100 cross-benchmark median 2.465x, H100 ->
    B200 / B300 measured chains) and the Vera Rubin datasheet deflator (hardware_fundamentals FP8 VR200 / B300);
  * 4+ years of AWS EC2 spot ticks (p4d A100 40GB since 2022-05, p5 H100 since 2023-07): log-linear price decay;
  * a datacenter OpEx power floor from TDP (hardware_fundamentals) x node overhead x PUE x electricity + ops.
Scenario parameters are explicit constants below; every derived number is written to outputs/valuation.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from lab.config import FINAL_ORIGIN, LabConfig, load_config

GPUS = {"H100_SXM_80GB": {"model": "H100 SXM 80GB", "aws": "p5.48xlarge", "hw": "NVIDIA H100 SXM5 80GB"},
        "A100_SXM_80GB": {"model": "A100 SXM 80GB", "aws": "p4d.24xlarge", "hw": "NVIDIA A100 SXM4 80 GB"}}
NODE_OVERHEAD = 1.5          # CPUs, NICs, fans, storage per GPU relative to the GPU TDP
PUE = (1.25, 1.45)           # facility power usage effectiveness band
POWER_USD_KWH = (0.07, 0.11)
OPS_USD_GPU_HR = (0.05, 0.08)  # colo space, maintenance, staff, insurance per GPU-hour
HOURS_PER_YEAR = 8760
SCENARIOS = {  # utilisation, discount rate, decay multiplier on the empirically fitted rate
    "bear": {"utilisation": 0.55, "discount": 0.15, "decay_mult": 1.4},
    "base": {"utilisation": 0.70, "discount": 0.12, "decay_mult": 1.0},
    "bull": {"utilisation": 0.85, "discount": 0.10, "decay_mult": 0.6},
}
LIFE_END_YEAR = 2032.0
VALUATION_DATES = (2027.0, 2028.0, 2029.0)
T0 = 2026 + (310 / 365)      # 2026-11-06 as a decimal year


def _years(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    return (d.dt.year + (d.dt.day_of_year - 1) / 365.0).to_numpy()


def current_levels(pack: Path) -> dict:
    ds = pd.read_csv(pack / "daily_statistics.csv", dtype={"date": str})
    ds = ds[(ds["date"] > "2026-10-23") & (ds["date"] <= FINAL_ORIGIN)]
    out = {}
    for key, g in GPUS.items():
        x = ds[ds["gpu_model"] == g["model"]]
        lv = lambda cls, tier: float(x[(x["instrument_class"] == cls) & (x["pricing_tier"] == tier)]["median_usd_per_gpu_hour"].median())
        out[key] = {"neocloud_rate_card_on_demand": lv("neocloud_rate_card", "on_demand"),
                    "marketplace_ask_on_demand": lv("marketplace_ask", "on_demand"),
                    "hyperscaler_spot": lv("hyperscaler_spot", "spot"),
                    "hyperscaler_list_on_demand": lv("hyperscaler_list", "on_demand")}
    for m in ("B200", "B300", "H200"):
        x = ds[(ds["gpu_model"] == m) & (ds["instrument_class"] == "neocloud_rate_card") & (ds["pricing_tier"] == "on_demand")]
        out[m] = {"neocloud_rate_card_on_demand": float(x["median_usd_per_gpu_hour"].median())}
    return out


def capability(pack: Path) -> dict:
    u = pd.read_csv(pack / "capability" / "capability_units.csv")
    cu = lambda s, g: float(u[(u["benchmark_set_id"] == s) & (u["gpu_generation"] == g)]["capability_units_per_accelerator"].iloc[0])
    hw = pd.read_csv(pack / "hardware_fundamentals" / "hardware_fundamentals.csv")
    fp8 = lambda name: float(hw[hw["accelerator"].str.startswith(name)]["fp8_tflops"].iloc[0])
    return {"a100_to_h100_measured": cu("ship_only_inf_cross_benchmark_median_a100ref_sxm80", "H100"),
            "h100_to_b200_measured": cu("ship_only_inf_cross_benchmark_median_h100ref_sxm80", "B200"),
            "h100_to_b300_measured": cu("ship_only_inf_cross_benchmark_median_h100ref_sxm80", "B300"),
            "b300_to_vr200_datasheet_fp8": fp8("NVIDIA VR200") / fp8("NVIDIA B300"),
            "vr200_release_date": str(hw[hw["accelerator"].str.startswith("NVIDIA VR200")]["release_date"].iloc[0]),
            "tdp_w": {k: float(hw[hw["accelerator"] == g["hw"]]["tdp_w"].iloc[0]) for k, g in GPUS.items()}}


def aws_decay(pack: Path) -> dict:
    import duckdb

    insts = ", ".join(f"'{g['aws']}'" for g in GPUS.values())
    q = f"""SELECT instance_type, date_trunc('month', CAST(ts_utc AS DATE)) AS m, median(usd_per_gpu_hour) AS p
            FROM read_parquet('{pack / "aws_spot_ticks"}/*.parquet')
            WHERE instance_type IN ({insts}) AND product_description = 'Linux/UNIX' GROUP BY 1, 2 ORDER BY 1, 2"""
    df = duckdb.connect().execute(q).df()
    out = {}
    for key, g in GPUS.items():
        x = df[df["instance_type"] == g["aws"]]
        t, lp = _years(x["m"]), np.log(x["p"].to_numpy())
        fit = lambda m: float(-np.polyfit(t[m], lp[m], 1)[0])
        pre = t < 2026.2
        out[key] = {"instance_type": g["aws"], "first_month": str(x["m"].min())[:7], "last_month": str(x["m"].max())[:7],
                    "first_price": round(float(x["p"].iloc[0]), 3), "last_price": round(float(x["p"].iloc[-1]), 3),
                    "decay_per_year_full": fit(np.ones_like(t, bool)), "decay_per_year_pre_2026": fit(pre),
                    "monthly": {str(m)[:7]: round(float(p), 4) for m, p in zip(x["m"], x["p"])}}
    return out


def opex_floor(tdp_w: float) -> tuple[float, float]:
    kw = tdp_w / 1000 * NODE_OVERHEAD
    return kw * PUE[0] * POWER_USD_KWH[0] + OPS_USD_GPU_HR[0], kw * PUE[1] * POWER_USD_KWH[1] + OPS_USD_GPU_HR[1]


def price_paths(t: np.ndarray, lv: dict, cap: dict, dec: dict, floors: dict, mult: float) -> dict[str, np.ndarray]:
    """Capability-parity rental paths.

    H100: converges to the price per capability unit of the newest part it competes with (B200 neocloud price /
    measured H100->B200 ratio), which itself deflates as Vera Rubin rolls out; the H100's premium over parity decays
    at the empirical AWS p5 rate. A100: priced off the H100 path through the measured 2.465x capability ratio; its
    current premium over that parity decays at the empirical AWS p4d rate. Both are floored at the cash-cost OpEx.
    """
    dt_ = t - T0
    rubin = math.log(cap["b300_to_vr200_datasheet_fp8"]) / 4 * mult
    k_h = 0.5 * (dec["H100_SXM_80GB"]["decay_per_year_full"] + dec["H100_SXM_80GB"]["decay_per_year_pre_2026"]) * mult
    k_a = 0.5 * (dec["A100_SXM_80GB"]["decay_per_year_full"] + dec["A100_SXM_80GB"]["decay_per_year_pre_2026"]) * mult
    p0_h, p0_a = lv["H100_SXM_80GB"]["neocloud_rate_card_on_demand"], lv["A100_SXM_80GB"]["neocloud_rate_card_on_demand"]
    parity_h = lv["B200"]["neocloud_rate_card_on_demand"] / cap["h100_to_b200_measured"] * np.exp(-rubin * dt_)
    h = parity_h + (p0_h - parity_h[0]) * np.exp(-k_h * dt_)
    r0 = p0_a * cap["a100_to_h100_measured"] / p0_h
    a = h / cap["a100_to_h100_measured"] * (1 + (r0 - 1) * np.exp(-k_a * dt_))
    return {"H100_SXM_80GB": np.maximum(h, floors["H100_SXM_80GB"]), "A100_SXM_80GB": np.maximum(a, floors["A100_SXM_80GB"])}


def dcf_value(t_from: float, t: np.ndarray, p: np.ndarray, opex: float, s: dict) -> float:
    """PV at t_from of monthly net rental cash flows until LIFE_END_YEAR or shutdown (price x utilisation <= OpEx)."""
    m = t >= t_from - 1e-9
    cf = np.maximum(p[m] * s["utilisation"] - opex, 0.0) * HOURS_PER_YEAR / 12
    return float((cf * (1 + s["discount"]) ** -(t[m] - t_from)).sum())


def run(cfg: LabConfig | None = None) -> dict:
    cfg = cfg or load_config()
    pack = cfg.pack_dir
    lv, cap, dec = current_levels(pack), capability(pack), aws_decay(pack)
    res = {"as_of": FINAL_ORIGIN, "levels": lv, "capability": cap, "aws_decay": dec, "assumptions": {
        "node_overhead": NODE_OVERHEAD, "pue": PUE, "power_usd_kwh": POWER_USD_KWH, "ops_usd_gpu_hr": OPS_USD_GPU_HR,
        "scenarios": SCENARIOS, "life_end_year": LIFE_END_YEAR}, "gpus": {}}
    t = np.arange(T0, LIFE_END_YEAR, 1 / 12)
    opex = {k: opex_floor(cap["tdp_w"][k]) for k in GPUS}
    mid = {k: (lo + hi) / 2 for k, (lo, hi) in opex.items()}
    res["capability_parity_now"] = {
        "H100_vs_B200_usd_gpu_hr": lv["B200"]["neocloud_rate_card_on_demand"] / cap["h100_to_b200_measured"],
        "A100_vs_H100_usd_gpu_hr": lv["H100_SXM_80GB"]["neocloud_rate_card_on_demand"] / cap["a100_to_h100_measured"]}
    for key in GPUS:
        res["gpus"][key] = {"p0_neocloud_on_demand": lv[key]["neocloud_rate_card_on_demand"],
                            "opex_floor_band": [round(x, 3) for x in opex[key]], "opex_mid": round(mid[key], 3), "scenarios": {}}
    for name, s in SCENARIOS.items():
        paths = price_paths(t, lv, cap, dec, mid, s["decay_mult"])
        for key in GPUS:
            p = paths[key]
            rows = []
            for ty in VALUATION_DATES:
                i = int(np.argmin(np.abs(t - ty)))
                v = dcf_value(ty, t, p, mid[key], s)
                rows.append({"year": int(ty), "rental_usd_gpu_hr": round(float(p[i]), 3),
                             "net_cash_usd_gpu_hr": round(float(p[i]) * s["utilisation"] - mid[key], 3),
                             "resale_value_usd_per_gpu": round(v, 0), "resale_value_usd_per_8gpu_node": round(8 * v, 0)})
            eff = -math.log(p[int(np.argmin(np.abs(t - 2029.0)))] / p[0]) / (2029.0 - T0)
            res["gpus"][key]["scenarios"][name] = {"effective_decay_per_year": round(eff, 4),
                                                   "value_now_usd_per_gpu": round(dcf_value(T0, t, p, mid[key], s), 0), "path": rows}
    (cfg.outputs_dir / "valuation.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def markdown(res: dict) -> str:
    cap = res["capability"]
    lines = [
        "### Inputs read from the pack",
        f"- Measured MLPerf capability: A100 -> H100 = **{cap['a100_to_h100_measured']:.3f}x** (cross-benchmark median, SXM 80GB, "
        f"SHIP-only); H100 -> B200 = {cap['h100_to_b200_measured']:.2f}x, H100 -> B300 = {cap['h100_to_b300_measured']:.2f}x. "
        f"Vera Rubin datasheet deflator (FP8 VR200 / B300) = {cap['b300_to_vr200_datasheet_fp8']:.2f}x (release {cap['vr200_release_date']}).",
    ]
    for k, d in res["aws_decay"].items():
        lines.append(f"- AWS spot {d['instance_type']} ({k}): {d['first_month']} ${d['first_price']}/GPU-hr -> {d['last_month']} "
                     f"${d['last_price']}; fitted log decay {d['decay_per_year_full']:.1%}/yr over the full archive, "
                     f"{d['decay_per_year_pre_2026']:.1%}/yr before the 2026 rebound.")
    for k, g in res["gpus"].items():
        lines.append(f"- {k}: neocloud on-demand median ${g['p0_neocloud_on_demand']:.2f}/GPU-hr ({res['as_of']} window); "
                     f"OpEx power floor ${g['opex_floor_band'][0]:.2f}-${g['opex_floor_band'][1]:.2f}/GPU-hr.")
    cp = res["capability_parity_now"]
    lines.append(f"- Capability parity today: an H100 priced like a B200 per measured capability unit would rent at "
                 f"${cp['H100_vs_B200_usd_gpu_hr']:.2f}/GPU-hr; an A100 priced like an H100 per unit at ${cp['A100_vs_H100_usd_gpu_hr']:.2f}. "
                 "Both older parts trade at a premium to parity; the model decays that premium at the AWS empirical rate of "
                 "the same part while the parity anchor itself deflates with the Rubin datasheet uplift spread over 4 years.")
    lines += ["", "### Rental price and secondary resale value (DCF of remaining rental cash flows, per GPU)", "",
              "| GPU | scenario | eff. decay/yr to 2029 | value now | 2027 rent | 2027 resale | 2028 rent | 2028 resale | 2029 rent | 2029 resale |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for k, g in res["gpus"].items():
        for name, s in g["scenarios"].items():
            p = {r["year"]: r for r in s["path"]}
            lines.append(f"| {k} | {name} | {s['effective_decay_per_year']:.1%} | ${s['value_now_usd_per_gpu']:,.0f} | "
                         + " | ".join(f"${p[y]['rental_usd_gpu_hr']:.2f} | ${p[y]['resale_value_usd_per_gpu']:,.0f}" for y in (2027, 2028, 2029)) + " |")
    a = res["assumptions"]
    lines += ["", f"Assumptions: node overhead {a['node_overhead']}x TDP, PUE {a['pue'][0]}-{a['pue'][1]}, power "
              f"${a['power_usd_kwh'][0]}-${a['power_usd_kwh'][1]}/kWh, ops ${a['ops_usd_gpu_hr'][0]}-${a['ops_usd_gpu_hr'][1]}/GPU-hr; "
              "utilisation / discount rate: bear 55% / 15%, base 70% / 12%, bull 85% / 10%; cash flows stop when price x "
              f"utilisation falls below OpEx or at {int(a['life_end_year'])}; an 8-GPU node is 8x the per-GPU value "
              "(chassis and networking residual ignored). Resale values are a buyer's DCF, not observed prints: the pack's "
              "resale tables are pointer-only (no values), so these numbers are model outputs to be checked against them."]
    return "\n".join(lines)


if __name__ == "__main__":
    r = run()
    print(markdown(r))
