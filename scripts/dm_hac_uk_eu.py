#!/usr/bin/env python
"""
DM-HAC inference for UK/EU panel (experiment_b1) -- mirrors dm_hac_inference.py.

For Phase 8 of UK_EU_PANEL_PLAN.md. Computes:
  1. Block-specific pooled AR(1)+FE on B1 (sanity vs paper's BA result)
  2. DM-HAC (Newey-West, bw=1,2,3) for M2 vs G1 to test H1 with serial-corr-robust SE
  3. Block-bootstrap CI (bl=2,3) for the M2-G1 delta

Reads: results/metrics/table5_uk_eu.parquet (per-window R^2)
Writes: results/metrics/dm_hac_uk_eu.{parquet,json}
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist
warnings.filterwarnings("ignore")

import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from harp.validation.metrics import oos_r_squared

INTENSITIES_PATH = None
REGISTRY_PATH = None
TABLE5_BASENAME = None
OUT_BASENAME = None
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2017, 2025))
PANEL_START = "2011-04-01"
PANEL_END = "2025-12-31"


def _resolve_paths(variant: str, t_yr: int = 5, block_scheme: str = "us_inherited"):
    suffix = f"_{variant}" if variant else ""
    panel = f"experiment_b1{suffix}"
    out_suffix = "" if t_yr == 5 else f"_T{t_yr}"
    bs_suffix = "" if block_scheme == "us_inherited" else f"_{block_scheme}"
    return (
        PROJECT_ROOT / "data" / "intensities" / f"{panel}_intensities.parquet",
        PROJECT_ROOT / "data" / "registries" / f"{panel}_registry.json",
        f"table5_uk_eu{suffix}{out_suffix}{bs_suffix}",
        f"dm_hac_uk_eu{suffix}{out_suffix}{bs_suffix}",
    )


def load_data():
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    meta = {a["actor_id"]: a for a in reg["actors"]}
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc[PANEL_START:PANEL_END]
    return panel, meta


def _prepare_window(panel, ty, T_yr=5):
    ts = pd.Timestamp(f"{ty - T_yr}-01-01")
    te = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    v = ad.columns[ad.notna().any()]
    ad = ad[v].fillna(ad[v].mean())
    N = len(v)
    if N < 5:
        return None
    tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
    otr = ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty - 1}-12-31"))
             ].values.astype(np.float64)
    if otr.shape[0] < 4:
        return None
    return ad, otr, tq, N, v


def run_window_block_ar1(panel, ty, actor_block_map, T_yr=5):
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    v_list = list(v)

    def estimate_block_params(otr_data):
        rho_vec = np.zeros(otr_data.shape[1])
        mean_vec = np.nan_to_num(otr_data.mean(axis=0), nan=0.5)
        blocks = {}
        for i, a in enumerate(v_list):
            b = actor_block_map.get(a, "REMAINDER")
            blocks.setdefault(b, []).append(i)
        for b, idx in blocks.items():
            bd = otr_data[:, idx]
            by = np.nan_to_num(bd.mean(axis=0), nan=0.5)
            tl = bd - by
            nm = np.sum(tl[1:] * tl[:-1])
            dn = np.sum(tl[:-1] ** 2)
            rb = float(nm / dn) if dn > 1e-12 else 0.0
            for j, ii in enumerate(idx):
                rho_vec[ii] = rb
                mean_vec[ii] = by[j]
        return rho_vec, mean_vec

    rho_vec, mean_vec = estimate_block_params(otr)
    ps, ac = [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64) if qd in ad.index else np.zeros((0, N))
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        pred = mean_vec + rho_vec * (prev - mean_vec)
        ps.append(pred)
        ac.append(obs)
        prev = obs
        otr = np.vstack([otr, qv])
        rho_vec, mean_vec = estimate_block_params(otr)
    if not ps:
        return None
    return float(oos_r_squared(np.array(ps).ravel(), np.array(ac).ravel()))


def dm_hac_test(deltas, bw=2):
    n = len(deltas)
    d_bar = deltas.mean()
    gamma0 = np.var(deltas, ddof=1)
    gamma_sum = 0.0
    for h in range(1, bw + 1):
        if h >= n:
            break
        gamma_h = np.mean((deltas[h:] - d_bar) * (deltas[:-h] - d_bar))
        gamma_sum += 2 * (1 - h / (bw + 1)) * gamma_h
    var_hac = max(gamma0 + gamma_sum, 1e-12)
    se_hac = np.sqrt(var_hac / n)
    dm_stat = d_bar / se_hac
    p_val = 2 * t_dist.sf(abs(dm_stat), n - 1)
    return dm_stat, p_val, se_hac


def block_bootstrap_ci(deltas, block_len=2, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_means = []
    for _ in range(n_boot):
        n_blocks = (n + block_len - 1) // block_len
        starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
        sample = np.concatenate([deltas[s:s + block_len] for s in starts])[:n]
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def main():
    global INTENSITIES_PATH, REGISTRY_PATH, TABLE5_BASENAME, OUT_BASENAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="")
    parser.add_argument("--t-yr", type=int, default=5)
    parser.add_argument("--block-scheme", type=str, default="us_inherited",
                        choices=["us_inherited", "sector_split"])
    args = parser.parse_args()
    INTENSITIES_PATH, REGISTRY_PATH, TABLE5_BASENAME, OUT_BASENAME = _resolve_paths(
        args.variant, args.t_yr, args.block_scheme)

    print("=" * 70)
    print(f"  DM-HAC INFERENCE -- {INTENSITIES_PATH.stem}")
    print(f"  Variant: '{args.variant or 'full'}' | T_yr: {args.t_yr} | Block scheme: {args.block_scheme}")
    print(f"  Reads: {TABLE5_BASENAME}.parquet")
    print("=" * 70)

    panel, meta = load_data()
    actors = list(panel.columns)

    if args.block_scheme == "us_inherited":
        blocks_def = {
            "SEC_diversified": [a for a in actors if meta.get(a, {}).get("sector") == "diversified"],
            "LAYER_macro_inst": [a for a in actors if meta.get(a, {}).get("layer", -1) in (0, 1)],
            "MERGED_tech_health": [a for a in actors
                                    if meta.get(a, {}).get("sector") in ("technology", "healthcare")],
        }
    elif args.block_scheme == "sector_split":
        blocks_def = {
            "LAYER_macro_inst": [a for a in actors if meta.get(a, {}).get("layer", -1) in (0, 1)],
        }
        for sec in ["energy", "financials", "healthcare", "technology", "consumer", "industrials"]:
            blocks_def[f"SEC_{sec}"] = [a for a in actors
                                         if meta.get(a, {}).get("sector") == sec
                                         and meta.get(a, {}).get("layer", -1) == 2]
    else:
        raise ValueError(args.block_scheme)
    local_set = set()
    for v in blocks_def.values():
        local_set.update(v)
    blocks_def["REMAINDER"] = [a for a in actors if a not in local_set]

    actor_block = {}
    for bname, bactors in blocks_def.items():
        for a in bactors:
            actor_block[a] = bname

    df_t5 = pd.read_parquet(METRICS_DIR / f"{TABLE5_BASENAME}.parquet")

    # 1. Block-specific AR(1)+FE
    print("\n--- 1. BLOCK-SPECIFIC POOLED AR(1)+FE (no Stage 2) ---")
    ba_dict = {}
    for ty in TEST_YEARS:
        r2 = run_window_block_ar1(panel, ty, actor_block, T_yr=args.t_yr)
        ba_dict[ty] = r2
        print(f"  W{ty}: R2 = {r2:.4f}" if r2 else f"  W{ty}: FAILED")

    # Align all four series on the years where every architecture (incl. block-AR1) ran.
    df_t5_by_year = {arch: dict(zip(df_t5[df_t5["architecture"] == arch]["year"],
                                     df_t5[df_t5["architecture"] == arch]["full_r2"]))
                     for arch in ["G0", "G1", "M2"]}
    common_years = sorted([y for y in TEST_YEARS
                            if ba_dict.get(y) is not None
                            and y in df_t5_by_year["G0"]
                            and y in df_t5_by_year["G1"]
                            and y in df_t5_by_year["M2"]])
    print(f"\n  Aligned on {len(common_years)} common years: {common_years}")
    g0_r2s = np.array([df_t5_by_year["G0"][y] for y in common_years])
    g1_r2s = np.array([df_t5_by_year["G1"][y] for y in common_years])
    m2_r2s = np.array([df_t5_by_year["M2"][y] for y in common_years])
    ba_r2s = np.array([ba_dict[y] for y in common_years])
    ba_mean = float(np.nanmean(ba_r2s))

    print(f"\n  ARCHITECTURE COMPARISON:")
    print(f"  {'Architecture':<42s} {'R2':>7s} {'D vs G1':>8s}")
    print(f"  {'-' * 60}")
    for name, r2s in [
        ("G0 (global pooled, no S2)", g0_r2s),
        ("Block-specific AR(1)+FE (no S2)", ba_r2s),
        ("G1 (global pooled + global S2)", g1_r2s),
        ("M2 (global pooled + local S2)", m2_r2s),
    ]:
        m = np.nanmean(r2s)
        d = m - np.nanmean(g1_r2s)
        print(f"  {name:<42s} {m:7.4f} {d:+8.4f}")

    m2_vs_ba = np.nanmean(m2_r2s) - ba_mean
    print(f"\n  M2 vs block-specific AR(1): Delta = {m2_vs_ba:+.4f}")
    if m2_vs_ba > 0.01:
        print(f"  --> M2 exceeds block-specific rho by {m2_vs_ba:+.4f}")
        print(f"  --> Local Stage 2 residual dynamics add genuine value beyond heterogeneous persistence.")
    else:
        print(f"  --> Gain may be partly from heterogeneous persistence (small or negative).")

    d_m2_ba = m2_r2s - ba_r2s
    valid = np.isfinite(d_m2_ba)
    d_clean = d_m2_ba[valid]
    if len(d_clean) >= 2:
        t_stat = d_clean.mean() / (d_clean.std(ddof=1) / np.sqrt(len(d_clean)))
        p_val = 2 * t_dist.sf(abs(t_stat), len(d_clean) - 1)
        rng = np.random.default_rng(42)
        bs = np.array([rng.choice(d_clean, len(d_clean), replace=True).mean() for _ in range(10000)])
        ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        wins = int((d_clean > 0).sum())
        print(f"  Paired test (M2 - block AR1): t={t_stat:.2f} p={p_val:.4f}")
        print(f"    CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  W={wins}/{len(d_clean)}")

    # 2. DM-HAC for M2 vs G1
    print("\n--- 2. DM-HAC INFERENCE (M2 vs G1) ---")
    deltas = m2_r2s - g1_r2s
    d_bar = deltas.mean()
    se_std = deltas.std(ddof=1) / np.sqrt(len(deltas))
    t_std = d_bar / se_std
    p_std = 2 * t_dist.sf(abs(t_std), len(deltas) - 1)
    rng2 = np.random.default_rng(42)
    bs_std = np.array([rng2.choice(deltas, len(deltas), replace=True).mean() for _ in range(10000)])
    ci_std = (float(np.percentile(bs_std, 2.5)), float(np.percentile(bs_std, 97.5)))

    print(f"  Mean Delta = {d_bar:+.4f}  (n={len(deltas)} windows)")
    print(f"  {'Method':<30s} {'t/DM':>7s} {'p':>7s} {'SE':>7s} {'CI':>23s}")
    print(f"  {'-' * 78}")
    print(f"  {'Standard paired t':<30s} {t_std:>7.2f} {p_std:>7.4f} {se_std:>7.4f}"
          f" [{ci_std[0]:+.4f}, {ci_std[1]:+.4f}]")

    hac_results = {"standard": {"t": t_std, "p": p_std, "se": se_std,
                                 "ci_lo": ci_std[0], "ci_hi": ci_std[1]}}
    for bw in [1, 2, 3]:
        dm, p, se = dm_hac_test(deltas, bw=bw)
        print(f"  {'NW-HAC bw=' + str(bw):<30s} {dm:>7.2f} {p:>7.4f} {se:>7.4f}")
        hac_results[f"hac_bw{bw}"] = {"t": float(dm), "p": float(p), "se": float(se)}
    for bl in [2, 3]:
        ci_bl = block_bootstrap_ci(deltas, block_len=bl)
        print(f"  {'Block bootstrap bl=' + str(bl):<30s} {'':>7s} {'':>7s} {'':>7s}"
              f" [{ci_bl[0]:+.4f}, {ci_bl[1]:+.4f}]")
        hac_results[f"block_bs_bl{bl}"] = {"ci_lo": ci_bl[0], "ci_hi": ci_bl[1]}

    ci_bl2 = block_bootstrap_ci(deltas, block_len=2)
    ci_bl3 = block_bootstrap_ci(deltas, block_len=3)
    all_positive = ci_std[0] > 0 and ci_bl2[0] > 0 and ci_bl3[0] > 0
    print(f"\n  All CIs exclude zero: {all_positive}")

    # H1 verdict per pre-registration (one-sided alpha=0.05)
    one_sided_p_hac1 = hac_results["hac_bw1"]["p"] / 2 if d_bar > 0 else 1 - hac_results["hac_bw1"]["p"] / 2
    print(f"\n  One-sided HAC bw=1 p-value: {one_sided_p_hac1:.4f}"
          f"  ({'<' if one_sided_p_hac1 < 0.05 else '>='} 0.05)")
    h1_verdict = "SUPPORTED" if d_bar > 0 and one_sided_p_hac1 < 0.05 else "NOT SUPPORTED"
    print(f"  H1 (M2 > G1, one-sided alpha=0.05): {h1_verdict}")

    # Save
    rows = [
        {"architecture": "G0_global_pooled", "r2_mean": float(np.nanmean(g0_r2s))},
        {"architecture": "Block_AR1_FE", "r2_mean": float(ba_mean)},
        {"architecture": "G1_global_augmented", "r2_mean": float(np.nanmean(g1_r2s))},
        {"architecture": "M2_mixture", "r2_mean": float(np.nanmean(m2_r2s))},
    ]
    per_window = []
    for i, ty in enumerate(common_years):
        per_window.append({
            "year": ty, "G0": float(g0_r2s[i]), "block_ar1": float(ba_r2s[i]),
            "G1": float(g1_r2s[i]), "M2": float(m2_r2s[i]),
        })
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(METRICS_DIR / f"{OUT_BASENAME}_baselines.parquet", index=False)
    pd.DataFrame(per_window).to_parquet(METRICS_DIR / f"{OUT_BASENAME}_perwindow.parquet", index=False)
    summary = {
        "panel": "experiment_b1",
        "test_years": TEST_YEARS,
        "n_test_quarters": 4 * len(TEST_YEARS),
        "mean_delta_M2_vs_G1": float(d_bar),
        "h1_verdict": h1_verdict,
        "h1_one_sided_p_hac_bw1": float(one_sided_p_hac1),
        "all_cis_exclude_zero": bool(all_positive),
        "hac_results": hac_results,
        "architectures_mean_r2": {
            "G0": float(np.nanmean(g0_r2s)),
            "block_AR1_FE": float(ba_mean),
            "G1": float(np.nanmean(g1_r2s)),
            "M2": float(np.nanmean(m2_r2s)),
        },
    }
    with open(METRICS_DIR / f"{OUT_BASENAME}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {OUT_BASENAME}_*.parquet, {OUT_BASENAME}.json")


if __name__ == "__main__":
    main()
