#!/usr/bin/env python
"""
Referee-requested experiments for paper revision.

1. Block-specific pooled AR(1)+FE with block-specific rho (no Stage 2)
2. Quarter-level DM-HAC inference
3. Formal estimator printout
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from harp.validation.metrics import oos_r_squared

INTENSITIES_PATH = PROJECT_ROOT / "data" / "intensities" / "experiment_a1_intensities.parquet"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registries" / "experiment_a1_registry.json"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2015, 2025))


def load_data():
    df = pd.read_parquet(INTENSITIES_PATH)
    with open(REGISTRY_PATH) as f:
        reg = json.load(f)
    meta = {a["actor_id"]: a for a in reg["actors"]}
    panel = df.pivot_table(index="period", columns="actor_id", values="intensity_value")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index().loc["2005-01-01":"2025-12-31"]
    return panel, meta


def _prepare_window(panel, ty, T_yr=5):
    ts = pd.Timestamp(f"{ty - T_yr}-01-01")
    if ts < pd.Timestamp("2005-01-01"):
        return None
    te = pd.Timestamp(f"{ty}-12-31")
    ad = panel[(panel.index >= ts) & (panel.index <= te)].copy()
    v = ad.columns[ad.notna().any()]
    ad = ad[v].fillna(ad[v].mean())
    N = len(v)
    if N < 10:
        return None
    tq = pd.date_range(f"{ty}-01-01", f"{ty}-12-31", freq="QS")
    otr = ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty - 1}-12-31"))
             ].values.astype(np.float64)
    if otr.shape[0] < 4:
        return None
    return ad, otr, tq, N, v


# ══════════════════════════════════════════════════════════
#  1. Block-specific pooled AR(1)+FE
# ══════════════════════════════════════════════════════════

def run_window_block_ar1(panel, ty, actor_block_map, T_yr=5):
    """Block-specific pooled AR(1)+FE: each block gets its own rho_b."""
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    v_list = list(v)

    def estimate_block_params(otr_data, v_list, actor_block_map):
        rho_vec = np.zeros(otr_data.shape[1])
        mean_vec = np.nan_to_num(otr_data.mean(axis=0), nan=0.5)
        blocks = {}
        for i, a in enumerate(v_list):
            b = actor_block_map.get(a, "REMAINDER")
            if b not in blocks:
                blocks[b] = []
            blocks[b].append(i)
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

    rho_vec, mean_vec = estimate_block_params(otr, v_list, actor_block_map)

    ps, ac = [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)
    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64)
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        pred = mean_vec + rho_vec * (prev - mean_vec)
        ps.append(pred)
        ac.append(obs)
        prev = obs
        otr = np.vstack([otr, qv])
        rho_vec, mean_vec = estimate_block_params(otr, v_list, actor_block_map)

    if not ps:
        return None
    return float(oos_r_squared(np.array(ps).ravel(), np.array(ac).ravel()))


# ══════════════════════════════════════════════════════════
#  2. DM-HAC inference
# ══════════════════════════════════════════════════════════

def dm_hac_test(deltas, bw=2):
    """Diebold-Mariano test with Newey-West HAC variance."""
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
    """Moving block bootstrap CI."""
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


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  REFEREE-REQUESTED EXPERIMENTS")
    print("=" * 70)

    panel, meta = load_data()
    actors = list(panel.columns)

    # Block definitions
    blocks_def = {
        "SEC_diversified": [a for a in actors if meta.get(a, {}).get("sector") == "diversified"],
        "LAYER_macro_inst": [a for a in actors if meta.get(a, {}).get("layer", -1) in (0, 1)],
        "MERGED_tech_health": [a for a in actors
                                if meta.get(a, {}).get("sector") in ("technology", "healthcare")],
    }
    local_set = set()
    for v in blocks_def.values():
        local_set.update(v)
    blocks_def["REMAINDER"] = [a for a in actors if a not in local_set]

    actor_block = {}
    for bname, bactors in blocks_def.items():
        for a in bactors:
            actor_block[a] = bname

    # Load stored results
    df_4b = pd.read_parquet(METRICS_DIR / "iter6_4b.parquet")
    g0_r2s = df_4b[df_4b["architecture"] == "G0"].sort_values("year")["full_r2"].values
    g1_r2s = df_4b[df_4b["architecture"] == "G1"].sort_values("year")["full_r2"].values
    m2_r2s = df_4b[df_4b["architecture"] == "M2"].sort_values("year")["full_r2"].values

    # ── 1. Block-specific AR(1)+FE ──
    print("\n--- 1. BLOCK-SPECIFIC POOLED AR(1)+FE (no Stage 2) ---")

    ba_r2s = []
    for ty in TEST_YEARS:
        r2 = run_window_block_ar1(panel, ty, actor_block)
        ba_r2s.append(r2 if r2 is not None else np.nan)
        print(f"  W{ty}: R2 = {r2:.4f}" if r2 else f"  W{ty}: FAILED")
    ba_r2s = np.array(ba_r2s)
    ba_mean = np.nanmean(ba_r2s)

    print(f"\n  ARCHITECTURE COMPARISON:")
    print(f"  {'Architecture':<42s} {'R2':>7s} {'D vs G1':>8s}")
    print(f"  {'-'*60}")
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
        print(f"  --> The gain is NOT just from heterogeneous persistence.")
        print(f"  --> Local Stage 2 residual dynamics add genuine value.")
    else:
        print(f"  --> Gain may be partly from heterogeneous persistence.")

    # Paired test: M2 vs block-specific AR1
    d_m2_ba = m2_r2s - ba_r2s
    valid = np.isfinite(d_m2_ba)
    d_clean = d_m2_ba[valid]
    t_stat = d_clean.mean() / (d_clean.std(ddof=1) / np.sqrt(len(d_clean)))
    p_val = 2 * t_dist.sf(abs(t_stat), len(d_clean) - 1)
    rng = np.random.default_rng(42)
    bs = np.array([rng.choice(d_clean, len(d_clean), replace=True).mean()
                    for _ in range(10000)])
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    wins = int((d_clean > 0).sum())
    print(f"  Paired test (M2 - block AR1): t={t_stat:.2f} p={p_val:.4f}")
    print(f"    CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  W={wins}/{len(d_clean)}")

    # ── 2. DM-HAC inference ──
    print("\n--- 2. DM-HAC INFERENCE (M2 vs G1) ---")

    deltas = m2_r2s - g1_r2s
    d_bar = deltas.mean()

    # Standard
    se_std = deltas.std(ddof=1) / np.sqrt(len(deltas))
    t_std = d_bar / se_std
    p_std = 2 * t_dist.sf(abs(t_std), len(deltas) - 1)
    rng2 = np.random.default_rng(42)
    bs_std = np.array([rng2.choice(deltas, len(deltas), replace=True).mean()
                        for _ in range(10000)])
    ci_std = (float(np.percentile(bs_std, 2.5)), float(np.percentile(bs_std, 97.5)))

    # HAC bandwidths 1, 2, 3
    print(f"  Mean Delta = {d_bar:+.4f}")
    print(f"  {'Method':<30s} {'t/DM':>7s} {'p':>7s} {'SE':>7s} {'CI':>23s}")
    print(f"  {'-'*78}")
    print(f"  {'Standard paired t':<30s} {t_std:>7.2f} {p_std:>7.4f} {se_std:>7.4f}"
          f" [{ci_std[0]:+.4f}, {ci_std[1]:+.4f}]")

    for bw in [1, 2, 3]:
        dm, p, se = dm_hac_test(deltas, bw=bw)
        print(f"  {'NW-HAC bw=' + str(bw):<30s} {dm:>7.2f} {p:>7.4f} {se:>7.4f}")

    for bl in [2, 3]:
        ci_bl = block_bootstrap_ci(deltas, block_len=bl)
        print(f"  {'Block bootstrap bl=' + str(bl):<30s} {'':>7s} {'':>7s} {'':>7s}"
              f" [{ci_bl[0]:+.4f}, {ci_bl[1]:+.4f}]")

    # Check: do ALL methods exclude zero?
    ci_bl2 = block_bootstrap_ci(deltas, block_len=2)
    ci_bl3 = block_bootstrap_ci(deltas, block_len=3)
    all_positive = ci_std[0] > 0 and ci_bl2[0] > 0 and ci_bl3[0] > 0
    print(f"\n  All CIs exclude zero: {all_positive}")

    # ── 3. Formal estimator ──
    print("\n--- 3. FORMAL ESTIMATOR DEFINITION ---")
    print("  y_{i,t+1} = alpha_i + rho * (y_{i,t} - alpha_i)")
    print("             + 1{i in b} * U_b * A_b * U_b' * r_{i,t}")
    print("             + epsilon_{i,t+1}")
    print("  where:")
    print("    rho:      pooled across all i (global Stage 1)")
    print("    alpha_i:  actor fixed effects")
    print("    b = b(i): pre-specified block (sector/layer)")
    print("    U_b:      block-specific PCA basis (K_b components)")
    print("    A_b:      block-specific ridge VAR on factors")
    print("    r_{i,t}:  Stage 1 residual = y_{i,t} - alpha_i - rho*(y_{i,t-1} - alpha_i)")
    print("")
    print("  Special cases:")
    print("    rho common, U_b = U global, A_b = A global  -->  G1 (global augmentation)")
    print("    rho common, U_b block-specific, A_b block-specific  -->  M2 (mixture)")
    print("    rho_b block-specific, no Stage 2  -->  Block-specific AR(1)+FE")
    print("")
    print("  Relation to existing estimators:")
    print("    - Interactive FE model (Bai 2009) with known groups")
    print("    - Grouped patterns (Bonhomme-Manresa 2015) with pre-specified membership")
    print("    - Forecast combination (Bates-Granger 1969) with block-diagonal weights")

    # ── Save ──
    rows = [
        {"architecture": "G0_global_pooled", "r2_mean": float(np.nanmean(g0_r2s))},
        {"architecture": "Block_AR1_FE", "r2_mean": float(ba_mean)},
        {"architecture": "G1_global_augmented", "r2_mean": float(np.nanmean(g1_r2s))},
        {"architecture": "M2_mixture", "r2_mean": float(np.nanmean(m2_r2s))},
    ]
    per_window = []
    for i, ty in enumerate(TEST_YEARS):
        per_window.append({
            "year": ty, "G0": float(g0_r2s[i]), "block_ar1": float(ba_r2s[i]),
            "G1": float(g1_r2s[i]), "M2": float(m2_r2s[i]),
        })
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(METRICS_DIR / "iter6_4b_referee_baselines.parquet", index=False)
    pd.DataFrame(per_window).to_parquet(METRICS_DIR / "iter6_4b_referee_perwindow.parquet", index=False)
    print(f"\nSaved: iter6_4b_referee_*.parquet")


if __name__ == "__main__":
    main()
