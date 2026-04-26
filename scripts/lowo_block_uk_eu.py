#!/usr/bin/env python
"""
Leave-One-Window-Out block selection for UK/EU panel (experiment_b1).
Mirrors lowo_block_selection.py.

For each test year, select which blocks get local treatment using ONLY
the per-block Delta = M2 - G1 from the OTHER 7 windows. Then run the
mixture with only those selected blocks. Compare to fixed (all-blocks)
mixture.

If LOWO Delta ~ fixed Delta -> block selection is not contaminated by
test-window data. If LOWO Delta << fixed Delta -> block selection was
informed by test data.
"""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import t as t_dist
warnings.filterwarnings("ignore")

import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from harp.spectral.dmd import ExactDMDDecomposer
from harp.validation.metrics import oos_r_squared

INTENSITIES_PATH = None
REGISTRY_PATH = None
TABLE5_BASENAME = None
OUT_BASENAME = None
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TEST_YEARS = list(range(2017, 2025))
PANEL_START = "2011-04-01"
PANEL_END = "2025-12-31"
K_DEFAULT, K_MAX, Q_INIT_SCALE, LAMBDA_Q = 8, 15, 0.5, 0.3


def _resolve_paths(variant: str, t_yr: int = 5, block_scheme: str = "us_inherited"):
    suffix = f"_{variant}" if variant else ""
    panel = f"experiment_b1{suffix}"
    out_suffix = "" if t_yr == 5 else f"_T{t_yr}"
    bs_suffix = "" if block_scheme == "us_inherited" else f"_{block_scheme}"
    return (
        PROJECT_ROOT / "data" / "intensities" / f"{panel}_intensities.parquet",
        PROJECT_ROOT / "data" / "registries" / f"{panel}_registry.json",
        f"table5_uk_eu{suffix}{out_suffix}{bs_suffix}",
        f"lowo_uk_eu{suffix}{out_suffix}{bs_suffix}",
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


def ewm_demean(obs, hl=12):
    T = obs.shape[0]
    w = np.exp(-np.arange(T)[::-1] * np.log(2) / hl)
    return (obs * w[:, None]).sum(0, keepdims=True) / w.sum()


def estimate_pooled_ar1(otr):
    bar_y = np.nan_to_num(otr.mean(axis=0), nan=0.5)
    tilde = otr - bar_y
    num = np.sum(tilde[1:] * tilde[:-1])
    den = np.sum(tilde[:-1] ** 2)
    return float(num / den) if den > 1e-12 else 0.0, bar_y


def sph_r(dm, U):
    N = U.shape[0]
    res = dm - (dm @ U) @ U.T
    return np.eye(N) * max(np.mean(res ** 2), 1e-8)


def _clip_sr(F, max_sr=0.99):
    ev = np.linalg.eigvals(F)
    mx = float(np.max(np.abs(ev)))
    return F * (max_sr / mx) if mx > max_sr else F


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
    otr = ad[(ad.index >= ts) & (ad.index <= pd.Timestamp(f"{ty - 1}-12-31"))].values.astype(np.float64)
    if otr.shape[0] < 4:
        return None
    return ad, otr, tq, N, v


def dmd_full(dm, k_svd=K_MAX):
    N = dm.shape[1]
    if dm.shape[0] < 3:
        return None
    try:
        return ExactDMDDecomposer().decompose_snapshots(dm.T, k=min(k_svd, N))
    except Exception:
        return None


def run_window_mixture(panel, ty, selected_blocks, all_blocks, T_yr=5):
    prep = _prepare_window(panel, ty, T_yr)
    if prep is None:
        return None
    ad, otr, tq, N, v = prep
    v_list = list(v)

    local_indices = {}
    for bname in selected_blocks:
        bactors = all_blocks.get(bname, [])
        bidx = [v_list.index(a) for a in bactors if a in v_list]
        if len(bidx) >= 5:
            local_indices[bname] = bidx

    rho, bar_y = estimate_pooled_ar1(otr)
    residuals = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))

    om_r = ewm_demean(residuals, 12)
    dm_r = residuals - om_r
    mf_r = dmd_full(dm_r, k_svd=K_MAX)
    if mf_r is None:
        return None
    ka = min(K_DEFAULT, mf_r.basis.shape[0] - 2, mf_r.K)
    F_r = _clip_sr(mf_r.metadata["Atilde"][:ka, :ka].real.copy())
    U_r = mf_r.metadata["U"][:, :ka]
    R_r = sph_r(dm_r, U_r)
    a_r, P_r = np.zeros(ka), np.eye(ka)
    Q_r = np.eye(ka) * Q_INIT_SCALE

    local_models = {}
    for bname, bidx in local_indices.items():
        N_b = len(bidx)
        block_resids = residuals[:, bidx]
        om_b = ewm_demean(block_resids, 12)
        dm_b = block_resids - om_b
        K_b = min(4, max(2, N_b // 5))
        C = dm_b.T @ dm_b / dm_b.shape[0]
        ev, ec = np.linalg.eigh(C)
        U_pca = ec[:, np.argsort(ev)[::-1]][:, :K_b]
        fac = dm_b @ U_pca
        X, Y = fac[:-1], fac[1:]
        try:
            A_pca = (np.linalg.solve(X.T @ X + 1.0 * np.eye(K_b), X.T @ Y)).T
        except np.linalg.LinAlgError:
            A_pca = np.eye(K_b) * 0.5
        local_models[bname] = {"U_pca": U_pca, "A_pca": A_pca, "om_b": om_b,
                                "K_b": K_b, "bidx": bidx, "N_b": N_b}

    preds_global, preds_mix, actuals = [], [], []
    prev = np.nan_to_num(otr[-1], nan=0.5)

    for qd in tq:
        qv = ad.loc[[qd]].values.astype(np.float64) if qd in ad.index else np.zeros((0, N))
        if qv.shape[0] == 0:
            continue
        obs = qv[0]
        y_pool = bar_y + rho * (prev - bar_y)

        ap_r = F_r @ a_r
        Pp_r = F_r @ P_r @ F_r.T + Q_r
        resid_pred = U_r @ ap_r + om_r.ravel()
        if not np.all(np.isfinite(resid_pred)):
            resid_pred = np.zeros(N)
        y_global = y_pool + resid_pred
        preds_global.append(y_global.copy())

        y_mix = y_global.copy()
        prev_resid = (prev - (bar_y + rho * (np.nan_to_num(
            otr[-2] if otr.shape[0] >= 2 else otr[-1], nan=0.5) - bar_y))
            if otr.shape[0] >= 2 else np.zeros(N))
        for bname, lm in local_models.items():
            bidx = lm["bidx"]
            prev_resid_b = prev_resid[bidx] - lm["om_b"].ravel()
            f_pca = lm["U_pca"].T @ prev_resid_b
            local_pred = lm["U_pca"] @ (lm["A_pca"] @ f_pca) + lm["om_b"].ravel()
            if np.all(np.isfinite(local_pred)):
                y_mix[bidx] = y_pool[bidx] + local_pred
            else:
                y_mix[bidx] = y_pool[bidx]
        preds_mix.append(y_mix)
        actuals.append(obs)

        actual_resid = obs - y_pool
        odm_r = actual_resid - om_r.ravel()
        S_r = U_r @ Pp_r @ U_r.T + R_r
        try:
            Kg_r = Pp_r @ U_r.T @ np.linalg.solve(S_r, np.eye(N))
        except Exception:
            Kg_r = np.zeros((ka, N))
        a_r = ap_r + Kg_r @ (odm_r - U_r @ ap_r)
        P_r = (np.eye(ka) - Kg_r @ U_r) @ Pp_r
        inn_r = a_r - ap_r
        Q_r = (1 - LAMBDA_Q) * Q_r + LAMBDA_Q * np.outer(inn_r, inn_r)
        Q_r = (Q_r + Q_r.T) / 2 + np.eye(ka) * 1e-6
        prev = obs

        otr = np.vstack([otr, qv])
        rho, bar_y = estimate_pooled_ar1(otr)
        residuals_new = otr[1:] - (bar_y + rho * (otr[:-1] - bar_y))
        om_r = ewm_demean(residuals_new, 12)
        dm_r = residuals_new - om_r
        mf_r2 = dmd_full(dm_r, k_svd=K_MAX)
        if mf_r2 is not None:
            k2 = min(K_DEFAULT, mf_r2.basis.shape[0] - 2, mf_r2.K)
            F_r = _clip_sr(mf_r2.metadata["Atilde"][:k2, :k2].real.copy())
            U_r2 = mf_r2.metadata["U"][:, :k2]
            a_r = U_r2.T @ (actual_resid - om_r.ravel())
            P_r = np.eye(k2); Q_r = np.eye(k2) * Q_INIT_SCALE
            R_r = sph_r(dm_r, U_r2); U_r = U_r2; ka = k2
        residuals = residuals_new

        for bname, lm in local_models.items():
            bidx = lm["bidx"]
            block_resids = residuals[:, bidx]
            om_b = ewm_demean(block_resids, 12)
            dm_b = block_resids - om_b
            K_b = lm["K_b"]
            C2 = dm_b.T @ dm_b / dm_b.shape[0]
            ev2, ec2 = np.linalg.eigh(C2)
            lm["U_pca"] = ec2[:, np.argsort(ev2)[::-1]][:, :K_b]
            fac = dm_b @ lm["U_pca"]
            X, Y = fac[:-1], fac[1:]
            try:
                lm["A_pca"] = (np.linalg.solve(X.T @ X + 1.0 * np.eye(K_b), X.T @ Y)).T
            except Exception:
                pass
            lm["om_b"] = om_b

    if not actuals:
        return None
    act_a = np.array(actuals)
    glob_a = np.array(preds_global)
    mix_a = np.array(preds_mix)
    if not np.all(np.isfinite(glob_a)) or not np.all(np.isfinite(mix_a)):
        return None
    return {
        "global": float(oos_r_squared(glob_a.ravel(), act_a.ravel())),
        "mixture": float(oos_r_squared(mix_a.ravel(), act_a.ravel())),
    }


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
    print(f"  LEAVE-ONE-WINDOW-OUT BLOCK SELECTION -- {INTENSITIES_PATH.stem}")
    print(f"  Variant: '{args.variant or 'full'}' | T_yr: {args.t_yr} | Block scheme: {args.block_scheme}")
    print(f"  Reads: {TABLE5_BASENAME}.parquet")
    print("=" * 70)

    panel, meta = load_data()
    actors = list(panel.columns)

    if args.block_scheme == "us_inherited":
        all_candidate_blocks = {
            "SEC_diversified": [a for a in actors if meta.get(a, {}).get("sector") == "diversified"],
            "LAYER_macro_inst": [a for a in actors if meta.get(a, {}).get("layer", -1) in (0, 1)],
            "MERGED_tech_health": [a for a in actors if meta.get(a, {}).get("sector") in ("technology", "healthcare")],
        }
    elif args.block_scheme == "sector_split":
        all_candidate_blocks = {
            "LAYER_macro_inst": [a for a in actors if meta.get(a, {}).get("layer", -1) in (0, 1)],
        }
        for sec in ["energy", "financials", "healthcare", "technology", "consumer", "industrials"]:
            all_candidate_blocks[f"SEC_{sec}"] = [
                a for a in actors
                if meta.get(a, {}).get("sector") == sec
                and meta.get(a, {}).get("layer", -1) == 2
            ]
    # Drop empty / too-small blocks (SEC_diversified is empty in B1)
    all_candidate_blocks = {k: v for k, v in all_candidate_blocks.items() if len(v) >= 5}
    print(f"\nCandidate blocks (after dropping empty/<5): "
          f"{ {k: len(v) for k, v in all_candidate_blocks.items()} }")

    df_t5 = pd.read_parquet(METRICS_DIR / f"{TABLE5_BASENAME}.parquet")

    print("\nBlock selection per window (using other 7):")
    selected_per_window = {}

    for held_out in TEST_YEARS:
        other_years = [y for y in TEST_YEARS if y != held_out]
        selected = []
        for bname in all_candidate_blocks:
            block_col = f"block_{bname}"
            if block_col not in df_t5.columns:
                continue
            deltas = []
            for yr in other_years:
                g1 = df_t5[(df_t5["architecture"] == "G1") & (df_t5["year"] == yr)][block_col].values
                m2 = df_t5[(df_t5["architecture"] == "M2") & (df_t5["year"] == yr)][block_col].values
                if len(g1) > 0 and len(m2) > 0:
                    deltas.append(float(m2[0]) - float(g1[0]))
            if deltas and np.mean(deltas) > 0:
                selected.append(bname)
        selected_per_window[held_out] = selected
        print(f"  {held_out}: {selected}")

    print(f"\n  {'Year':>6s}  {'#Blk':>5s}  {'G1':>8s}  {'LOWO':>8s}  {'Fixed':>8s}  {'LOWO-G1':>9s}")
    print(f"  {'-' * 53}")

    lowo_deltas = []
    fixed_deltas = []
    per_year_rows = []
    for ty in TEST_YEARS:
        selected = selected_per_window[ty]
        res = run_window_mixture(panel, ty, selected, all_candidate_blocks, T_yr=args.t_yr)
        fixed_m2 = df_t5[(df_t5["architecture"] == "M2") & (df_t5["year"] == ty)]["full_r2"].values[0]

        if res:
            d_lowo = res["mixture"] - res["global"]
            d_fixed = fixed_m2 - res["global"]
            lowo_deltas.append(d_lowo)
            fixed_deltas.append(d_fixed)
            print(f"  {ty:>6d}  {len(selected):>5d}  {res['global']:>8.4f}  {res['mixture']:>8.4f}"
                  f"  {fixed_m2:>8.4f}  {d_lowo:>+9.4f}")
            per_year_rows.append({"year": ty, "n_blocks_selected": len(selected),
                                  "selected_blocks": ",".join(selected),
                                  "global_r2": res["global"], "lowo_mixture_r2": res["mixture"],
                                  "fixed_mixture_r2": float(fixed_m2),
                                  "lowo_delta": d_lowo, "fixed_delta": d_fixed})

    lowo_arr = np.array(lowo_deltas)
    fixed_arr = np.array(fixed_deltas)
    lowo_mean = float(lowo_arr.mean())
    fixed_mean = float(fixed_arr.mean())
    se = lowo_arr.std(ddof=1) / np.sqrt(len(lowo_arr)) if len(lowo_arr) > 1 else 0
    t_stat = lowo_mean / se if se > 1e-15 else 0
    p_val = 2 * t_dist.sf(abs(t_stat), len(lowo_arr) - 1) if len(lowo_arr) > 1 else float("nan")
    wins = int((lowo_arr > 0).sum())

    rng = np.random.default_rng(42)
    bs = np.array([rng.choice(lowo_arr, len(lowo_arr), replace=True).mean() for _ in range(10000)])
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

    contamination = fixed_mean - lowo_mean
    pct = contamination / fixed_mean * 100 if abs(fixed_mean) > 1e-6 else 0

    print(f"\n  SUMMARY:")
    print(f"    Fixed M2 mean Delta: {fixed_mean:+.4f}")
    print(f"    LOWO mean Delta:     {lowo_mean:+.4f}")
    print(f"    LOWO t-stat:         {t_stat:.2f}")
    print(f"    LOWO p-value:        {p_val:.4f}")
    print(f"    LOWO CI:             [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"    LOWO wins:           {wins}/{len(lowo_arr)}")
    print(f"    Contamination:       {contamination:+.4f} ({pct:.1f}% of fixed gain)")

    print()
    if ci[0] > 0 and lowo_mean > 0.005:
        verdict = "PASSES"
        msg = ("The gain survives strict causal block selection -- the choice of which blocks "
               "to model locally does not depend on test-window data.")
    elif lowo_mean > 0:
        verdict = "PARTIAL"
        msg = ("LOWO mean is positive but CI may include zero or contamination is material. "
               "Acceptable for a small panel (n=8 windows).")
    else:
        verdict = "FAILS"
        msg = "Block selection was materially contaminated by test-window data."
    print(f"  VERDICT: LOWO {verdict}")
    print(f"  --> {msg}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_year_rows).to_parquet(METRICS_DIR / f"{OUT_BASENAME}.parquet", index=False)
    summary = {
        "panel": "experiment_b1",
        "test_years": TEST_YEARS,
        "n_windows": len(lowo_deltas),
        "fixed_mean_delta": fixed_mean,
        "lowo_mean_delta": lowo_mean,
        "lowo_t": float(t_stat),
        "lowo_p": float(p_val),
        "lowo_ci_lo": ci[0], "lowo_ci_hi": ci[1],
        "lowo_wins": wins,
        "contamination_abs": float(contamination),
        "contamination_pct": float(pct),
        "verdict": verdict,
        "selected_per_window": {str(k): v for k, v in selected_per_window.items()},
    }
    with open(METRICS_DIR / f"{OUT_BASENAME}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {OUT_BASENAME}.{{parquet,json}}")


if __name__ == "__main__":
    main()
