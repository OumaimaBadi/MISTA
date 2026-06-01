import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path


def generate_reports(output_dir, iteration, frobenius, delta_W, delta_loss, mars_masks, mars_probs=None):
    """
    Génère tous les rapports (CSV, TXT, PDF) avec support pour mars_probs.
    
    Args:
        mars_probs: dict[rank] -> array(n,) - Soft MARS probabilities σ(φ/T)
    """
    base_dir = Path(output_dir) / f"iter_{iteration:06d}_complete"
    base_dir.mkdir(parents=True, exist_ok=True)

    def fix_len(x, n, tag=""):
        x = np.asarray(x).reshape(-1)
        if x.size == n:
            return x
        print(f"[WARN] {tag}: fixing length {x.size} -> {n}")
        if x.size > n:
            return x[:n]
        return np.pad(x, (0, n - x.size), constant_values=np.nan)

    def safe_corr(a, b):
        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 2:
            return np.nan

        aa = a[ok]; bb = b[ok]
        if np.std(aa) == 0 or np.std(bb) == 0:
            return np.nan

        return float(np.corrcoef(aa, bb)[0, 1])

    for rank_name in ['r1', 'r2', 'r3', 'r4']:
        rank_dir = base_dir / rank_name
        rank_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------
        # 1) Load base signals
        # ------------------------
        frob_L = np.asarray(frobenius[rank_name]['frob_L']).reshape(-1)
        frob_R = np.asarray(frobenius[rank_name]['frob_R']).reshape(-1)
        n = int(len(frob_L))

        # ------------------------
        # 2) Load ΔW
        # ------------------------
        dW_mean = delta_W[rank_name]["deltaW_mean"]
        dW_max  = delta_W[rank_name]["deltaW_max"]

        # ------------------------
        # 3) Load ΔLoss summary
        # ------------------------
        summ = delta_loss[rank_name]["summary"]
        dQ_mean = summ["deltaLquality_mean"]
        dQ_max  = summ["deltaLquality_max"]
        dI_mean = summ["deltaLimg_mean"]
        dI_max  = summ["deltaLimg_max"]
        dR_mean = summ["deltaLreg_mean"]
        dR_max  = summ["deltaLreg_max"]

        # ------------------------
        # 4) Load MARS mask (hard, 0 or 1)
        # ------------------------
        if mars_masks is not None and (rank_name in mars_masks):
            mask = mars_masks[rank_name].detach().cpu().numpy().reshape(-1)
        else:
            mask = np.full(n, np.nan, dtype=np.float32)

        # ------------------------
        # 5) ✅ NOUVEAU : Load MARS probabilities (soft, σ(φ/T))
        # ------------------------
        if mars_probs is not None and (rank_name in mars_probs):
            mars_prob = mars_probs[rank_name].reshape(-1)
        else:
            mars_prob = np.full(n, np.nan, dtype=np.float32)

        # ------------------------
        # 6) Align lengths
        # ------------------------
        frob_L    = fix_len(frob_L,    n, f"{rank_name}/frob_L")
        frob_R    = fix_len(frob_R,    n, f"{rank_name}/frob_R")
        dW_mean   = fix_len(dW_mean,   n, f"{rank_name}/deltaW_mean")
        dW_max    = fix_len(dW_max,    n, f"{rank_name}/deltaW_max")
        dQ_mean   = fix_len(dQ_mean,   n, f"{rank_name}/deltaLquality_mean")
        dQ_max    = fix_len(dQ_max,    n, f"{rank_name}/deltaLquality_max")
        dI_mean   = fix_len(dI_mean,   n, f"{rank_name}/deltaLimg_mean")
        dI_max    = fix_len(dI_max,    n, f"{rank_name}/deltaLimg_max")
        dR_mean   = fix_len(dR_mean,   n, f"{rank_name}/deltaLreg_mean")
        dR_max    = fix_len(dR_max,    n, f"{rank_name}/deltaLreg_max")
        mask      = fix_len(mask,      n, f"{rank_name}/mars_mask")
        mars_prob = fix_len(mars_prob, n, f"{rank_name}/mars_prob")  # ✅ NOUVEAU

        # Derived metrics
        frob_prod = frob_L * frob_R
        frob_sum  = frob_L + frob_R
        imbalance = np.abs(np.log((frob_R + 1e-8) / (frob_L + 1e-8)))

        has_mask = np.isfinite(mask).any()
        has_prob = np.isfinite(mars_prob).any()

        # ------------------------
        # 7) ✅ CSV avec mars_prob
        # ------------------------
        df = pd.DataFrame({
            "component_id": np.arange(n),

            "frob_L": frob_L,
            "frob_R": frob_R,
            "frob_prod": frob_prod,
            "frob_sum": frob_sum,
            "imbalance": imbalance,

            "deltaW_mean": dW_mean,
            "deltaW_max": dW_max,

            "deltaL_quality_mean": dQ_mean,
            "deltaL_quality_max": dQ_max,
            "deltaL_img_mean": dI_mean,
            "deltaL_img_max": dI_max,
            "deltaL_reg_mean": dR_mean,
            "deltaL_reg_max": dR_max,

            "mars_mask": mask,              # Hard mask (0 or 1)
            "mars_prob": mars_prob,         # ✅ NOUVEAU: Soft probability σ(φ/T)
            
            "mars_status": (
                ["active" if m > 0.5 else "pruned" for m in mask]
                if has_mask else ["warmup"] * n
            )
        })

        df.to_csv(rank_dir / "metrics_complete.csv", index=False, float_format="%.6f")

        # ------------------------
        # 8) TXT summary
        # ------------------------
        corr_frob_prob = safe_corr(frob_sum, mars_prob) if has_prob else np.nan
        corr_dW_prob   = safe_corr(dW_mean, mars_prob)  if has_prob else np.nan

        # keep these (not related to MARS)
        corr_dW_dQ   = safe_corr(dW_mean, dQ_mean)
        corr_frob_dQ = safe_corr(frob_sum, dQ_mean)

        # hard correlations disabled
        corr_frob_mars = np.nan
        corr_dW_mars   = np.nan

        with open(rank_dir / "summary.txt", "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"{rank_name.upper()} - IMPORTANCE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("Correlations:\n")
            f.write(f"  Frob_sum ↔ MARS (soft σ):    {corr_frob_prob:.4f}\n")
            f.write(f"  ΔW_mean  ↔ MARS (soft σ):    {corr_dW_prob:.4f}\n")
            f.write(f"  ΔW_mean  ↔ ΔL_quality_mean:  {corr_dW_dQ:.4f}\n")
            f.write(f"  Frob_sum ↔ ΔL_quality_mean:  {corr_frob_dQ:.4f}\n\n")

            if has_mask:
                n_active = int((mask > 0.5).sum())
                n_pruned = int((mask <= 0.5).sum())
                f.write("MARS Status (hard mask):\n")
                f.write(f"  Active: {n_active}/{n} ({100*n_active/n:.1f}%)\n")
                f.write(f"  Pruned: {n_pruned}/{n} ({100*n_pruned/n:.1f}%)\n\n")
            
            if has_prob:
                prob_mean = np.nanmean(mars_prob)
                prob_std = np.nanstd(mars_prob)
                f.write("MARS Probabilities (soft σ(φ/T)):\n")
                f.write(f"  Mean: {prob_mean:.4f}\n")
                f.write(f"  Std:  {prob_std:.4f}\n")
                f.write(f"  Min:  {np.nanmin(mars_prob):.4f}\n")
                f.write(f"  Max:  {np.nanmax(mars_prob):.4f}\n\n")

        # ------------------------
        # 9) PDF (4 graphs) ← NOUVEAU graph avec mars_prob
        # ------------------------
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Graph 1: Frob_L vs Frob_R
        axes[0, 0].scatter(frob_L, frob_R, alpha=0.6, s=80)
        axes[0, 0].set_xlabel("Frob_L", fontsize=12)
        axes[0, 0].set_ylabel("Frob_R", fontsize=12)
        axes[0, 0].set_title(f"{rank_name} - Frob_L vs Frob_R", fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)

        # Graph 2: Histogram Imbalance
        axes[0, 1].hist(imbalance, bins=30, alpha=0.7)
        axes[0, 1].set_xlabel("Imbalance |log(R/L)|", fontsize=12)
        axes[0, 1].set_ylabel("Count", fontsize=12)
        axes[0, 1].set_title(f"{rank_name} - Imbalance Distribution", fontsize=14)
        axes[0, 1].axvline(np.log(10), color="red", linestyle="--", label="Starved threshold")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Graph 3: ΔL_quality_mean vs MARS (hard)
        axes[1, 0].scatter(dQ_mean, mask, alpha=0.6, s=80, c='blue')
        axes[1, 0].set_xlabel("ΔL_quality_mean", fontsize=12)
        axes[1, 0].set_ylabel("MARS mask (hard)", fontsize=12)
        axes[1, 0].set_title(f"{rank_name} - ΔL vs MARS hard (constant)", fontsize=14)
        axes[1, 0].axhline(0.5, color="red", linestyle="--", alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)

        # Graph 4: ✅ NOUVEAU - ΔL_quality_mean vs MARS prob (soft σ)
        if has_prob:
            axes[1, 1].scatter(dQ_mean, mars_prob, alpha=0.6, s=80, c='orange')
            axes[1, 1].set_xlabel("ΔL_quality_mean", fontsize=12)
            axes[1, 1].set_ylabel("MARS prob (σ(φ/T))", fontsize=12)
            axes[1, 1].set_title(
                f"{rank_name} - ΔL vs MARS soft σ (r={corr_dW_prob:.3f})",
                fontsize=14
            )
            axes[1, 1].axhline(0.5, color="red", linestyle="--", alpha=0.5)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "MARS prob not available", 
                           ha='center', va='center', fontsize=14)
            axes[1, 1].set_title(f"{rank_name} - MARS soft σ (N/A)", fontsize=14)

        plt.tight_layout()
        plt.savefig(rank_dir / "visualization.pdf", dpi=150, bbox_inches="tight")
        plt.close()

        # ------------------------
        # 10) JSON summary
        # ------------------------
        summary_json = {
            "rank": rank_name,
            "iteration": int(iteration),
            "n_components": int(n),
            "has_mars_mask": bool(has_mask),
            "has_mars_prob": bool(has_prob),
            "corr": {
                "frob_sum_vs_mars_soft": corr_frob_prob,
                "dW_mean_vs_mars_soft": corr_dW_prob,
                "dW_mean_vs_dQ_mean": corr_dW_dQ,
                "frob_sum_vs_dQ_mean": corr_frob_dQ,
            }
        }
        with open(rank_dir / "summary.json", "w") as f:
            json.dump(summary_json, f, indent=2)

    print(f"✅ Reports generated in: {base_dir}")