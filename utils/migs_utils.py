# migs_utils.py
import os
import math
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_param_names_and_splits(M: int) -> Tuple[List[str], List[int]]:
    """
    Build canonical parameter names and split sizes for a W tensor of width M.
    Layout: [xyz(3), scaling(3), rotation(4), dc(1), rest(?), opacity(1)]
    """
    base = {"xyz": 3, "scaling": 3, "rotation": 4, "dc": 1, "opacity": 1}
    # 'rest' gets whatever is left
    used = sum(base.values())
    rest = max(M - used, 0)
    split_sizes = [base["xyz"], base["scaling"], base["rotation"], base["dc"], rest, base["opacity"]]

    names = []
    names += ["x", "y", "z"]
    names += ["scale_x", "scale_y", "scale_z"]
    names += ["rot_w", "rot_x", "rot_y", "rot_z"]
    names += ["dc"]
    names += [f"feat_{i}" for i in range(rest)]
    names += ["opacity"]
    return names, split_sizes


def compare_reconstruction_per_block(
    W_original: torch.Tensor,
    W_reconstructed: torch.Tensor,
    split_sizes: Optional[List[int]] = None,
    names: Optional[List[str]] = None,
    extra_metrics: bool = False,
    prefix: str = "[Diag]"
) -> None:
    """
    Print reconstruction quality per semantic block.
    Set extra_metrics=True to also print RMSE/MAE/cos/pearson (TT-style).
    """
    assert W_original.shape == W_reconstructed.shape, "Shape mismatch."
    M = W_original.shape[1]
    if names is None or split_sizes is None:
        names, split_sizes = get_param_names_and_splits(M)

    total = sum(split_sizes)
    if total != M:
        # auto-fix 'rest' to fill up to M
        rest_idx = names.index("feat_0") - 1 if "feat_0" in names else names.index("dc") + 1
        other = total - split_sizes[4]
        split_sizes = split_sizes.copy()
        split_sizes[4] = M - other
        assert sum(split_sizes) == M

    orig_parts = torch.split(W_original, split_sizes, dim=1)
    recon_parts = torch.split(W_reconstructed, split_sizes, dim=1)

    def _stats(t: torch.Tensor):
        return t.min().item(), t.max().item(), t.mean().item()

    def _pearson(a: torch.Tensor, b: torch.Tensor):
        a = a.flatten(); b = b.flatten()
        ac, bc = a - a.mean(), b - b.mean()
        denom = ac.std(unbiased=False) * bc.std(unbiased=False)
        if denom.item() == 0.0:
            return float("nan")
        return float((ac * bc).mean().item() / denom.item())

    print(f"\n{prefix} Reconstruction per block:\n")
    idx = 0
    for name, o, r, width in zip(
        ["xyz","scaling","rotation","dc","rest","opacity"],
        orig_parts, recon_parts, split_sizes
    ):
        d = o - r
        mse = torch.mean(d * d).item()
        l2  = torch.norm(d).item()
        line = f"  {name:8s} | width:{width:2d}  MSE:{mse:.6e}  L2:{l2:.6e}"
        if extra_metrics:
            rmse = torch.sqrt(torch.mean(d * d)).item()
            mae  = torch.mean(torch.abs(d)).item()
            cos  = torch.nn.functional.cosine_similarity(o.reshape(1, -1), r.reshape(1, -1)).item()
            rho  = _pearson(o, r)
            line += f"  RMSE:{rmse:.6e}  MAE:{mae:.6e}  cos:{cos:.6f}  ρ:{rho:.6f}"
        print(line)

        if extra_metrics:
            o_min, o_max, o_mean = _stats(o)
            r_min, r_max, r_mean = _stats(r)
            print(f"             ORIG[min:{o_min:.6f}, max:{o_max:.6f}, mean:{o_mean:.6f}]  "
                  f"RECON[min:{r_min:.6f}, max:{r_max:.6f}, mean:{r_mean:.6f}]")
    print()


def plot_correlation_across_parameters(
    W_original: torch.Tensor,
    W_reconstructed: torch.Tensor,
    outdir: str = "diagnostic",
    drop_zero_variance: bool = False,
    plot_variances: bool = False,
    fname_prefix: str = "param_corr"
) -> None:
    """
    Save correlation heatmaps for original, reconstructed, and their difference.
    - drop_zero_variance=True replicates the CP version (removes constant columns).
    - plot_variances=True replicates the TT version (also saves a variance bar plot).
    """
    _ensure_dir(outdir)
    M = W_original.shape[1]
    names, _ = get_param_names_and_splits(M)

    def _corr_and_names(t: torch.Tensor):
        df = pd.DataFrame(t.detach().cpu().numpy(), columns=names)
        if plot_variances:
            var = df.var()
            plt.figure(figsize=(12, 4))
            var.plot(kind="bar")
            plt.title("Parameter variances")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{fname_prefix}_variances.png"))
            plt.close()
        if drop_zero_variance:
            valid = df.var() > 1e-8
            df = df.loc[:, valid]
            cols = df.columns.tolist()
        else:
            cols = names
        corr = df.corr().reindex(index=cols, columns=cols).fillna(0)
        return corr.values, cols

    C_o, cols_o = _corr_and_names(W_original)
    C_r, cols_r = _corr_and_names(W_reconstructed)

    # Align to the intersection of columns if they differ
    cols = sorted(set(cols_o).intersection(cols_r))
    df_o = pd.DataFrame(C_o, index=cols_o, columns=cols_o).loc[cols, cols]
    df_r = pd.DataFrame(C_r, index=cols_r, columns=cols_r).loc[cols, cols]

    diff = df_o.values - df_r.values
    frob = np.linalg.norm(diff, ord="fro")

    for mat, title, fname in [
        (df_o.values, "Original",      f"{fname_prefix}_orig.png"),
        (df_r.values, "Reconstructed", f"{fname_prefix}_recon.png"),
        (diff,        f"Difference (Frobenius={frob:.4f})", f"{fname_prefix}_diff.png"),
    ]:
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            mat, xticklabels=cols, yticklabels=cols,
            cmap=("bwr" if "Difference" in title else "coolwarm"),
            center=0, annot=False
        )
        plt.title(f"Correlation Matrix – {title}")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname))
        plt.close()

    print(f"[Corr] Frobenius norm (orig - recon): {frob:.6f}")


def plot_pca_groupwise_xyz_auto(
    W_original: torch.Tensor,
    W_reconstructed: torch.Tensor,
    num_groups: int = 10,
    outpath: str = "diagnostic/pca_groupwise_xyz_auto.png",
    random_state: int = 0
) -> None:
    """
    PCA of XYZ (first 3 dims) for original vs reconstructed, colored by KMeans on original.
    """
    _ensure_dir(os.path.dirname(outpath) or ".")
    xyz_orig = W_original[:, :3].detach().cpu().numpy()
    xyz_recon = W_reconstructed[:, :3].detach().cpu().numpy()

    labels = KMeans(n_clusters=num_groups, random_state=random_state).fit_predict(xyz_orig)
    pca_o = PCA(n_components=2).fit_transform(xyz_orig)
    pca_r = PCA(n_components=2).fit_transform(xyz_recon)

    cmap = plt.cm.get_cmap("tab20", num_groups)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(num_groups):
        m = labels == i
        plt.scatter(pca_o[m, 0], pca_o[m, 1], alpha=0.6, s=10, color=cmap(i))
    plt.title("Original XYZ PCA")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for i in range(num_groups):
        m = labels == i
        plt.scatter(pca_r[m, 0], pca_r[m, 1], alpha=0.6, s=10, color=cmap(i))
    plt.title("Reconstructed XYZ PCA")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
