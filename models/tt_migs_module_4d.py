import math
import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor
from hilbertcurve.hilbertcurve import HilbertCurve
from utils.migs_utils import (
    compare_reconstruction_per_block,
    plot_correlation_across_parameters,
    plot_pca_groupwise_xyz_auto,
)
import hashlib

tl.set_backend('pytorch')

try:
    isqrt = math.isqrt
except AttributeError:
    def isqrt(n: int) -> int:
        if n <= 0:
            return 0
        x = int(n ** 0.5)  # first guess
        while (x + 1) * (x + 1) <= n:
            x += 1
        while x * x > n:
            x -= 1
        return x
# ------------------------------------------------------------
# Rank suggestion (generic over K spatial modes)
# ------------------------------------------------------------
def suggest_tt_ranks_weighted(
    shape, target_ratio, cp_R=100, cap_last=43,
    rM_hint=None,
    # controls
    min_r1=16,              # lower bound for r1
    r1_cap=None,            # upper bound for r1 (e.g., 36/48/56)
    identity_bias=0.90,     # <1.0 → shrinks r1 (identity side)
    mid_boost=1.15,         # >1.0 → grows r2..rK (intermediate cores)
    weight_power=1.15,      # favors larger spatial modes
):
    """
    Given TT shape = [I, n1, ..., nK, M], choose ranks = [1, r1..rK, rM, 1]
    to fit a parameter budget, guided by soft weights over spatial modes.
    """
    I = int(shape[0]); M = int(shape[-1])
    mids = [int(x) for x in shape[1:-1]]
    K = len(mids)
    if K < 1:
        raise ValueError("shape must contain at least one G-mode between I and M")

    # --- CP budget (R fixed for comparability) ---
    G = 1
    for x in mids: G *= x
    cp_params = cp_R * (I + G + M)
    budget = float(target_ratio) * float(cp_params)

    # --- choose rM by ratio if no hint ---
    if rM_hint is None:
        if target_ratio <= 0.051:   rM = min(cap_last, 32)
        elif target_ratio <= 0.11:  rM = min(cap_last, 38)
        else:                       rM = min(cap_last, 43)
    else:
        rM = min(cap_last, int(rM_hint))

    # --- trivial K==1 case ---
    if K == 1:
        n1 = mids[0]
        def P_of_r1(r1): return I*r1 + r1*n1*rM + rM*M
        lo, hi = 1, max(min_r1, 256)
        while P_of_r1(hi) < budget: hi *= 2
        while lo < hi:
            md = (lo + hi + 1)//2
            if P_of_r1(md) <= budget: lo = md
            else: hi = md - 1
        r1 = max(min_r1, lo)
        if r1_cap is not None: r1 = min(r1, int(r1_cap))
        return [1, r1, rM, 1], int(P_of_r1(r1)), int(budget), rM, {"only_last": True}

    # --- weights over spatial modes n1..nK ---
    w = [float(n)**float(weight_power) for n in mids]
    if K > 0:
        w[0] *= float(identity_bias)         # calm r1 (near identity)
        for j in range(1, K):                # boost mid links
            w[j] *= float(mid_boost)
    s = sum(w); w = [x/s for x in w] if s > 0 else [1.0/K]*K

    # --- continuous budget with fixed rM ---
    def P_of_alpha(alpha: float) -> float:
        r = [max(1e-9, alpha*wj) for wj in w]   # r1..rK (continuous)
        r1_eff = max(r[0], float(min_r1))
        total = I * r1_eff
        for j in range(0, K-1):                # all spatial links except last
            left  = r1_eff if j == 0 else r[j]
            right = r[j+1]
            total += max(left,1.0) * mids[j+1] * max(right,1.0)
        total += max(r[-1],1.0) * mids[-1] * rM   # last spatial link (to rM)
        total += rM * M                            # rM → M
        return total

    lo, hi = 1e-6, 1e6
    for _ in range(50):
        md = 0.5*(lo+hi)
        if P_of_alpha(md) > budget: hi = md
        else: lo = md
    alpha = lo

    # --- discretize + clamp r1 ---
    r_float = [alpha*wj for wj in w]
    r_int = [max(1, int(round(x))) for x in r_float]
    r_int[0] = max(r_int[0], int(min_r1))
    if r1_cap is not None:
        r_int[0] = min(r_int[0], int(r1_cap))

    ranks = [1] + r_int + [rM, 1]  # [r0=1, r1..rK, rM, r_{K+2}=1]

    # --- exact param count + greedy +1 if budget remains ---
    def tt_params(shape, ranks) -> int:
        I = int(shape[0]); M = int(shape[-1]); mids = [int(x) for x in shape[1:-1]]
        r = ranks
        total = I * r[1]
        for j in range(len(mids)-1):                 # all spatial links except last
            total += r[j+1] * mids[j+1] * r[j+2]
        total += r[-3] * mids[-1] * r[-2]           # last spatial link (to rM)
        total += r[-2] * M * r[-1]                  # rM -> M -> 1
        return int(total)

    P = tt_params(shape, ranks)

    def bump_once(ranks, P):
        K = len(shape[1:-1])
        best = None
        for j in range(1, 1+K):         # r1..rK
            cand = ranks[:]; cand[j] += 1
            val = tt_params(shape, cand)
            if val <= budget:
                gain = val - P
                if best is None or gain > best[0]:
                    best = (gain, cand, val)
        return best

    while True:
        res = bump_once(ranks, P)
        if res is None: break
        _, ranks, P = res

    return ranks, int(P), int(budget), rM, {
        "alpha": alpha, "weights": w, "min_r1": min_r1, "r1_cap": r1_cap,
        "identity_bias": identity_bias, "mid_boost": mid_boost
    }


# ============================================================
#                   TT Ultra MIGS (2D G, 4 cores)
# ============================================================
class TTUltraMIGSModule4D(nn.Module):
    """
    Tensor-Train MIGS (2D tiling, 4 TT cores):
    - Reshape (G, M) → (I, n1, n2, M) et TT-factorize.
    - Le dernier cœur (index 3) est le cœur 'feature' sur M, splitté en
      slices sémantiques (xyz, scaling, rotation, dc, rest, opacity).
    - Les rangs cibles sont proposés par l'allocateur (budget). Après TT-SVD
      (avec r1=1 si I=1), on :
        * réplique+scale r1 (préserve le tenseur),
        * zero-pad r2 et rM (préserve le tenseur).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        tt_cfg = cfg.migs if not isinstance(cfg, dict) else cfg["migs"]
        self._base_seed = int(getattr(cfg, "seed", 123))
        # Freeze TT params jusqu'à cette itération
        self.tt_delay = tt_cfg.get("delay", 1000)
        if self.tt_delay is None:
            self.tt_delay = cfg.model.gaussian.get("delay", 0)

        self.tt_rank = tt_cfg.get("rank")          # défini après init_from_tensor
        self.tt_shape = tt_cfg.get("tt_shape")
        self.verbose = bool(tt_cfg.get("verbose", False))

        self.optimizer = None
        self.scheduler = None
        self._opt_cfg = None
        self._needs_opt_rebuild = False
        self._tt_unfrozen = False

        # MARS (optionnel)
        self.use_mars = bool(tt_cfg.get("use_mars", False))
        _m = tt_cfg.get("mars", {})
        self._mars_in_optimizer = False
        self.mars_logit_lr = tt_cfg.get("mars_logit_lr", 1e-3)
        self.mars_start_iter = int(_m.get("start_iter", 1000))
        self.mars_active = False
        self.mars_cfg = {
            "init_keep_prob": _m.get("init_keep_prob", 0.90),
            "tau_start":  _m.get("tau", {}).get("start", 2.0),
            "tau_end":    _m.get("tau", {}).get("end",   0.5),
            "tau_iters":  _m.get("tau", {}).get("iters", 30000),
            "l0_start":   _m.get("l0_lambda", {}).get("start", 1e-5),
            "l0_end":     _m.get("l0_lambda", {}).get("end",   2e-4),
            "l0_iters":   _m.get("l0_lambda", {}).get("iters", 30000),
            "gamma":      _m.get("hard_concrete", {}).get("gamma", -0.1),
            "zeta":       _m.get("hard_concrete", {}).get("zeta",   1.1),
            "mask_core0": _m.get("mask_core0", False),
            "mask_core5_broadcast_over_M": True,    # compat
            "r_last_must_be_one": True,             # dernier rang = 1 pour fast path
            "effective_rank_eps": 1e-3,
            # 2D → masques pour c1, c2 et core3 (3 masques)
            "l0_weights": _m.get("l0_weights", [1.0, 1.0, 0.7]),
            "report_every": _m.get("report_every", 500),
        }
        self._tau = self.mars_cfg["tau_start"]
        self._l0_lambda = self.mars_cfg["l0_start"]
        self.mars_logits = None  # défini après init TT

        # TT cores (remplis dans init_from_tensor): c0..c3
        self.tt_tensor_gpu = nn.ParameterList()

        # Cœur features (core3) splitté (Param pour LRs par slice)
        self.core3_xyz      = nn.Parameter(torch.zeros(1, 3, 1))
        self.core3_scaling  = nn.Parameter(torch.zeros(1, 3, 1))
        self.core3_rotation = nn.Parameter(torch.zeros(1, 4, 1))
        self.core3_dc       = nn.Parameter(torch.zeros(1, 1, 1))
        self.core3_rest     = nn.Parameter(torch.zeros(1, 31, 1))
        self.core3_opacity  = nn.Parameter(torch.zeros(1, 1, 1))
        self.save_dir = getattr(self.cfg, "hilbert_vis_4d", "./exports")
        os.makedirs(self.save_dir, exist_ok=True)

    def _stream(self, tag: str, device) -> torch.Generator:
        h = int.from_bytes(hashlib.md5(tag.encode("utf8")).digest()[:8], 'little')
        g = torch.Generator(device=device)
        g.manual_seed(self._base_seed ^ h)
        return g

    def _randn_like(self, ref, tag):  # Gaussian noise ~ N(0,1)
        g = self._stream(tag, ref.device)
        return torch.randn(ref.shape, device=ref.device, dtype=ref.dtype, generator=g)

    def _rand_like(self, ref, tag):   # Uniform noise ~ U(0,1)
        g = self._stream(tag, ref.device)
        return torch.rand(ref.shape, device=ref.device, dtype=ref.dtype, generator=g)



    # --------------------- INITIALISATION ---------------------
    def init_from_tensor(self, gaussian_model):
        """
        1) Stack Gaussian params → W_GM (G, M).
        2) Ordre Hilbert 3D sur XYZ → permutation des lignes (localité TT).
        3) Choisir une factorisation 2D (n1,n2) de G en minimisant un coût d'adjacence.
        4) Reshape en (I=1, n1, n2, M) et TT-SVD avec r1=1.
        5) Split du cœur features (index 3) en slices sémantiques.
        6) Expand r1 (réplication+scale) puis zero-pad r2 et rM (préserve le tenseur).
        """
        G = gaussian_model._xyz.shape[0]

        # (1) Assemble (G, M)
        xyz           = gaussian_model._xyz
        scaling       = gaussian_model._scaling
        rotation      = gaussian_model._rotation
        features_dc   = gaussian_model._features_dc.squeeze(-1)
        features_rest = gaussian_model._features_rest.squeeze(-1)
        opacity       = gaussian_model._opacity

        def print_param_stats(name, tensor):
            t = tensor.detach().cpu()
            print(f"{name:12s} shape={tuple(t.shape)} | "
                f"min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}")

        print_param_stats("xyz", xyz)
        print_param_stats("scaling", scaling)
        print_param_stats("rotation", rotation)
        print_param_stats("features_dc", features_dc)
        print_param_stats("features_rest", features_rest)
        print_param_stats("opacity", opacity)
        
        all_params = [xyz, scaling, rotation, features_dc, features_rest, opacity]
        W_GM = torch.cat([p if p.ndim == 2 else p.view(p.shape[0], -1) for p in all_params], dim=1)
        M = W_GM.shape[1]


        # (2) Permutation Hilbert
        perm = self._build_spatial_order_from_xyz(W_GM[:, 0:3], method="hilbert", bits=15)
        inv_perm = torch.empty_like(perm); inv_perm[perm] = torch.arange(G, device=perm.device)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)
        W_perm = W_GM[self.perm.to(W_GM.device)]
        # --- Export après permutation (et defs numpy pour plus tard) ---
        with torch.no_grad():
            idx = self.perm.to(gaussian_model._xyz.device)
            xyz_perm        = gaussian_model._xyz[idx].detach().cpu().numpy()
            scaling_perm    = gaussian_model._scaling[idx].detach().cpu().numpy()
            rotation_perm   = gaussian_model._rotation[idx].detach().cpu().numpy()
            features_dc_p   = gaussian_model._features_dc[idx].detach().cpu().numpy()
            features_rest_p = gaussian_model._features_rest[idx].detach().cpu().numpy()
            opacity_perm    = gaussian_model._opacity[idx].detach().cpu().numpy()

            use_sh = bool(getattr(gaussian_model, "use_sh", False))
            sh_deg = int(getattr(gaussian_model, "max_sh_degree", 0)) if use_sh else 0

            np.savez_compressed(
                os.path.join(self.save_dir, "snapshot_after_perm_full.npz"),
                xyz=xyz_perm, scaling=scaling_perm, rotation=rotation_perm,
                features_dc=features_dc_p, features_rest=features_rest_p, opacity=opacity_perm,
                perm=self.perm.detach().cpu().numpy(),
                use_sh=np.array([use_sh], dtype=np.bool_), sh_deg=np.array([sh_deg], dtype=np.int64)
            )
            print("[EXPORT] snapshot_after_perm_full.npz")

        # (3) Choix d'une forme 2D (n1,n2)
        candidates = self._candidate_shapes_2d(G)
        best_shape, scored = self._pick_best_shape_2d(self.perm, W_GM[:, :3], candidates)
        if self.verbose:
            print(f"[TT] 2D adjacency top samples: {scored[:6]}")
            print(f"[TT] picked 2D shape: {best_shape}")
        assert best_shape[0] * best_shape[1] == G
        n1, n2 = best_shape

        # (4) TT-SVD
        self.tt_shape = (1, n1, n2, int(M))
        W_tt = W_perm.unsqueeze(0).reshape(self.tt_shape)
        # ===== EXPORT COMPLET APRES RESHAPE (mêmes points + grille i,j) =====
        G_chk = xyz_perm.shape[0]
        assert n1 * n2 == G_chk, "n1*n2 doit égaler G"
        Ig, Jg = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
        ij = np.stack([Ig.ravel(), Jg.ravel()], axis=1).astype(np.int64)

        np.savez_compressed(
            os.path.join(self.save_dir, "snapshot_after_reshape_full.npz"),
            xyz=xyz_perm, scaling=scaling_perm, rotation=rotation_perm,
            features_dc=features_dc_p, features_rest=features_rest_p, opacity=opacity_perm,
            ij=ij, shapeG=np.array([n1, n2], dtype=np.int64),
            perm=self.perm.detach().cpu().numpy(),
            use_sh=np.array([use_sh], dtype=np.bool_), sh_deg=np.array([sh_deg], dtype=np.int64)
        )
        print("[EXPORT] snapshot_after_reshape_full.npz")
        
        migs_cfg = self.cfg.migs if not isinstance(self.cfg, dict) else self.cfg["migs"]
        final_I       = int(migs_cfg.get("final_identities"))
        target_ratio  = float(migs_cfg.get("target_ratio"))
        rM_hint       = migs_cfg.get("rM_hint")
        cap_last      = int(migs_cfg.get("cap_last"))
        min_r1        = int(migs_cfg.get("min_r1"))
        r1_cap        = migs_cfg.get("r1_cap")
        identity_bias = float(migs_cfg.get("identity_bias"))
        mid_boost     = float(migs_cfg.get("mid_boost"))
        weight_power  = float(migs_cfg.get("weight_power"))

        shape_final = [final_I, n1, n2, int(M)]
        ranks_target, params_cur, params_budget, rM_used, _ = suggest_tt_ranks_weighted(
            shape_final,
            target_ratio=target_ratio,
            cp_R=100,
            cap_last=cap_last,
            rM_hint=rM_hint,
            min_r1=min_r1,
            r1_cap=r1_cap,
            identity_bias=identity_bias,
            mid_boost=mid_boost,
            weight_power=weight_power,
        )
        if self.verbose:
            print(f"[TT] target ranks = {ranks_target} | params {params_cur}/{params_budget} | rM={rM_used}")

        # r1=1 pour SVD init si I=1
        ranks_init = list(ranks_target)
        ranks_init[1] = 1
        self.tt_rank = ranks_target

        tt_tensor = tensor_train(W_tt, rank=ranks_init, verbose=self.verbose)
        self.tt_tensor_gpu = nn.ParameterList([nn.Parameter(c.to(W_tt.device)) for c in tt_tensor.factors])

        # (5) Split du cœur features (core index 3)
        core3 = self.tt_tensor_gpu[3]  # (rM, M, 1)
        rM_act, M_act, r_last = core3.shape
        assert M_act == M and r_last == 1, f"Feature core mismatch: {(rM_act, M_act, r_last)} vs (*,{M},1)"

        self.core3_xyz      = nn.Parameter(core3[:, 0:3,   :].detach().clone())
        self.core3_scaling  = nn.Parameter(core3[:, 3:6,   :].detach().clone())
        self.core3_rotation = nn.Parameter(core3[:, 6:10,  :].detach().clone())
        self.core3_dc       = nn.Parameter(core3[:, 10:11, :].detach().clone())
        self.core3_rest     = nn.Parameter(core3[:, 11:42, :].detach().clone())
        self.core3_opacity  = nn.Parameter(core3[:, 42:43, :].detach().clone())

        # (6a) Expand r1 par réplication + scale
        def _repeat_to(x: torch.Tensor, dim: int, target: int) -> torch.Tensor:
            cur = x.shape[dim]
            if cur == target:
                return x
            times = math.ceil(target / cur)
            reps = [1] * x.dim(); reps[dim] = times
            x_rep = x.repeat(*reps)
            sl = [slice(None)] * x.dim(); sl[dim] = slice(0, target)
            return x_rep[tuple(sl)]

        with torch.no_grad():
            c0 = self.tt_tensor_gpu[0]  # (1, 1, r1_cur)
            c1 = self.tt_tensor_gpu[1]  # (r1_cur, n1, r2_cur)
            r1_cur = c0.shape[2]
            r1_tgt = max(r1_cur, self.tt_rank[1])
            if r1_tgt != r1_cur:
                scale = r1_cur / float(r1_tgt)
                c0_new = _repeat_to(c0.detach(), dim=2, target=r1_tgt) * scale
                c1_new = _repeat_to(c1.detach(), dim=0, target=r1_tgt)
                self.tt_tensor_gpu[0] = nn.Parameter(c0_new)
                self.tt_tensor_gpu[1] = nn.Parameter(c1_new)
                self._needs_opt_rebuild = True

        # (6b) Expand r2 et rM par zero-padding (préserve le tenseur)
        self._expand_ranks_to_targets_preserve_2d(self.tt_rank)

        if self.verbose:
            print("TT core shapes after init/expand:")
            for i, core in enumerate(self.tt_tensor_gpu[:3]):
                print(f"  core[{i}] -> {tuple(core.shape)}")
            rM_now, Mx, rlast = self.recombine_core3().shape
            print(f"  core[3] (feature, recombined) -> {(rM_now, Mx, rlast)}")

        # MARS (optionnel)
        if self.use_mars:
            self._init_mars_masks()
            if self.mars_logits is not None:
                for p in self.mars_logits:
                    p.requires_grad = False

        if self.optimizer is not None and self._needs_opt_rebuild:
            self._rebuild_optimizer_like_before()
            self._needs_opt_rebuild = False

        # Diagnostics
        W_rec = self.get_W_for_identity(0, original_order=True).to(W_GM.device)
        if self.verbose:
            print(f"[TT] recon shape: {tuple(W_rec.shape)}")
        compare_reconstruction_per_block(
            W_GM, W_rec, split_sizes=[3, 3, 4, 1, 31, 1],
            names=['xyz', 'scaling', 'rotation', 'dc', 'rest', 'opacity']
        )
        plot_correlation_across_parameters(W_GM, W_rec)
        plot_pca_groupwise_xyz_auto(W_GM, W_rec, num_groups=10)

    # ---------- ZERO-PAD EXPANSION HELPERS (préserve le tenseur) ----------
    @torch.no_grad()
    def _zero_pad_pair_preserve(self, left: torch.Tensor, right: torch.Tensor,
                                add: int, dim_left: int, dim_right: int):
        if add <= 0:
            return left, right
        dev = left.device
        dl_shape = list(left.shape);  dl_shape[dim_left]  = add
        dr_shape = list(right.shape); dr_shape[dim_right] = add
        pad_left  = torch.zeros(dl_shape,  device=dev, dtype=left.dtype)
        pad_right = torch.zeros(dr_shape, device=dev, dtype=right.dtype)
        new_left  = torch.cat([left,  pad_left],  dim=dim_left)
        new_right = torch.cat([right, pad_right], dim=dim_right)
        return new_left, new_right

    @torch.no_grad()
    def _expand_ranks_to_targets_preserve_2d(self, ranks_target):
        """
        Faire correspondre r2 et rM aux cibles par zero-padding :
          c1(..,r2) — c2(r2,..)
          c2(..,rM) — core3(rM,..)
        """
        c1 = self.tt_tensor_gpu[1]  # (r1,n1,r2)
        c2 = self.tt_tensor_gpu[2]  # (r2,n2,rM)

        # r2
        r2_cur = c1.shape[2]
        r2_tgt = int(ranks_target[2])
        if r2_tgt > r2_cur:
            add = r2_tgt - r2_cur
            new_c1, new_c2 = self._zero_pad_pair_preserve(c1, c2, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[1] = nn.Parameter(new_c1)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)

        # rM
        c2 = self.tt_tensor_gpu[2]
        core3_full = self.recombine_core3()  # (rM, M, 1)
        rM_cur = c2.shape[2]
        rM_tgt = int(ranks_target[-2])
        if rM_tgt > rM_cur:
            add = rM_tgt - rM_cur
            new_c2, new_core3 = self._zero_pad_pair_preserve(c2, core3_full, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)

            # re-split sémantique (lignes ajoutées = 0)
            rM_new, M, rlast = new_core3.shape
            assert rlast == 1
            self.core3_xyz      = nn.Parameter(new_core3[:, 0:3,   :])
            self.core3_scaling  = nn.Parameter(new_core3[:, 3:6,   :])
            self.core3_rotation = nn.Parameter(new_core3[:, 6:10,  :])
            self.core3_dc       = nn.Parameter(new_core3[:, 10:11, :])
            self.core3_rest     = nn.Parameter(new_core3[:, 11:42, :])
            self.core3_opacity  = nn.Parameter(new_core3[:, 42:43, :])

        self._needs_opt_rebuild = True

    # --------------------- ORDER/SHAPE HELPERS ---------------------
    @staticmethod
    def _hilbert_code(x, y, z, bits=15):
        hc = HilbertCurve(bits, 3)
        def q(u):
            u = np.clip(u, 0, 1)
            return (u * (2**bits - 1) + 0.5).astype(np.int64)
        xi, yi, zi = q(x), q(y), q(z)
        if hasattr(hc, "distances_from_points"):
            pts = [[int(a), int(b), int(c)] for a, b, c in zip(xi, yi, zi)]
            return np.asarray(hc.distances_from_points(pts), dtype=np.int64)
        for name in ("distance_from_coordinates", "point_to_distance", "coordinates_to_distance"):
            if hasattr(hc, name):
                fn = getattr(hc, name)
                out = np.empty_like(xi, dtype=np.int64)
                for i, (a, b, c) in enumerate(zip(xi, yi, zi)):
                    out[i] = int(fn([int(a), int(b), int(c)]))
                return out
        raise RuntimeError("No compatible HilbertCurve distance method found.")

    @staticmethod
    def _build_spatial_order_from_xyz(xyz_t: torch.Tensor, method="hilbert", bits=15) -> torch.Tensor:
        xyz = xyz_t.detach().cpu().numpy()
        mn, mx = xyz.min(0), xyz.max(0)
        xyz01 = (xyz - mn) / (mx - mn + 1e-8)
        codes = TTUltraMIGSModule4D._hilbert_code(xyz01[:, 0], xyz01[:, 1], xyz01[:, 2], bits=bits)
        return torch.from_numpy(np.argsort(codes)).long()

    # ----- 2D balanced factorization -----
    @staticmethod
    def _balanced_shape2_for(G: int) -> tuple:
        # a <= b, a*b = G, a le plus proche de sqrt(G)
        a = 1
        r = isqrt(G)
        for d in range(1, r + 1):
            if G % d == 0:
                a = d
        b = G // a
        return (a, b)

    @staticmethod
    def _candidate_shapes_2d(G: int) -> list:
        a, b = TTUltraMIGSModule4D._balanced_shape2_for(G)
        # tester (a,b) et (b,a)
        if a == b:
            return [(a, b)]
        else:
            return [(a, b), (b, a)]

    @staticmethod
    def _adjacency_cost_2d(order: np.ndarray, xyz: np.ndarray, shape: tuple) -> float:
        n1, n2 = shape
        idx = order.reshape(n1, n2)
        pts = xyz[idx]  # (n1,n2,3)
        def axis_cost(a):
            front = np.take(pts, range(0, shape[a]-1), axis=a)
            back  = np.take(pts, range(1, shape[a]  ), axis=a)
            d = back - front
            return np.sum((d*d).sum(axis=-1))
        return axis_cost(0) + axis_cost(1)

    @staticmethod
    def _pick_best_shape_2d(order_t: torch.Tensor, xyz_t: torch.Tensor, candidates: list) -> tuple:
        order = order_t.cpu().numpy()
        xyz = xyz_t.detach().cpu().numpy()
        scored = [(sh, TTUltraMIGSModule4D._adjacency_cost_2d(order, xyz, sh)) for sh in candidates]
        scored.sort(key=lambda t: t[1])
        return scored[0][0], scored

    # --------------------- MARS (optionnel) ---------------------
    def _init_mars_masks(self):
        if self.mars_cfg["r_last_must_be_one"]:
            rM, M, rlast = self.recombine_core3().shape
            assert rlast == 1, "MARS requires final TT rank = 1."

        init_p = self.mars_cfg["init_keep_prob"]
        init_logit = math.log(init_p / (1.0 - init_p))
        logs = []

        # core0 optionnel
        if self.mars_cfg["mask_core0"]:
            r0, I, r1 = self.tt_tensor_gpu[0].shape
            logs.append(nn.Parameter(torch.full((r0, 1, r1), init_logit, device=self.tt_tensor_gpu[0].device)))

        # masques pour c1 et c2
        for k in [1, 2]:
            rk, nk, rkp1 = self.tt_tensor_gpu[k].shape
            logs.append(nn.Parameter(torch.full((rk, 1, rkp1), init_logit, device=self.tt_tensor_gpu[k].device)))

        # masque pour cœur features (core3)
        rM, M, rlast = self.recombine_core3().shape
        logs.append(nn.Parameter(torch.full((rM, 1, rlast), init_logit, device=self.core3_xyz.device)))
        self.mars_logits = nn.ParameterList(logs)

        # normaliser l0_weights à 3 masques (hors core0)
        needed = len(logs) - (1 if self.mars_cfg["mask_core0"] else 0)
        w = list(self.mars_cfg["l0_weights"])
        if len(w) < needed:
            w += [w[-1]] * (needed - len(w))
        elif len(w) > needed:
            w = w[:needed]
        self.mars_cfg["l0_weights"] = w

    def _hard_concrete(self, logits, training=True):
        tau = self._tau; gamma = self.mars_cfg["gamma"]; zeta = self.mars_cfg["zeta"]
        if training:
            u = torch.rand_like(logits)
            s = torch.sigmoid((torch.log(u) - torch.log(1-u) + logits) / tau)
        else:
            s = torch.sigmoid(logits / tau)
        s = s * (zeta - gamma) + gamma
        return torch.clamp(s, 0.0, 1.0)

    def _expected_gate(self, logits):
        tau = self._tau; gamma = self.mars_cfg["gamma"]; zeta = self.mars_cfg["zeta"]
        s = torch.sigmoid(logits / tau)
        s = s * (zeta - gamma) + gamma
        return torch.clamp(s, 0.0, 1.0)

    def loss_mars(self):
        if not (self.use_mars and self.mars_active and (self.mars_logits is not None)):
            return torch.tensor(0.0, device=self.tt_tensor_gpu[0].device)
        w = self.mars_cfg["l0_weights"]
        offset = 1 if self.mars_cfg["mask_core0"] else 0
        terms = []
        for i in range(len(w)):  # c1, c2, core3
            E = self._expected_gate(self.mars_logits[offset + i]).mean()
            terms.append(w[i]*E)
        return self._l0_lambda * sum(terms)

    def update_mars_schedule(self, iteration: int):
        if not self.use_mars: return
        def anneal(st, en, iters):
            if not iters: return en
            t = min(max(iteration, 0), iters) / float(iters)
            return (1.0 - t)*st + t*en
        self._tau = anneal(self.mars_cfg["tau_start"], self.mars_cfg["tau_end"], self.mars_cfg["tau_iters"])
        self._l0_lambda = anneal(self.mars_cfg["l0_start"], self.mars_cfg["l0_end"], self.mars_cfg["l0_iters"])

    def effective_ranks(self):
        if not (self.use_mars and (self.mars_logits is not None)): return None
        eps = self.mars_cfg["effective_rank_eps"]
        offset = 1 if self.mars_cfg["mask_core0"] else 0
        names = ["r1", "r2", "rM"]
        eff = {}
        for i, name in enumerate(names):
            E = self._expected_gate(self.mars_logits[offset + i])
            active_rk   = (E.sum(dim=2) > eps).sum().item()
            active_rkp1 = (E.sum(dim=0) > eps).sum().item()
            eff[name] = (int(active_rk), int(active_rkp1))
        return eff

    # --------------------- RECONSTRUCTION ---------------------
    def recombine_core3(self):
        """Recombine le cœur features en un seul TT core (rM, 43, 1)."""
        return torch.cat(
            [self.core3_xyz, self.core3_scaling, self.core3_rotation,
             self.core3_dc, self.core3_rest, self.core3_opacity],
            dim=1
        )

    def get_core0(self, idx):
        assert 0 <= idx < self.tt_tensor_gpu[0].shape[1], f"Invalid identity index {idx}"
        return self.tt_tensor_gpu[0][:, idx:idx+1, :]  # (1, 1, r1)

    def expand_first_core(self, n_identities):
        if not len(self.tt_tensor_gpu):
            raise RuntimeError("TT cores must be initialized before expansion.")
        first = self.tt_tensor_gpu[0]   # (1, N, r1)
        r0, n_cur, r1 = first.shape
        if n_cur >= n_identities:
            if self.verbose: print(f"[TT] identity axis already ≥ {n_identities}")
            return
        base = first[:, 0:1, :].detach()
        rep   = base.repeat(1, n_identities, 1)
        noise = self._randn_like(rep, tag="core0_expand_noise") * 1e-3
        new   = rep + noise
        self.tt_tensor_gpu[0] = nn.Parameter(new)
        self._needs_opt_rebuild = True
        if self.optimizer is not None:
            self._rebuild_optimizer_like_before()
        if self.verbose:
            print(f"[TT] expanded first core → {self.tt_tensor_gpu[0].shape}")

    def reconstruct(self):
        full = list(self.tt_tensor_gpu[:3]) + [self.recombine_core3()]
        return tt_to_tensor(full)

    def get_tt_tensor(self, idx=None):
        core0 = self.get_core0(idx) if idx is not None else self.tt_tensor_gpu[0]
        cores = [core0, self.tt_tensor_gpu[1], self.tt_tensor_gpu[2], self.recombine_core3()]
        if self.use_mars and (self.mars_logits is not None) and self.mars_active:
            offset = 1 if self.mars_cfg["mask_core0"] else 0
            cores[1] = self._apply_mask_core (cores[1], self.mars_logits[offset + 0])  # c1
            cores[2] = self._apply_mask_core (cores[2], self.mars_logits[offset + 1])  # c2
            cores[3] = self._apply_mask_core3(cores[3], self.mars_logits[offset + 2])  # core3
        return cores

    def _contract_tt_identity_gemm(self, idx: int) -> torch.Tensor:
        """
        Fast GEMM path pour (I=1) en 2D:
          c0:(1,1,r1), c1:(r1,n1,r2), c2:(r2,n2,rM), core3:(rM,M,1)
        """
        c0 = self.get_core0(idx)
        c1 = self.tt_tensor_gpu[1]
        c2 = self.tt_tensor_gpu[2]
        c3 = self.recombine_core3()

        r1, n1, r2 = c1.shape
        _,  n2, rM = c2.shape
        rM_c3, M, rlast = c3.shape

        assert c0.shape == (1, 1, r1), "core0 incompatible with c1"
        assert rM_c3 == rM, "feature core rM mismatch"
        assert rlast == 1, "last TT rank must be 1 for this path"

        X = c0.reshape(1, r1) @ c1.reshape(r1, n1 * r2)
        X = X.reshape(n1, r2) @ c2.reshape(r2, n2 * rM)
        X = X.reshape(n1 * n2, rM)
        C3 = c3.squeeze(-1).reshape(rM, M)
        return X @ C3


        # X = c0.view(1, r1) @ c1.view(r1, n1 * r2)
        # X = X.view(n1, r2) @ c2.view(r2, n2 * rM)
        # X = X.view(n1 * n2, rM)
        # C3 = c3.squeeze(-1).contiguous().view(rM, M)
        # return X @ C3  # (G, M)

    def get_W_for_identity(self, idx: int, original_order: bool = True) -> torch.Tensor:
        try:
            W_perm = self._contract_tt_identity_gemm(idx)
        except AssertionError:
            T = tt_to_tensor(self.get_tt_tensor(idx))
            M = T.shape[-1]
            W_perm = T.squeeze(0).contiguous().view(-1, M)
        # pour revenir à l'ordre original, décommente si souhaité:
        # if original_order and hasattr(self, "inv_perm"):
        #     return W_perm[self.inv_perm.to(W_perm.device)]
        return W_perm

    # --------------------- TRAINING CONTROL ---------------------
    def optimize_parameters(self):
        return list(self.tt_tensor_gpu[:3]) + [
            self.core3_xyz, self.core3_scaling, self.core3_rotation,
            self.core3_dc, self.core3_rest, self.core3_opacity,
        ]

    def freeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = False
        if self.mars_logits is not None:
            for p in self.mars_logits:
                p.requires_grad = False

    def unfreeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = True
        if self.mars_logits is not None:
            for p in self.mars_logits:
                p.requires_grad = False

    def set_optimizer(self, opt_cfg):
        self._opt_cfg = dict(opt_cfg) if opt_cfg is not None else {}
        want_groups = max(0, len(self.tt_tensor_gpu) - 1)  # exclure le cœur features
        tt_lrs = list(self._opt_cfg.get("tt_lrs", [1.6e-4] * want_groups))
        tt_final_lrs = list(self._opt_cfg.get("tt_final_lrs", [1.6e-6] * want_groups))
        while len(tt_lrs) < want_groups: tt_lrs.append(tt_lrs[-1])
        while len(tt_final_lrs) < want_groups: tt_final_lrs.append(tt_final_lrs[-1])
        tt_decay_iters = int(self._opt_cfg.get("tt_decay_iters", 50000))

        param_groups = []
        for i in range(want_groups):
            param_groups.append({
                "params": [self.tt_tensor_gpu[i]],
                "lr": float(tt_lrs[i]),
                "initial_lr": float(tt_lrs[i]),
                "final_lr": float(tt_final_lrs[i]),
            })

        # slices features (mêmes LRs par défaut que ta 6-cœurs)
        param_groups.append({
            "params": [self.core3_xyz],
            "lr": 1.6e-4, "initial_lr": 1.6e-4, "final_lr": 1.6e-6
        })
        param_groups += [
            {"params": [self.core3_scaling],  "lr": 5e-3},
            {"params": [self.core3_rotation], "lr": 1e-3},
            {"params": [self.core3_dc],       "lr": 2.5e-3},
            {"params": [self.core3_rest],     "lr": 2.5e-3},
            {"params": [self.core3_opacity],  "lr": 5e-2},
        ]

        self._mars_in_optimizer = False
        self.optimizer = torch.optim.Adam(param_groups)

        if any("final_lr" in g for g in param_groups):
            gamma = (1.6e-6 / 1.6e-4) ** (1.0 / max(1, tt_decay_iters))
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        else:
            self.scheduler = None

        self._needs_opt_rebuild = False

    def _rebuild_optimizer_like_before(self):
        self.set_optimizer(self._opt_cfg)

    def attach_mars_to_optimizer(self, mars_lr=None):
        if not (self.use_mars and (self.mars_logits is not None) and (self.optimizer is not None)):
            return
        if self._mars_in_optimizer:
            return
        lr = self.mars_logit_lr if mars_lr is None else mars_lr
        self.optimizer.add_param_group({"params": list(self.mars_logits), "lr": lr})
        self._mars_in_optimizer = True

    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, iteration=None):
        if hasattr(self, "_ft_opt") and (self._ft_opt is not None):
            if iteration is not None:
                self.update_mars_schedule(iteration)
            self.ft_step()
            return

        if iteration is not None:
            self.update_mars_schedule(iteration)

        if self.optimizer is None:
            return

        if (iteration is not None) and (iteration < self.tt_delay):
            if iteration == self.tt_delay - 1 and self.verbose:
                print(f"[TT] TT cores frozen until iter {iteration}")
            self.freeze_tt_parameters()
            self.optimizer.zero_grad()
            return

        if (iteration is not None) and (not self._tt_unfrozen) and (iteration >= self.tt_delay):
            self.unfreeze_tt_parameters()
            self._tt_unfrozen = True
            if self.verbose:
                print(f"[TT] TT cores unfrozen at iter {iteration}")

        if self.use_mars and (self.mars_logits is not None):
            want_active = (iteration is not None and iteration >= self.mars_start_iter)
            if want_active and not self.mars_active:
                self.mars_active = True
                for p in self.mars_logits:
                    p.requires_grad = True
                if not self._mars_in_optimizer:
                    self.attach_mars_to_optimizer()
                if self.verbose:
                    print(f"[MARS] enabled at iter {iteration}")
            elif (not want_active) and self.mars_active:
                self.mars_active = False
                for p in self.mars_logits:
                    p.requires_grad = False
                if self.verbose:
                    print(f"[MARS] disabled at iter {iteration}")

        if self.mars_active:
            self.update_mars_schedule(iteration)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_learning_rate()

    def _apply_mask_core(self, core, logit):
        if not (self.use_mars and self.mars_active):
            return core
        m = self._hard_concrete(logit, self.training)
        return core * m

    def _apply_mask_core3(self, core3, logit):
        if not (self.use_mars and self.mars_active):
            return core3
        m = self._hard_concrete(logit, self.training)
        return core3 * m

    # --------------------- IDENTITY FINETUNE ---------------------
    @torch.no_grad()
    def add_identity(self, noise_scale: float = 0.05, rebuild_optimizer: bool = True) -> int:
        if not len(self.tt_tensor_gpu):
            raise RuntimeError("TT cores must be initialized before adding an identity.")
        core0 = self.tt_tensor_gpu[0]   # (1, N, r1)
        r0, n_id, r1 = core0.shape
        device = core0.device
        if n_id > 0:
            U = core0.detach()
            mu  = U.mean(dim=1, keepdim=True)
            sig = U.std(dim=1, unbiased=False, keepdim=True).clamp_(min=1e-8)
            eps     = self._randn_like(mu, tag="add_identity_noise").expand_as(mu)  # même shape
            new_row = mu + noise_scale * sig * eps
            norms = U.view(n_id, r1).norm(dim=1)
            target_norm = norms.median()
            cur_norm = new_row.view(-1).norm()
            new_row = new_row / (cur_norm + 1e-8) * float(target_norm)
        else:
            new_row = self._randn_like(core0[:, :1, :], tag="add_identity_boot") * 0.02
        self.tt_tensor_gpu[0] = nn.Parameter(torch.cat([core0, new_row], dim=1))
        self._needs_opt_rebuild = True
        if rebuild_optimizer and (self.optimizer is not None):
            self._rebuild_optimizer_like_before()
        if self.verbose:
            print(f"[TT] added identity → core0 shape {self.tt_tensor_gpu[0].shape}")
        return self.tt_tensor_gpu[0].shape[1] - 1

    def enable_identity_finetune(self, idx: int, color_mlp: nn.Module,
                                 lr_id: float = 3e-3, lr_tex: float = 1e-3,
                                 include_color_in_ft_opt: bool = False):
        for p in self.parameters():
            p.requires_grad = False
        core0 = self.tt_tensor_gpu[0]
        core0.requires_grad = True
        mask = torch.zeros_like(core0); mask[:, idx:idx+1, :] = 1.0
        if hasattr(self, "_core0_mask_hook") and (self._core0_mask_hook is not None):
            self._core0_mask_hook.remove()
        self._core0_mask_hook = core0.register_hook(lambda g: g * mask)
        for p in color_mlp.parameters():
            p.requires_grad = include_color_in_ft_opt
        params = [{"params": [core0], "lr": lr_id}]
        if include_color_in_ft_opt:
            params.append({"params": list(color_mlp.parameters()), "lr": lr_tex})
        self._ft_opt = torch.optim.Adam(params)

    def ft_step(self):
        if hasattr(self, "_ft_opt") and (self._ft_opt is not None):
            self._ft_opt.step()
            self._ft_opt.zero_grad()

    def disable_identity_finetune(self):
        if hasattr(self, "_core0_mask_hook") and (self._core0_mask_hook is not None):
            self._core0_mask_hook.remove()
        self._core0_mask_hook = None
        self._ft_opt = None
        for p in self.parameters():
            p.requires_grad = True
