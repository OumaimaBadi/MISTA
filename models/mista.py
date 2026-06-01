import math
import os
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
import time

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class _Timer:
    def __enter__(self):
        _cuda_sync()
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *args):
        _cuda_sync()
        self.dt = time.perf_counter() - self.t0


tl.set_backend('pytorch')

def suggest_tt_ranks_weighted(
    shape, target_ratio, cp_R=100, cap_last=43,
    rM_hint=None,
    # controls
    min_r1=16,              # lower bound for r1
    r1_cap=None,            # upper bound for r1 (e.g., 36/48/56)
    identity_bias=0.90,     # <1.0 → shrinks r1 (identity side)
    mid_boost=1.15,         # >1.0 → grows r2..rK (intermediate cores)
    weight_power=1.15,      # favors larger spatial modes (50,40,25)
):
    I = int(shape[0]); M = int(shape[-1])
    mids = [int(x) for x in shape[1:-1]]
    K = len(mids)
    if K < 1:
        raise ValueError("shape must contain at least one G-mode between I and M")

    # CP budget
    G = 1
    for x in mids: G *= x
    cp_params = cp_R * (I + G + M)
    budget = float(target_ratio) * float(cp_params)

    # choose rM by ratio if no hint
    if rM_hint is None:
        if target_ratio <= 0.051:   rM = min(cap_last, 32)
        elif target_ratio <= 0.11:  rM = min(cap_last, 38)
        else:                       rM = min(cap_last, 43)
    else:
        rM = min(cap_last, int(rM_hint))

    # trivial K==1 case 
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

    # weights: identity vs intermediates
    w = [float(n)**float(weight_power) for n in mids]   # K weights
    if K > 0:
        w[0] *= float(identity_bias)                    # identity core
        for j in range(1, K):                           # intermediate cores
            w[j] *= float(mid_boost)
    s = sum(w); w = [x/s for x in w] if s > 0 else [1.0/K]*K

    # continuous budget with fixed rM 
    def P_of_alpha(alpha: float) -> float:
        r = [max(1e-9, alpha*wj) for wj in w]   # r1..rK
        r1_eff = max(r[0], float(min_r1))
        total = I * r1_eff
        for j in range(0, K-1):
            left  = r1_eff if j == 0 else r[j]
            right = r[j+1]
            total += max(left,1.0) * mids[j+1] * max(right,1.0)
        total += max(r[-1],1.0) * mids[-1] * rM
        total += rM * M
        return total

    lo, hi = 1e-6, 1e6
    for _ in range(50):
        md = 0.5*(lo+hi)
        if P_of_alpha(md) > budget: hi = md
        else: lo = md
    alpha = lo

    # discretize + clamp r1
    r_float = [alpha*wj for wj in w]
    r_int = [max(1, int(round(x))) for x in r_float]
    r_int[0] = max(r_int[0], int(min_r1))
    if r1_cap is not None:
        r_int[0] = min(r_int[0], int(r1_cap))

    ranks = [1] + r_int + [rM, 1]

    # exact param count + greedy +1 if budget remains
    def tt_params(shape, ranks) -> int:
        I = int(shape[0]); M = int(shape[-1]); mids = [int(x) for x in shape[1:-1]]
        r = ranks
        total = I * r[1]
        for j in range(len(mids)-1):
            total += r[j+1] * mids[j+1] * r[j+2]
        total += r[-3] * mids[-1] * r[-2]
        total += r[-2] * M * r[-1]
        return int(total)

    P = tt_params(shape, ranks)

    def bump_once(ranks, P):
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


class TTUltraMIGSModule5D(nn.Module):
    """
    Tensor-Train MIGS:
    factorizes a (I, n1, n2, n3, M) tensor of Gaussian parameters into TT cores,
    then learns those cores. The last mode M (43) is split into semantic slices.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        tt_cfg = cfg.migs if not isinstance(cfg, dict) else cfg["migs"]
        self._base_seed = int(getattr(cfg, "seed", 123))
        # Training delay (TT-only)
        self.tt_delay = tt_cfg.get("delay", 1000)
        if self.tt_delay is None:
            self.tt_delay = cfg.model.gaussian.get("delay", 0)

        # TT ranks and working shape (will be replaced after init_from_tensor)
        self.tt_rank = tt_cfg.get("rank")
        self.tt_shape = tt_cfg.get("tt_shape")
        self.verbose = bool(tt_cfg.get("verbose", False))

        self.optimizer = None
        self.scheduler = None
        self._opt_cfg = None
        self._needs_opt_rebuild = False
        self._tt_unfrozen = False

        # MARS (optional)
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
            "mask_core4_broadcast_over_M": _m.get("mask_core4_broadcast_over_M", True),
            "r5_must_be_one": _m.get("r5_must_be_one", True),
            "effective_rank_eps": _m.get("effective_rank_eps", 1e-3),
            "l0_weights": _m.get("l0_weights", [1.0, 1.0, 1.0, 0.7]),
            "report_every": _m.get("report_every", 500),
        }
        self._tau = self.mars_cfg["tau_start"]
        self._l0_lambda = self.mars_cfg["l0_start"]
        self.mars_logits = None  # set after TT init

        # Actual TT cores (filled in init_from_tensor)
        self.tt_tensor_gpu = nn.ParameterList()

        # Last-mode split Parameters (filled/reshaped in init_from_tensor)
        self.core4_xyz      = nn.Parameter(torch.zeros(1, 3, 1))
        self.core4_scaling  = nn.Parameter(torch.zeros(1, 3, 1))
        self.core4_rotation = nn.Parameter(torch.zeros(1, 4, 1))
        self.core4_dc       = nn.Parameter(torch.zeros(1, 1, 1))
        self.core4_rest     = nn.Parameter(torch.zeros(1, 31, 1))
        self.core4_opacity  = nn.Parameter(torch.zeros(1, 1, 1))
        self.save_dir = getattr(self.cfg, "hilbert_vis_5d", "./exports")
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




    def init_from_tensor(self, gaussian_model):
        """Build TT from the current Gaussian parameters and initialize trainable cores."""
        G = gaussian_model._xyz.shape[0]
        print("******************", G)

        # Assemble (G, M) = [xyz|scaling|rotation|dc|rest|opacity]
        xyz          = gaussian_model._xyz
        scaling      = gaussian_model._scaling
        rotation     = gaussian_model._rotation
        features_dc  = gaussian_model._features_dc.squeeze(-1)
        features_rest= gaussian_model._features_rest.squeeze(-1)
        opacity      = gaussian_model._opacity

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

        #Spatial permutation (Hilbert order) for better TT locality
        with _Timer() as t_h:
            perm = self._build_spatial_order_from_xyz(W_GM[:, :3], method="hilbert", bits=15)
        print(f"[COST] hilbert_order_sec={t_h.dt:.6f}")


        inv_perm = torch.empty_like(perm); inv_perm[perm] = torch.arange(G, device=perm.device)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)
        W_perm = W_GM[self.perm.to(W_GM.device)]  # (G, M)

            
        with torch.no_grad():
            # paramètres dans l’ordre Hilbert (perm)
            idx = self.perm.to(gaussian_model._xyz.device)
            xyz_perm       = gaussian_model._xyz[idx].detach().cpu().numpy()          # (G,3)
            scaling_perm   = gaussian_model._scaling[idx].detach().cpu().numpy()      # (G,3) (log-scale 3DGS)
            rotation_perm  = gaussian_model._rotation[idx].detach().cpu().numpy()     # (G,4) quaternion
            features_dc_p  = gaussian_model._features_dc[idx].detach().cpu().numpy()  # (G,3,1) ou (G,1,1)
            features_rest_p= gaussian_model._features_rest[idx].detach().cpu().numpy()
            opacity_perm   = gaussian_model._opacity[idx].detach().cpu().numpy()      # (G,1)

            # méta (pour le viewer)
            use_sh  = bool(gaussian_model.use_sh)
            sh_deg  = int(gaussian_model.max_sh_degree) if use_sh else 0

            np.savez_compressed(
                os.path.join(self.save_dir, "snapshot_after_perm_full.npz"),
                xyz=xyz_perm, scaling=scaling_perm, rotation=rotation_perm,
                features_dc=features_dc_p, features_rest=features_rest_p, opacity=opacity_perm,
                perm=self.perm.detach().cpu().numpy(),
                use_sh=np.array([use_sh], dtype=np.bool_), sh_deg=np.array([sh_deg], dtype=np.int64)
            )
            print("[EXPORT] snapshot_after_perm_full.npz")


        # Choose a balanced (n1,n2,n3) tiling of G
        with _Timer() as t_shape:
            candidates = self._candidate_shapes(G)
            best_shape, scored = self._pick_best_shape(self.perm, W_GM[:, :3], candidates)
        print(f"[COST] shape_search_sec={t_shape.dt:.6f} | num_candidates={len(candidates)}")

        if self.verbose:
            print(f"[TT] adjacency scores: {scored}")
            print(f"[TT] picked shape: {best_shape}")
        assert best_shape[0] * best_shape[1] * best_shape[2] == G
        n1, n2, n3 = best_shape

        # EXPORT COMPLET APRES RESHAPE (mêmes points + grille i,j,k)
        G_chk = xyz_perm.shape[0]
        assert n1 * n2 * n3 == G_chk, "n1*n2*n3 doit égaler G"
        Ig, Jg, Kg = np.meshgrid(
            np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij"
        )
        ijk = np.stack([Ig.ravel(), Jg.ravel(), Kg.ravel()], axis=1).astype(np.int64)

        np.savez_compressed(
            os.path.join(self.save_dir, "snapshot_after_reshape_full.npz"),
            xyz=xyz_perm, scaling=scaling_perm, rotation=rotation_perm,
            features_dc=features_dc_p, features_rest=features_rest_p, opacity=opacity_perm,
            ijk=ijk, shapeG=np.array([n1, n2, n3], dtype=np.int64),
            perm=self.perm.detach().cpu().numpy(),
            use_sh=np.array([use_sh], dtype=np.bool_), sh_deg=np.array([sh_deg], dtype=np.int64)
        )
        print("[EXPORT] snapshot_after_reshape_full.npz")


        # (I=1, n1, n2, n3, M)
        self.tt_shape = (1, n1, n2, n3, int(M))
        W_tt = W_perm.unsqueeze(0).reshape(self.tt_shape)

        # Auto-select TT ranks for the FINAL config (not for I=1)
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

        shape_final = [final_I, n1, n2, n3, int(M)]
        ranks_target, params_cur, params_budget, rM_used, info = suggest_tt_ranks_weighted(
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
            print(f"[TT] auto-ranks(target) = {ranks_target} | params {params_cur}/{params_budget} | rM={rM_used}")

        # For TT-SVD on (I=1, ...), enforce r1=1; then expand to target via replication.
        ranks_init = list(ranks_target)
        ranks_init[1] = 1
        self.tt_rank = ranks_target

        # TT-SVD decomposition with safe init ranks
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with _Timer() as t_tt:
            tt_tensor = tensor_train(W_tt, rank=ranks_init, verbose=self.verbose)

        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"[COST] tt_svd_sec={t_tt.dt:.6f} | peak_vram_gb={peak_vram_gb:.3f}")
        else:
            print(f"[COST] tt_svd_sec={t_tt.dt:.6f} | peak_vram_gb=N/A")

        self.tt_tensor_gpu = nn.ParameterList([nn.Parameter(c.to(W_tt.device)) for c in tt_tensor.factors])

        # Split last core by semantic parameter groups
        core4 = self.tt_tensor_gpu[4]  # (r4, M, r5)
        r4_act, M_act, r5_act = core4.shape
        assert M_act == M, f"Last mode mismatch: {M_act} vs {M}"
        self.core4_xyz      = nn.Parameter(core4[:, 0:3,   :].detach().clone())
        self.core4_scaling  = nn.Parameter(core4[:, 3:6,   :].detach().clone())
        self.core4_rotation = nn.Parameter(core4[:, 6:10,  :].detach().clone())
        self.core4_dc       = nn.Parameter(core4[:, 10:11, :].detach().clone())
        self.core4_rest     = nn.Parameter(core4[:, 11:42, :].detach().clone())
        self.core4_opacity  = nn.Parameter(core4[:, 42:43, :].detach().clone())

        # r1 expansion (replicate + rescale) 
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

        self._expand_ranks_to_targets_preserve(self.tt_rank)

        if self.verbose:
            print("TT core shapes:")
            for i, core in enumerate(self.tt_tensor_gpu[:4]):
                print(f"  core[{i}] -> {tuple(core.shape)}")
            r4, Mx, r5 = self.recombine_core4().shape
            print(f"  core[4] (recombined) -> {(r4, Mx, r5)}")

        # Initialize MARS masks once core shapes are known
        if self.use_mars:
            self._init_mars_masks()
            if self.mars_logits is not None:
                for p in self.mars_logits:
                    p.requires_grad = False  

        if self.optimizer is not None and self._needs_opt_rebuild:
            self._rebuild_optimizer_like_before()
            self._needs_opt_rebuild = False

        # Diagnostics in original order
        W_rec = self.get_W_for_identity(0, original_order=True).to(W_GM.device)
        if self.verbose:
            print(f"[TT] recon shape: {tuple(W_rec.shape)}")
        compare_reconstruction_per_block(
            W_GM, W_rec, split_sizes=[3, 3, 4, 1, 31, 1],
            names=['xyz', 'scaling', 'rotation', 'dc', 'rest', 'opacity']
        )
        plot_correlation_across_parameters(W_GM, W_rec)
        plot_pca_groupwise_xyz_auto(W_GM, W_rec, num_groups=10)


    @torch.no_grad()
    def _zero_pad_pair_preserve(self, left: torch.Tensor, right: torch.Tensor,
                                add: int, dim_left: int, dim_right: int):
        """
        Expand a shared rank by concatenating zeros on BOTH sides:
          - left is zero-padded along its 'outgoing' (last) rank axis,
          - right is zero-padded along its 'incoming' (first) rank axis.

        Because at least one factor on every *new* index is zero, the contraction is unchanged.
        """
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
    def _expand_ranks_to_targets_preserve(self, ranks_target):
        """
        Ensure that TT ranks (r2, r3, r4) reach their target values after TT-SVD
        **without changing the represented tensor** by zero-padding the shared ranks.
        - Only expands when current < target. Never shrinks.
        - Re-splits core4 after expanding r4.
        """
        # cores
        c0 = self.tt_tensor_gpu[0]  # (1, I(=1), r1)
        c1 = self.tt_tensor_gpu[1]  # (r1, n1, r2)
        c2 = self.tt_tensor_gpu[2]  # (r2, n2, r3)
        c3 = self.tt_tensor_gpu[3]  # (r3, n3, r4)

        r2_cur = c1.shape[2]
        r2_tgt = int(ranks_target[2])
        if r2_tgt > r2_cur:
            add = r2_tgt - r2_cur
            new_c1, new_c2 = self._zero_pad_pair_preserve(c1, c2, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[1] = nn.Parameter(new_c1)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)

        # refresh
        c1 = self.tt_tensor_gpu[1]
        c2 = self.tt_tensor_gpu[2]
        c3 = self.tt_tensor_gpu[3]


        r3_cur = c2.shape[2]
        r3_tgt = int(ranks_target[3])
        if r3_tgt > r3_cur:
            add = r3_tgt - r3_cur
            new_c2, new_c3 = self._zero_pad_pair_preserve(c2, c3, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)
            self.tt_tensor_gpu[3] = nn.Parameter(new_c3)

        # refresh
        c2 = self.tt_tensor_gpu[2]
        c3 = self.tt_tensor_gpu[3]


        r4_cur = c3.shape[2]
        r4_tgt = int(ranks_target[4])
        if r4_tgt > r4_cur:
            add = r4_tgt - r4_cur
            core4_full = self.recombine_core4()  # (r4, M, r5=1)

            # zero-pad both sides on the shared r4 axis
            new_c3, new_core4 = self._zero_pad_pair_preserve(c3, core4_full, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[3] = nn.Parameter(new_c3)

            # re-split semantic slices (new rows are zeros everywhere)
            r4n, M, r5 = new_core4.shape
            assert r5 == 1
            self.core4_xyz      = nn.Parameter(new_core4[:, 0:3,   :])
            self.core4_scaling  = nn.Parameter(new_core4[:, 3:6,   :])
            self.core4_rotation = nn.Parameter(new_core4[:, 6:10,  :])
            self.core4_dc       = nn.Parameter(new_core4[:, 10:11, :])
            self.core4_rest     = nn.Parameter(new_core4[:, 11:42, :])
            self.core4_opacity  = nn.Parameter(new_core4[:, 42:43, :])

        # mark that optimizer needs to be rebuilt (new params present)
        self._needs_opt_rebuild = True



    @staticmethod
    def _hilbert_code(x, y, z, bits=15):
        hc = HilbertCurve(bits, 3)
        def q(u):
            u = np.clip(u, 0, 1)
            return (u * (2**bits - 1) + 0.5).astype(np.int64)
        xi, yi, zi = q(x), q(y), q(z)

        # Robust API selection (library versions differ)
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
        codes = TTUltraMIGSModule5D._hilbert_code(xyz01[:, 0], xyz01[:, 1], xyz01[:, 2], bits=bits) \
                if method == "hilbert" else TTUltraMIGSModule5D._morton_code_10bit(xyz01[:, 0], xyz01[:, 1], xyz01[:, 2])
        return torch.from_numpy(np.argsort(codes)).long()

    @staticmethod
    def _morton_code_10bit(x, y, z):
        def q(u):
            u = np.clip(u, 0, 1)
            return (u * 1023 + 0.5).astype(np.uint32)
        xi, yi, zi = q(x), q(y), q(z)
        def part1by2(n):
            n = (n | (n << 16)) & 0x030000FF
            n = (n | (n << 8)) & 0x0300F00F
            n = (n | (n << 4)) & 0x030C30C3
            n = (n | (n << 2)) & 0x09249249
            return n
        return (part1by2(xi) << 2) | (part1by2(yi) << 1) | part1by2(zi)

    @staticmethod
    def _adjacency_cost(order: np.ndarray, xyz: np.ndarray, shape: tuple) -> float:
        n1, n2, n3 = shape
        idx = order.reshape(n1, n2, n3)
        pts = xyz[idx]
        def axis_cost(a):
            front = np.take(pts, range(0, shape[a]-1), axis=a)
            back  = np.take(pts, range(1, shape[a]  ), axis=a)
            d = back - front
            return np.sum((d*d).sum(axis=-1))
        return axis_cost(0) + axis_cost(1) + axis_cost(2)

    @staticmethod
    def _balanced_shape_for(G: int) -> tuple:
        best = None
        lim1 = int(round(G ** (1/3))) + 2
        for n1 in range(1, lim1 + 1):
            if G % n1:
                continue
            G1 = G // n1
            lim2 = int(round(G1 ** 0.5)) + 2
            for n2 in range(n1, lim2 + 1):
                if G1 % n2:
                    continue
                n3 = G1 // n2
                if n2 > n3:
                    continue
                score = n3 - n1
                if (best is None) or (score < best[0]):
                    best = (score, (n1, n2, n3))
        if best is None:
            n2 = max(1, int(np.sqrt(G)))
            n3 = G // n2
            return (1, min(n2, n3), max(n2, n3))
        return best[1]

    @staticmethod
    def _candidate_shapes(G: int) -> list:
        a, b, c = TTUltraMIGSModule5D._balanced_shape_for(G)
        perms = {(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)}
        return list(perms)

    @staticmethod
    def _pick_best_shape(order_t: torch.Tensor, xyz_t: torch.Tensor, candidates: list) -> tuple:
        order = order_t.cpu().numpy()
        xyz = xyz_t.detach().cpu().numpy()
        scored = [(sh, TTUltraMIGSModule5D._adjacency_cost(order, xyz, sh)) for sh in candidates]
        scored.sort(key=lambda t: t[1])
        return scored[0][0], scored


    def recombine_core4(self):
        return torch.cat(
            [self.core4_xyz, self.core4_scaling, self.core4_rotation,
             self.core4_dc, self.core4_rest, self.core4_opacity],
            dim=1
        )  # (r4, 43, r5)

    def get_core0(self, idx):
        assert 0 <= idx < self.tt_tensor_gpu[0].shape[1], f"Invalid identity index {idx}"
        return self.tt_tensor_gpu[0][:, idx:idx+1, :]  # (1, 1, r1)

    def expand_first_core(self, n_identities):
        """Duplicate identity axis (core 0) to the target size."""
        if not len(self.tt_tensor_gpu):
            raise RuntimeError("TT cores must be initialized before expansion.")
        first = self.tt_tensor_gpu[0]   # (1, N, r1)
        r0, n_cur, r1 = first.shape
        if n_cur >= n_identities:
            if self.verbose:
                print(f"[TT] identity axis already ≥ {n_identities}")
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
        """Full dense reconstruction as a tensorly TT → dense."""
        full = list(self.tt_tensor_gpu[:4]) + [self.recombine_core4()]
        return tt_to_tensor(full)

    def get_tt_tensor(self, idx=None):
        core0 = self.get_core0(idx) if idx is not None else self.tt_tensor_gpu[0]
        cores = [core0, self.tt_tensor_gpu[1], self.tt_tensor_gpu[2], self.tt_tensor_gpu[3], self.recombine_core4()]
        if self.use_mars and (self.mars_logits is not None) and self.mars_active:
            offset = 1 if self.mars_cfg["mask_core0"] else 0
            cores[1] = self._apply_mask_core (cores[1], self.mars_logits[offset + 0])
            cores[2] = self._apply_mask_core (cores[2], self.mars_logits[offset + 1])
            cores[3] = self._apply_mask_core (cores[3], self.mars_logits[offset + 2])
            cores[4] = self._apply_mask_core4(cores[4], self.mars_logits[offset + 3])
        return cores

    def _contract_tt_identity_gemm(self, idx: int) -> torch.Tensor:
        """Fast GEMM path for (I=1) contraction, returning W_perm (G, M)."""
        c0 = self.get_core0(idx)          # (1, 1, r1)
        c1 = self.tt_tensor_gpu[1]        # (r1, n1, r2)
        c2 = self.tt_tensor_gpu[2]        # (r2, n2, r3)
        c3 = self.tt_tensor_gpu[3]        # (r3, n3, r4)
        c4 = self.recombine_core4()       # (r4, M, r5)

        r1, n1, r2 = c1.shape
        _,  n2, r3 = c2.shape
        _,  n3, r4 = c3.shape
        r4_c4, M, r5 = c4.shape
        assert c0.shape == (1, 1, r1), "core0 incompatible with c1"
        assert r4_c4 == r4, "core4 r4 mismatch"
        assert r5 == 1, "last TT rank must be 1 for this path"

        X = c0.reshape(1, r1) @ c1.reshape(r1, n1 * r2)
        X = X.reshape(n1, r2) @ c2.reshape(r2, n2 * r3)
        X = X.reshape(n1 * n2, r3) @ c3.reshape(r3, n3 * r4)
        X = X.reshape(n1 * n2 * n3, r4)          # (G, r4)
        C4 = c4.squeeze(-1).reshape(r4, M)       # (r4, M)
        return X @ C4


    def get_W_for_identity(self, idx: int, original_order: bool = True) -> torch.Tensor:
        """Return (G, M) for identity idx; optionally undo spatial permutation."""
        try:
            W_perm = self._contract_tt_identity_gemm(idx)
        except AssertionError:
            T = tt_to_tensor(self.get_tt_tensor(idx))   # fallback
            M = T.shape[-1]
            W_perm = T.squeeze(0).contiguous().view(-1, M)
        # if original_order and hasattr(self, "inv_perm"):
        #     return W_perm[self.inv_perm.to(W_perm.device)]
        return W_perm


    def optimize_parameters(self):
        return list(self.tt_tensor_gpu[:4]) + [
            self.core4_xyz, self.core4_scaling, self.core4_rotation,
            self.core4_dc, self.core4_rest, self.core4_opacity,
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
                p.requires_grad = False  # enabled later

    def set_optimizer(self, opt_cfg):
        """Create optimizer and LR scheduler for TT cores and core4 slices."""
        self._opt_cfg = dict(opt_cfg) if opt_cfg is not None else {}
        tt_lrs = self._opt_cfg.get("tt_lrs", [1.6e-4] * 4)
        tt_final_lrs = self._opt_cfg.get("tt_final_lrs", [1.6e-6] * 4)
        tt_decay_iters = self._opt_cfg.get("tt_decay_iters", 50000)

        param_groups = []
        # cores 0..3 with decay
        for i in range(len(self.tt_tensor_gpu) - 1):
            param_groups.append({
                "params": [self.tt_tensor_gpu[i]],
                "lr": tt_lrs[i], "initial_lr": tt_lrs[i], "final_lr": tt_final_lrs[i]
            })
        # xyz slice with decay
        param_groups.append({
            "params": [self.core4_xyz],
            "lr": 1.6e-4, "initial_lr": 1.6e-4, "final_lr": 1.6e-6
        })
        # other slices fixed LR
        param_groups += [
            {"params": [self.core4_scaling],  "lr": 5e-3},
            {"params": [self.core4_rotation], "lr": 1e-3},
            {"params": [self.core4_dc],       "lr": 2.5e-3},
            {"params": [self.core4_rest],     "lr": 2.5e-3},
            {"params": [self.core4_opacity],  "lr": 5e-2},
        ]

        self._mars_in_optimizer = False
        self.optimizer = torch.optim.Adam(param_groups)

        if any("final_lr" in g for g in param_groups):
            gamma = (1.6e-6 / 1.6e-4) ** (1.0 / tt_decay_iters)
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
        # Finetune fast-path (core0 slice + optional texture MLP lives elsewhere)
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

        # MARS activation gate
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

    def _apply_mask_core4(self, core4, logit):
        if not (self.use_mars and self.mars_active):
            return core4
        m = self._hard_concrete(logit, self.training)
        return core4 * m


    @torch.no_grad()
    def add_identity(self, noise_scale: float = 0.05, rebuild_optimizer: bool = True) -> int:
        """
        Append one identity to the first TT core (identity axis) using a neutral init:
        mean + noise_scale * std * N(0,1), rescaled to the median row norm.
        """
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
        """
        Finetune only the identity slice `idx` of core0 (mask gradients);
        optionally include the color MLP.
        """
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
