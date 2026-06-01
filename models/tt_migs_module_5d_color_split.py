import math
import os
import numpy as np
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
    min_r1=16,
    r1_cap=None,
    identity_bias=0.90,
    mid_boost=1.15,
    weight_power=1.15,
):
    I = int(shape[0]); M = int(shape[-1])
    mids = [int(x) for x in shape[1:-1]]
    K = len(mids)
    if K < 1:
        raise ValueError("shape must contain at least one G-mode between I and M")

    G = 1
    for x in mids: G *= x
    cp_params = cp_R * (I + G + M)
    budget = float(target_ratio) * float(cp_params)

    if rM_hint is None:
        if target_ratio <= 0.051:   rM = min(cap_last, 32)
        elif target_ratio <= 0.11:  rM = min(cap_last, 38)
        else:                       rM = min(cap_last, 43)
    else:
        rM = min(cap_last, int(rM_hint))

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

    w = [float(n)**float(weight_power) for n in mids]
    if K > 0:
        w[0] *= float(identity_bias)
        for j in range(1, K):
            w[j] *= float(mid_boost)
    s = sum(w); w = [x/s for x in w] if s > 0 else [1.0/K]*K

    def P_of_alpha(alpha: float) -> float:
        r = [max(1e-9, alpha*wj) for wj in w]
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

    r_float = [alpha*wj for wj in w]
    r_int = [max(1, int(round(x))) for x in r_float]
    r_int[0] = max(r_int[0], int(min_r1))
    if r1_cap is not None:
        r_int[0] = min(r_int[0], int(r1_cap))

    ranks = [1] + r_int + [rM, 1]

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
        for j in range(1, 1+K):
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


class TTUltraMIGSModule5DColorSplit(nn.Module):
    """
    TT-MIGS avec compression séparée géométrie / couleur :

      TT_geo   : (I, n1, n2, n3, GEO_DIM=11)  — axe identité conservé
                  params : [xyz(3), scaling(3), rotation(4), opacity(1)]

      TT_color : (n1, n2, n3, COLOR_DIM=32)   — UNE TT par identité, PAS d'axe identité
                  params : [dc(1), rest(31)]

    Mêmes rangs spatiaux [r_n1, r_n2] pour les deux branches.
    Sortie get_W_for_identity → (G, 43) dans l'ordre original :
      [xyz | scaling | rotation | dc | rest | opacity]
    """
    GEO_DIM   = 11   # xyz(3) + scaling(3) + rotation(4) + opacity(1)
    COLOR_DIM = 32   # dc(1)  + rest(31)

    # ──────────────────────────────────────────────────────────────────
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        tt_cfg = cfg.migs if not isinstance(cfg, dict) else cfg["migs"]
        self._base_seed = int(getattr(cfg, "seed", 123))

        self.tt_delay = tt_cfg.get("delay", 1000)
        if self.tt_delay is None:
            self.tt_delay = cfg.model.gaussian.get("delay", 0)

        self.tt_rank_geo   = None
        self.tt_rank_color = None
        self.tt_shape      = tt_cfg.get("tt_shape")
        self.verbose       = bool(tt_cfg.get("verbose", False))

        self.optimizer  = None
        self.scheduler  = None
        self._opt_cfg   = None
        self._needs_opt_rebuild = False
        self._tt_unfrozen       = False

        # MARS désactivé dans cette variante
        self.use_mars    = False
        self.mars_logits = None
        self.mars_active = False

        # ── GEO TT : 5 cores (c0…c4), c4 découpé en slices ──────────
        self.tt_tensor_gpu      = nn.ParameterList()
        self.core4_geo_xyz      = nn.Parameter(torch.zeros(1, 3, 1))
        self.core4_geo_scaling  = nn.Parameter(torch.zeros(1, 3, 1))
        self.core4_geo_rotation = nn.Parameter(torch.zeros(1, 4, 1))
        self.core4_geo_opacity  = nn.Parameter(torch.zeros(1, 1, 1))

        # ── COLOR TT : une ParameterList(4 cores) par identité ───────
        self.tt_color_list = nn.ModuleList()
        self._n_color_ids  = 0

        self.save_dir = getattr(self.cfg, "hilbert_vis_5d", "./exports")
        os.makedirs(self.save_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # helpers internes
    # ──────────────────────────────────────────────────────────────────
    def _stream(self, tag: str, device) -> torch.Generator:
        h = int.from_bytes(hashlib.md5(tag.encode("utf8")).digest()[:8], 'little')
        g = torch.Generator(device=device)
        g.manual_seed(self._base_seed ^ h)
        return g

    def _randn_like(self, ref, tag):
        g = self._stream(tag, ref.device)
        return torch.randn(ref.shape, device=ref.device, dtype=ref.dtype, generator=g)

    def _rand_like(self, ref, tag):
        g = self._stream(tag, ref.device)
        return torch.rand(ref.shape, device=ref.device, dtype=ref.dtype, generator=g)

    @staticmethod
    def _repeat_to(x: torch.Tensor, dim: int, target: int) -> torch.Tensor:
        cur = x.shape[dim]
        if cur == target:
            return x
        times = math.ceil(target / cur)
        reps = [1] * x.dim(); reps[dim] = times
        x_rep = x.repeat(*reps)
        sl = [slice(None)] * x.dim(); sl[dim] = slice(0, target)
        return x_rep[tuple(sl)]

    # ──────────────────────────────────────────────────────────────────
    # INITIALISATION
    # ──────────────────────────────────────────────────────────────────
    def init_from_tensor(self, gaussian_model):
        """
        Construit les deux branches TT à partir des paramètres gaussiens courants.
        Appelé une seule fois avec la première identité (I=1).
        """
        G = gaussian_model._xyz.shape[0]
        print(f"[TTColorSplit] init_from_tensor  G={G}")

        xyz           = gaussian_model._xyz
        scaling       = gaussian_model._scaling
        rotation      = gaussian_model._rotation
        features_dc   = gaussian_model._features_dc.squeeze(-1)
        features_rest = gaussian_model._features_rest.squeeze(-1)
        opacity       = gaussian_model._opacity

        all_params = [xyz, scaling, rotation, features_dc, features_rest, opacity]
        W_GM = torch.cat(
            [p if p.ndim == 2 else p.view(p.shape[0], -1) for p in all_params], dim=1
        )
        # W_GM : (G, 43) = [xyz(0:3)|scal(3:6)|rot(6:10)|dc(10:11)|rest(11:42)|opa(42:43)]

        # ── ordre Hilbert ─────────────────────────────────────────────
        with _Timer() as t_h:
            perm = self._build_spatial_order_from_xyz(W_GM[:, :3], method="hilbert", bits=15)
        print(f"[COST] hilbert={t_h.dt:.4f}s")

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(G, device=perm.device)
        self.register_buffer("perm",     perm)
        self.register_buffer("inv_perm", inv_perm)
        W_perm = W_GM[self.perm.to(W_GM.device)]   # (G, 43)

        # ── forme spatiale équilibrée ─────────────────────────────────
        with _Timer() as t_shape:
            candidates = self._candidate_shapes(G)
            best_shape, scored = self._pick_best_shape(self.perm, W_GM[:, :3], candidates)
        print(f"[COST] shape_search={t_shape.dt:.4f}s  picked={best_shape}")
        assert best_shape[0] * best_shape[1] * best_shape[2] == G
        n1, n2, n3 = best_shape
        self.n1, self.n2, self.n3 = n1, n2, n3

        # ── séparation géo / couleur ──────────────────────────────────
        #  GEO   : [xyz(0:3), scal(3:6), rot(6:10), opa(42:43)] → (G, 11)
        W_geo   = torch.cat([W_perm[:, 0:10], W_perm[:, 42:43]], dim=1)
        #  COLOR : [dc(10:11), rest(11:42)]                      → (G, 32)
        W_color = W_perm[:, 10:42]

        # ── sélection des rangs ───────────────────────────────────────
        migs_cfg     = self.cfg.migs if not isinstance(self.cfg, dict) else self.cfg["migs"]
        final_I      = int(migs_cfg.get("final_identities"))
        target_ratio = float(migs_cfg.get("target_ratio"))
        cap_last     = int(migs_cfg.get("cap_last"))
        min_r1       = int(migs_cfg.get("min_r1"))
        r1_cap       = migs_cfg.get("r1_cap")
        rM_hint      = migs_cfg.get("rM_hint")
        identity_bias= float(migs_cfg.get("identity_bias"))
        mid_boost    = float(migs_cfg.get("mid_boost"))
        weight_power = float(migs_cfg.get("weight_power"))

        # rangs géo pour forme [final_I, n1, n2, n3, 11]
        cap_geo     = min(cap_last, self.GEO_DIM)
        rM_hint_geo = min(self.GEO_DIM, int(rM_hint)) if rM_hint is not None else None
        ranks_geo, p_geo, b_geo, rM_geo, _ = suggest_tt_ranks_weighted(
            [final_I, n1, n2, n3, self.GEO_DIM],
            target_ratio=target_ratio, cp_R=100, cap_last=cap_geo,
            rM_hint=rM_hint_geo, min_r1=min_r1, r1_cap=r1_cap,
            identity_bias=identity_bias, mid_boost=mid_boost, weight_power=weight_power,
        )
        # rangs couleur : mêmes rangs spatiaux r_n1=ranks_geo[2], r_n2=ranks_geo[3]
        r_n1       = ranks_geo[2]
        r_n2       = ranks_geo[3]
        rM_color   = min(self.COLOR_DIM, cap_last)
        ranks_color = [1, r_n1, r_n2, rM_color, 1]

        self.tt_rank_geo   = ranks_geo
        self.tt_rank_color = ranks_color
        print(f"[TT] ranks_geo   = {ranks_geo}  (params {p_geo}/{b_geo}  rM={rM_geo})")
        print(f"[TT] ranks_color = {ranks_color}  (mêmes rangs spatiaux r_n1={r_n1} r_n2={r_n2})")

        # ── TT-SVD géo : (1, n1, n2, n3, 11) ────────────────────────
        W_geo_tt = W_geo.unsqueeze(0).reshape(1, n1, n2, n3, self.GEO_DIM)
        ranks_geo_init    = list(ranks_geo)
        ranks_geo_init[1] = 1   # rang identité = 1 pour le SVD (I=1 à l'init)

        with _Timer() as t:
            tt_geo = tensor_train(W_geo_tt, rank=ranks_geo_init, verbose=self.verbose)
        print(f"[COST] tt-svd geo={t.dt:.4f}s")

        self.tt_tensor_gpu = nn.ParameterList(
            [nn.Parameter(c.to(W_geo_tt.device)) for c in tt_geo.factors]
        )

        # découpage du dernier core géo
        c4 = self.tt_tensor_gpu[4]   # (rM_geo, 11, 1)
        self.core4_geo_xyz      = nn.Parameter(c4[:, 0:3,  :].detach().clone())
        self.core4_geo_scaling  = nn.Parameter(c4[:, 3:6,  :].detach().clone())
        self.core4_geo_rotation = nn.Parameter(c4[:, 6:10, :].detach().clone())
        self.core4_geo_opacity  = nn.Parameter(c4[:, 10:11,:].detach().clone())

        # expansion rang identité : 1 → ranks_geo[1]
        with torch.no_grad():
            c0 = self.tt_tensor_gpu[0]   # (1, 1, r_id_cur)
            c1 = self.tt_tensor_gpu[1]   # (r_id_cur, n1, r_n1_cur)
            r_id_cur = c0.shape[2]
            r_id_tgt = max(r_id_cur, ranks_geo[1])
            if r_id_tgt != r_id_cur:
                scale  = r_id_cur / float(r_id_tgt)
                c0_new = self._repeat_to(c0.detach(), dim=2, target=r_id_tgt) * scale
                c1_new = self._repeat_to(c1.detach(), dim=0, target=r_id_tgt)
                self.tt_tensor_gpu[0] = nn.Parameter(c0_new)
                self.tt_tensor_gpu[1] = nn.Parameter(c1_new)
                self._needs_opt_rebuild = True

        # expansion r_n1, r_n2, rM_geo par zero-padding
        self._expand_geo_ranks_preserve(ranks_geo)
        print(f"[TT-GEO] shapes après expansion:")
        print(f"  c0 (identité) : {tuple(self.tt_tensor_gpu[0].shape)}")
        print(f"  c1            : {tuple(self.tt_tensor_gpu[1].shape)}")
        print(f"  c2            : {tuple(self.tt_tensor_gpu[2].shape)}")
        print(f"  c3            : {tuple(self.tt_tensor_gpu[3].shape)}")
        print(f"  c4 géo        : {tuple(self.recombine_core4_geo().shape)}")


        if self.verbose:
            print("[TT] geo core shapes:")
            for i, c in enumerate(self.tt_tensor_gpu[:4]):
                print(f"  c{i} = {tuple(c.shape)}")
            print(f"  c4 (geo recombined) = {tuple(self.recombine_core4_geo().shape)}")

        # ── TT-SVD couleur : (n1, n2, n3, 32) — identité 0 ──────────
        W_color_tt = W_color.reshape(n1, n2, n3, self.COLOR_DIM)

        with _Timer() as t:
            tt_color = tensor_train(W_color_tt, rank=ranks_color, verbose=self.verbose)
        print(f"[COST] tt-svd color={t.dt:.4f}s")


        color_pl = nn.ParameterList(
            [nn.Parameter(c.to(W_color_tt.device)) for c in tt_color.factors]
        )
        self.tt_color_list = nn.ModuleList([color_pl])
        self._n_color_ids  = 1
        self._expand_color_ranks_preserve(self.tt_color_list[0], ranks_color)

        print(f"[TT-COLOR] shapes identité 0 après expansion:")
        for i, c in enumerate(self.tt_color_list[0]):
            print(f"  cc{i} : {tuple(c.shape)}")

        if self.verbose:
            print("[TT] color core shapes (id 0):")
            for i, c in enumerate(self.tt_color_list[0]):
                print(f"  cc{i} = {tuple(c.shape)}")

        # ── diagnostics ───────────────────────────────────────────────
        W_rec = self.get_W_for_identity(0).to(W_GM.device)
        compare_reconstruction_per_block(
            W_GM, W_rec, split_sizes=[3, 3, 4, 1, 31, 1],
            names=['xyz', 'scaling', 'rotation', 'dc', 'rest', 'opacity']
        )
        plot_correlation_across_parameters(W_GM, W_rec)
        plot_pca_groupwise_xyz_auto(W_GM, W_rec, num_groups=10)

    # ──────────────────────────────────────────────────────────────────
    # ZERO-PAD EXPANSION (géo)
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _zero_pad_pair_preserve(self, left, right, add, dim_left, dim_right):
        if add <= 0:
            return left, right
        dev = left.device
        dl = list(left.shape);  dl[dim_left]  = add
        dr = list(right.shape); dr[dim_right] = add
        return (
            torch.cat([left,  torch.zeros(dl,  device=dev, dtype=left.dtype)],  dim=dim_left),
            torch.cat([right, torch.zeros(dr,  device=dev, dtype=right.dtype)], dim=dim_right),
        )

    @torch.no_grad()
    def _expand_geo_ranks_preserve(self, ranks_target):
        """
        Zero-pad les rangs géo [r_n1, r_n2, rM_geo] pour atteindre les valeurs cibles
        sans changer le tenseur représenté.
        ranks_target = [1, r_id, r_n1, r_n2, rM_geo, 1]
        """
        # ── r_n1 : entre c1 (dim 2) et c2 (dim 0) ──
        c1 = self.tt_tensor_gpu[1]; c2 = self.tt_tensor_gpu[2]
        r_n1_cur = c1.shape[2]; r_n1_tgt = int(ranks_target[2])
        if r_n1_tgt > r_n1_cur:
            c1, c2 = self._zero_pad_pair_preserve(c1, c2, r_n1_tgt - r_n1_cur, 2, 0)
            self.tt_tensor_gpu[1] = nn.Parameter(c1)
            self.tt_tensor_gpu[2] = nn.Parameter(c2)

        # ── r_n2 : entre c2 (dim 2) et c3 (dim 0) ──
        c2 = self.tt_tensor_gpu[2]; c3 = self.tt_tensor_gpu[3]
        r_n2_cur = c2.shape[2]; r_n2_tgt = int(ranks_target[3])
        if r_n2_tgt > r_n2_cur:
            c2, c3 = self._zero_pad_pair_preserve(c2, c3, r_n2_tgt - r_n2_cur, 2, 0)
            self.tt_tensor_gpu[2] = nn.Parameter(c2)
            self.tt_tensor_gpu[3] = nn.Parameter(c3)

        # ── rM_geo : entre c3 (dim 2) et c4 (dim 0) ──
        c3 = self.tt_tensor_gpu[3]
        rM_cur = c3.shape[2]; rM_tgt = int(ranks_target[4])
        if rM_tgt > rM_cur:
            core4 = self.recombine_core4_geo()   # (rM_cur, 11, 1)
            c3, core4 = self._zero_pad_pair_preserve(c3, core4, rM_tgt - rM_cur, 2, 0)
            self.tt_tensor_gpu[3] = nn.Parameter(c3)
            self.core4_geo_xyz      = nn.Parameter(core4[:, 0:3,  :])
            self.core4_geo_scaling  = nn.Parameter(core4[:, 3:6,  :])
            self.core4_geo_rotation = nn.Parameter(core4[:, 6:10, :])
            self.core4_geo_opacity  = nn.Parameter(core4[:, 10:11,:])

        self._needs_opt_rebuild = True


    @torch.no_grad()
    def _expand_color_ranks_preserve(self, color_pl: nn.ParameterList, ranks_target_color: list):
        """
        Zero-pad les rangs couleur [r1c, r2c, rMc] pour atteindre les valeurs cibles.
        ranks_target_color = [1, r1c, r2c, rMc, 1]
        """
        c0, c1, c2, c3 = color_pl[0], color_pl[1], color_pl[2], color_pl[3]

        # r1c : entre c0 (dim 2) et c1 (dim 0)
        r1c_cur = c0.shape[2]; r1c_tgt = int(ranks_target_color[1])
        if r1c_tgt > r1c_cur:
            c0, c1 = self._zero_pad_pair_preserve(c0, c1, r1c_tgt - r1c_cur, 2, 0)
            color_pl[0] = nn.Parameter(c0)
            color_pl[1] = nn.Parameter(c1)

        c0, c1, c2, c3 = color_pl[0], color_pl[1], color_pl[2], color_pl[3]

        # r2c : entre c1 (dim 2) et c2 (dim 0)
        r2c_cur = c1.shape[2]; r2c_tgt = int(ranks_target_color[2])
        if r2c_tgt > r2c_cur:
            c1, c2 = self._zero_pad_pair_preserve(c1, c2, r2c_tgt - r2c_cur, 2, 0)
            color_pl[1] = nn.Parameter(c1)
            color_pl[2] = nn.Parameter(c2)

        c2 = color_pl[2]; c3 = color_pl[3]

        # rMc : entre c2 (dim 2) et c3 (dim 0)
        rMc_cur = c2.shape[2]; rMc_tgt = int(ranks_target_color[3])
        if rMc_tgt > rMc_cur:
            c2, c3 = self._zero_pad_pair_preserve(c2, c3, rMc_tgt - rMc_cur, 2, 0)
            color_pl[2] = nn.Parameter(c2)
            color_pl[3] = nn.Parameter(c3)

    # ──────────────────────────────────────────────────────────────────
    # RECONSTRUCTION
    # ──────────────────────────────────────────────────────────────────
    def recombine_core4_geo(self):
        """Recompose le dernier core géo : (rM_geo, 11, 1)."""
        return torch.cat([
            self.core4_geo_xyz,
            self.core4_geo_scaling,
            self.core4_geo_rotation,
            self.core4_geo_opacity,
        ], dim=1)

    # alias pour compatibilité avec les wrappers MARS / adapters
    def recombine_core4(self):
        return self.recombine_core4_geo()

    def get_core0(self, idx: int):
        assert 0 <= idx < self.tt_tensor_gpu[0].shape[1], f"Identité {idx} invalide"
        return self.tt_tensor_gpu[0][:, idx:idx+1, :]   # (1, 1, r_id)

    def _contract_geo_gemm(self, idx: int) -> torch.Tensor:
        """Contraction GEMM géo → (G, 11) : [xyz, scaling, rotation, opacity]."""
        c0 = self.get_core0(idx)           # (1, 1, r_id)
        c1 = self.tt_tensor_gpu[1]         # (r_id, n1, r_n1)
        c2 = self.tt_tensor_gpu[2]         # (r_n1, n2, r_n2)
        c3 = self.tt_tensor_gpu[3]         # (r_n2, n3, rM_geo)
        c4 = self.recombine_core4_geo()    # (rM_geo, 11, 1)

        r_id, n1, r_n1 = c1.shape
        _,    n2, r_n2 = c2.shape
        _,    n3, rM   = c3.shape

        X = c0.reshape(1, r_id) @ c1.reshape(r_id, n1 * r_n1)
        X = X.reshape(n1, r_n1) @ c2.reshape(r_n1, n2 * r_n2)
        X = X.reshape(n1 * n2, r_n2) @ c3.reshape(r_n2, n3 * rM)
        X = X.reshape(n1 * n2 * n3, rM)
        return X @ c4.squeeze(-1)   # (G, 11)

    def _contract_color_gemm(self, idx: int) -> torch.Tensor:
        """Contraction GEMM couleur → (G, 32) : [dc, rest]."""
        cores = self.tt_color_list[idx]
        c0, c1, c2, c3 = cores[0], cores[1], cores[2], cores[3]
        # c0:(1,n1,r1c)  c1:(r1c,n2,r2c)  c2:(r2c,n3,rMc)  c3:(rMc,32,1)

        _, n1, r1c = c0.shape
        _, n2, r2c = c1.shape
        _, n3, rMc = c2.shape

        X = c0.reshape(n1, r1c)   # r0=1 → squeeze
        X = X @ c1.reshape(r1c, n2 * r2c)
        X = X.reshape(n1 * n2, r2c) @ c2.reshape(r2c, n3 * rMc)
        X = X.reshape(n1 * n2 * n3, rMc) @ c3.squeeze(-1)   # (G, 32)
        return X

    def get_W_for_identity(self, idx: int, original_order: bool = True) -> torch.Tensor:
        """
        Retourne (G, 43) dans l'ordre original :
          [xyz(3) | scaling(3) | rotation(4) | dc(1) | rest(31) | opacity(1)]
        """
        W_geo   = self._contract_geo_gemm(idx)    # (G,11): [xyz(0:3),scal(3:6),rot(6:10),opa(10:11)]
        W_color = self._contract_color_gemm(idx)  # (G,32): [dc(0:1), rest(1:32)]
        # recomposition → ordre attendu par scene.py / update_gaussians_from_migs
        return torch.cat([W_geo[:, 0:10], W_color, W_geo[:, 10:11]], dim=1)   # (G, 43)

    # ──────────────────────────────────────────────────────────────────
    # GESTION DES IDENTITÉS
    # ──────────────────────────────────────────────────────────────────
    def expand_first_core(self, n_identities: int):
        """
        Étend l'axe identité géo (core0) ET clone les TT couleur
        pour chaque nouvelle identité.
        """
        if not len(self.tt_tensor_gpu):
            raise RuntimeError("Appeler init_from_tensor avant expand_first_core.")

        # géo core0
        first = self.tt_tensor_gpu[0]   # (1, N_cur, r_id)
        _, n_cur, _ = first.shape
        if n_cur < n_identities:
            base  = first[:, 0:1, :].detach()
            rep   = base.repeat(1, n_identities, 1)
            noise = self._randn_like(rep, tag="core0_expand_noise") * 1e-3
            self.tt_tensor_gpu[0] = nn.Parameter(rep + noise)
            self._needs_opt_rebuild = True

        # couleur : clonage pour chaque nouvelle identité
        while self._n_color_ids < n_identities:
            self._clone_color_identity(src_idx=0)

        if self.optimizer is not None and self._needs_opt_rebuild:
            self._rebuild_optimizer_like_before()
            self._needs_opt_rebuild = False

        if self.verbose:
            print(f"[TT] expand_first_core → geo {self.tt_tensor_gpu[0].shape}, "
                  f"n_color={self._n_color_ids}")

        print(f"[expand_first_core] GEO  core0 : {tuple(self.tt_tensor_gpu[0].shape)}")
        print(f"[expand_first_core] COLOR n_ids : {self._n_color_ids}")
        for i, pl in enumerate(self.tt_color_list):
            print(f"  color[{i}] cc0={tuple(pl[0].shape)} cc1={tuple(pl[1].shape)} "
                f"cc2={tuple(pl[2].shape)} cc3={tuple(pl[3].shape)}")

    def _clone_color_identity(self, src_idx: int = 0):
        """Clone une TT couleur existante avec un bruit additif."""
        src    = self.tt_color_list[src_idx]
        new_pl = nn.ParameterList()
        for p in src:
            noise = torch.randn_like(p.data) * 1e-3
            new_pl.append(nn.Parameter(p.data.clone() + noise))
        self.tt_color_list.append(new_pl)
        self._n_color_ids += 1
        self._needs_opt_rebuild = True

    @torch.no_grad()
    def add_identity(self, noise_scale: float = 0.05, rebuild_optimizer: bool = True) -> int:
        """
        Ajoute une identité :
          - slice géo dans core0 (même logique que TTUltraMIGSModule5D)
          - clone TT couleur de l'identité 0
        """
        # ── géo ──
        core0 = self.tt_tensor_gpu[0]   # (1, N, r_id)
        r0, n_id, r1 = core0.shape
        if n_id > 0:
            U   = core0.detach()
            mu  = U.mean(dim=1, keepdim=True)
            sig = U.std(dim=1, unbiased=False, keepdim=True).clamp_(min=1e-8)
            eps = self._randn_like(mu, tag="add_identity_noise").expand_as(mu)
            new_row = mu + noise_scale * sig * eps
            norms = U.view(n_id, r1).norm(dim=1)
            target_norm = norms.median()
            cur_norm    = new_row.view(-1).norm()
            new_row = new_row / (cur_norm + 1e-8) * float(target_norm)
        else:
            new_row = self._randn_like(core0[:, :1, :], tag="add_identity_boot") * 0.02
        self.tt_tensor_gpu[0] = nn.Parameter(torch.cat([core0, new_row], dim=1))

        # ── couleur ──
        self._clone_color_identity(src_idx=0)

        self._needs_opt_rebuild = True
        if rebuild_optimizer and self.optimizer is not None:
            self._rebuild_optimizer_like_before()

        new_idx = self.tt_tensor_gpu[0].shape[1] - 1
        if self.verbose:
            print(f"[TT] add_identity → idx={new_idx}, geo core0={self.tt_tensor_gpu[0].shape}, "
                  f"n_color={self._n_color_ids}")
        return new_idx

    # ──────────────────────────────────────────────────────────────────
    # CONTRÔLE D'ENTRAÎNEMENT
    # ──────────────────────────────────────────────────────────────────
    def optimize_parameters(self):
        """Liste tous les paramètres à optimiser (géo + couleur toutes identités)."""
        params = list(self.tt_tensor_gpu[:4]) + [
            self.core4_geo_xyz,
            self.core4_geo_scaling,
            self.core4_geo_rotation,
            self.core4_geo_opacity,
        ]
        for pl in self.tt_color_list:
            params.extend(list(pl))
        return params

    def freeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = False

    def unfreeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = True

    def set_optimizer(self, opt_cfg):
        """Crée l'optimiseur Adam avec des LR différentes par groupe."""
        self._opt_cfg    = dict(opt_cfg) if opt_cfg is not None else {}
        # ── Guard : cores pas encore initialisés (ex: predict/skip_init) ──
        if len(self.tt_tensor_gpu) == 0 or len(self.tt_color_list) == 0:
            self._needs_opt_rebuild = True
            return

        tt_lrs           = self._opt_cfg.get("tt_lrs",       [1.6e-4] * 4)
        tt_final_lrs     = self._opt_cfg.get("tt_final_lrs", [1.6e-6] * 4)
        tt_decay         = self._opt_cfg.get("tt_decay_iters", 50000)

        param_groups = []

        # ── cores spatiaux géo 0..3 avec decay ──
        for i in range(4):
            param_groups.append({
                "params":     [self.tt_tensor_gpu[i]],
                "lr":         tt_lrs[i],
                "initial_lr": tt_lrs[i],
                "final_lr":   tt_final_lrs[i],
            })

        # ── slices dernier core géo ──
        param_groups += [
            {"params": [self.core4_geo_xyz],      "lr": 1.6e-4, "initial_lr": 1.6e-4, "final_lr": 1.6e-6},
            {"params": [self.core4_geo_scaling],  "lr": 5e-3},
            {"params": [self.core4_geo_rotation], "lr": 1e-3},
            {"params": [self.core4_geo_opacity],  "lr": 5e-2},
        ]

        # ── TT couleur de toutes les identités ──
        for i, pl in enumerate(self.tt_color_list):
            # cores spatiaux couleur (0,1,2) : même LR que cores géo spatiaux
            for j in range(3):
                param_groups.append({"params": [pl[j]], "lr": 1.6e-4})
            # dernier core couleur (dc + rest fusionnés) : LR features
            param_groups.append({"params": [pl[3]], "lr": 2.5e-3})

        self.optimizer = torch.optim.Adam(param_groups)

        # scheduler exponentiel sur les groupes avec final_lr
        gamma = (1.6e-6 / 1.6e-4) ** (1.0 / max(int(tt_decay), 1))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        self._needs_opt_rebuild = False

    def _rebuild_optimizer_like_before(self):
        self.set_optimizer(self._opt_cfg)

    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, iteration=None):
        if self.optimizer is None:
            return

        if iteration is not None and iteration < self.tt_delay:
            self.freeze_tt_parameters()
            self.optimizer.zero_grad()
            return

        if iteration is not None and not self._tt_unfrozen and iteration >= self.tt_delay:
            self.unfreeze_tt_parameters()
            self._tt_unfrozen = True
            if self.verbose:
                print(f"[TT] cores dégelés à iter {iteration}")

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_learning_rate()

    # stub MARS (pour compatibilité avec wrappers existants)
    def loss_mars(self):
        dev = self.tt_tensor_gpu[0].device if len(self.tt_tensor_gpu) else "cpu"
        return torch.tensor(0.0, device=dev)

    def update_mars_schedule(self, iteration: int):
        pass

    # ──────────────────────────────────────────────────────────────────
    # MÉTHODES STATIQUES (identiques à TTUltraMIGSModule5D)
    # ──────────────────────────────────────────────────────────────────
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
                fn  = getattr(hc, name)
                out = np.empty_like(xi, dtype=np.int64)
                for i, (a, b, c) in enumerate(zip(xi, yi, zi)):
                    out[i] = int(fn([int(a), int(b), int(c)]))
                return out
        raise RuntimeError("Aucune méthode HilbertCurve compatible trouvée.")

    @staticmethod
    def _build_spatial_order_from_xyz(xyz_t: torch.Tensor, method="hilbert", bits=15) -> torch.Tensor:
        xyz  = xyz_t.detach().cpu().numpy()
        mn, mx = xyz.min(0), xyz.max(0)
        xyz01   = (xyz - mn) / (mx - mn + 1e-8)
        codes   = (
            TTUltraMIGSModule5DColorSplit._hilbert_code(xyz01[:,0], xyz01[:,1], xyz01[:,2], bits=bits)
            if method == "hilbert"
            else TTUltraMIGSModule5DColorSplit._morton_code_10bit(xyz01[:,0], xyz01[:,1], xyz01[:,2])
        )
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
            back  = np.take(pts, range(1, shape[a]),   axis=a)
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
            G1   = G // n1
            lim2 = int(round(G1 ** 0.5)) + 2
            for n2 in range(n1, lim2 + 1):
                if G1 % n2:
                    continue
                n3 = G1 // n2
                if n2 > n3:
                    continue
                score = n3 - n1
                if best is None or score < best[0]:
                    best = (score, (n1, n2, n3))
        if best is None:
            n2 = max(1, int(np.sqrt(G)))
            n3 = G // n2
            return (1, min(n2, n3), max(n2, n3))
        return best[1]

    @staticmethod
    def _candidate_shapes(G: int) -> list:
        a, b, c = TTUltraMIGSModule5DColorSplit._balanced_shape_for(G)
        perms = {(a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)}
        return list(perms)

    @staticmethod
    def _pick_best_shape(order_t: torch.Tensor, xyz_t: torch.Tensor, candidates: list) -> tuple:
        order  = order_t.cpu().numpy()
        xyz    = xyz_t.detach().cpu().numpy()
        scored = [(sh, TTUltraMIGSModule5DColorSplit._adjacency_cost(order, xyz, sh))
                  for sh in candidates]
        scored.sort(key=lambda t: t[1])
        return scored[0][0], scored