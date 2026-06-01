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

tl.set_backend('pytorch')


class TTUltraMIGSModule5Dopacity(nn.Module):
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

        # Actual TT cores (filled in init_from_tensor)
        self.tt_tensor_gpu = nn.ParameterList()

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



    # --------------------- INITIALIZATION ---------------------


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

        # Spatial permutation (Hilbert order) for better TT locality
        perm = self._build_spatial_order_from_xyz(W_GM[:, :3], method="hilbert", bits=15)
        inv_perm = torch.empty_like(perm); inv_perm[perm] = torch.arange(G, device=perm.device)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)
        W_perm = W_GM[self.perm.to(W_GM.device)]  # (G, M)

        # Export snapshot after permutation
        with torch.no_grad():
            idx = self.perm.to(gaussian_model._xyz.device)
            xyz_perm       = gaussian_model._xyz[idx].detach().cpu().numpy()
            scaling_perm   = gaussian_model._scaling[idx].detach().cpu().numpy()
            rotation_perm  = gaussian_model._rotation[idx].detach().cpu().numpy()
            features_dc_p  = gaussian_model._features_dc[idx].detach().cpu().numpy()
            features_rest_p= gaussian_model._features_rest[idx].detach().cpu().numpy()
            opacity_perm   = gaussian_model._opacity[idx].detach().cpu().numpy()

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
        candidates = self._candidate_shapes(G)
        best_shape, scored = self._pick_best_shape(self.perm, W_GM[:, :3], candidates)
        if self.verbose:
            print(f"[TT] adjacency scores: {scored}")
            print(f"[TT] picked shape: {best_shape}")
        assert best_shape[0] * best_shape[1] * best_shape[2] == G
        n1, n2, n3 = best_shape

        # Export snapshot after reshape
        G_chk = xyz_perm.shape[0]
        assert n1 * n2 * n3 == G_chk, "n1*n2*n3 doit égaler G"
        Ig, Jg, Kg = np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij")
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

        # ========== TT UNIQUEMENT SUR OPACITY ==========
        
        # opacity permuté : (G, 1)
        opacity_perm_torch = W_perm[:, 42:43]

        # On sauvegarde les autres blocs (déjà permutés) comme BASE non factorizée
        with torch.no_grad():
            xyz_perm_direct      = W_perm[:, :3].detach()
            scaling_perm_direct  = W_perm[:, 3:6].detach()
            rotation_perm_direct = W_perm[:, 6:10].detach()
            dc_perm_direct       = W_perm[:, 10:11].detach()
            rest_perm_direct     = W_perm[:, 11:42].detach()

        self.xyz_param      = nn.Parameter(xyz_perm_direct.unsqueeze(0))       # (1, G, 3)
        self.scaling_param  = nn.Parameter(scaling_perm_direct.unsqueeze(0))   # (1, G, 3)
        self.rotation_param = nn.Parameter(rotation_perm_direct.unsqueeze(0))  # (1, G, 4)
        self.dc_param       = nn.Parameter(dc_perm_direct.unsqueeze(0))        # (1, G, 1)
        self.rest_param     = nn.Parameter(rest_perm_direct.unsqueeze(0))      # (1, G, 31)

        # Rangs cibles (comme avant, mais sur un dernier mode de taille 1)
        R = int(self.cfg.migs.get("init_rank", 64))
        ranks_target = [1, R, R, R, R, 1]

        # (I=1, n1, n2, n3, 1) : TT uniquement sur opacity
        self.tt_shape = (1, n1, n2, n3, 1)
        W_tt = opacity_perm_torch.unsqueeze(0).reshape(self.tt_shape)

        self.tt_rank = ranks_target
        tt_tensor = tensor_train(W_tt, rank=ranks_target, verbose=self.verbose)
        self.tt_tensor_gpu = nn.ParameterList(
            [nn.Parameter(c.to(W_tt.device)) for c in tt_tensor.factors]
        )
        self._expand_r1_by_replication(ranks_target[1])
        self._expand_ranks_to_targets_preserve(ranks_target)

        if self.verbose:
            print("TT core shapes (opacity only) :")
            for i, core in enumerate(self.tt_tensor_gpu):
                print(f"  core[{i}] -> {tuple(core.shape)}")

        # Si un optimizer existait déjà, on le reconstruit
        if self.optimizer is not None and self._needs_opt_rebuild:
            self._rebuild_optimizer_like_before()
            self._needs_opt_rebuild = False

        # Diagnostics
        W_rec = self.get_W_for_identity(0, original_order=False).to(W_GM.device)
        if self.verbose:
            print(f"[TT-opacity] recon shape: {tuple(W_rec.shape)}")
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
        Expand a shared rank by adding *learnable* channels:
        - nouveaux rangs initialisés avec un petit bruit gaussien
        - amplitude du bruit proportionnelle à la variance des tenseurs existants

        → la reconstruction initiale est à peine perturbée,
        mais les nouveaux rangs ont des gradients ≠ 0 et peuvent apprendre.
        """
        if add <= 0:
            return left, right

        dev = left.device

        # Shapes des nouveaux blocs
        dl_shape = list(left.shape);  dl_shape[dim_left]  = add
        dr_shape = list(right.shape); dr_shape[dim_right] = add

        # Échelle basée sur l'amplitude actuelle
        left_std  = left.detach().std()
        right_std = right.detach().std()

        # fallback si std quasi nulle
        if not torch.isfinite(left_std)  or left_std  < 1e-8:
            left_std  = left.detach().abs().mean()
        if not torch.isfinite(right_std) or right_std < 1e-8:
            right_std = right.detach().abs().mean()

        # très petit bruit, pour ne PAS casser la décomposition initiale
        scale = 1e-2  # tu peux tester 1e-3 si tu veux encore plus conservateur

        pad_left  = scale * left_std  * torch.randn(dl_shape, device=dev, dtype=left.dtype)
        pad_right = scale * right_std * torch.randn(dr_shape, device=dev, dtype=right.dtype)

        new_left  = torch.cat([left,  pad_left],  dim=dim_left)
        new_right = torch.cat([right, pad_right], dim=dim_right)
        return new_left, new_right


    @torch.no_grad()
    def _expand_ranks_to_targets_preserve(self, ranks_target):
        """
        Gonfle r2, r3, r4 en ajoutant des canaux avec un petit bruit gaussien.
        Fonctionne pour le cas TT-only-opacity :
            core0: (1, I,  r1)
            core1: (r1, n1, r2)
            core2: (r2, n2, r3)
            core3: (r3, n3, r4)
            core4: (r4, 1,  1)
        """

        # Raccourcis
        c0 = self.tt_tensor_gpu[0]
        c1 = self.tt_tensor_gpu[1]
        c2 = self.tt_tensor_gpu[2]
        c3 = self.tt_tensor_gpu[3]
        c4 = self.tt_tensor_gpu[4]

        # ---- r2: entre c1 (.., r2) et c2 (r2, ..) ----
        r2_cur = c1.shape[2]
        r2_tgt = int(ranks_target[2])
        if r2_tgt > r2_cur:
            add = r2_tgt - r2_cur
            new_c1, new_c2 = self._zero_pad_pair_preserve(
                c1, c2, add,
                dim_left=2,  # dernière dim de c1 = r2
                dim_right=0  # première dim de c2 = r2
            )
            self.tt_tensor_gpu[1] = nn.Parameter(new_c1)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)

        # refresh
        c1 = self.tt_tensor_gpu[1]
        c2 = self.tt_tensor_gpu[2]
        c3 = self.tt_tensor_gpu[3]
        c4 = self.tt_tensor_gpu[4]

        # ---- r3: entre c2 (.., r3) et c3 (r3, ..) ----
        r3_cur = c2.shape[2]
        r3_tgt = int(ranks_target[3])
        if r3_tgt > r3_cur:
            add = r3_tgt - r3_cur
            new_c2, new_c3 = self._zero_pad_pair_preserve(
                c2, c3, add,
                dim_left=2,  # dernière dim de c2 = r3
                dim_right=0  # première dim de c3 = r3
            )
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)
            self.tt_tensor_gpu[3] = nn.Parameter(new_c3)

        # refresh
        c2 = self.tt_tensor_gpu[2]
        c3 = self.tt_tensor_gpu[3]
        c4 = self.tt_tensor_gpu[4]

        # ---- r4: entre c3 (.., r4) et c4 (r4, ..) ----
        r4_cur = c3.shape[2]
        r4_tgt = int(ranks_target[4])
        if r4_tgt > r4_cur:
            add = r4_tgt - r4_cur
            new_c3, new_c4 = self._zero_pad_pair_preserve(
                c3, c4, add,
                dim_left=2,  # dernière dim de c3 = r4
                dim_right=0  # première dim de c4 = r4
            )
            self.tt_tensor_gpu[3] = nn.Parameter(new_c3)
            self.tt_tensor_gpu[4] = nn.Parameter(new_c4)

        self._needs_opt_rebuild = True


    @torch.no_grad()
    def _expand_r1_by_replication(self, r1_target: int):
        """
        Expand r1 by replicating core0 and core1.
        This is needed because TT-SVD with I=1 produces r1=1, but we want r1=64.
        Replication distributes the single identity's representation across more channels.
        """
        c0 = self.tt_tensor_gpu[0]  # (1, N, r1_cur)
        c1 = self.tt_tensor_gpu[1]  # (r1_cur, n1, r2_cur)
        
        r1_cur = c0.shape[2]
        if r1_cur >= r1_target:
            if self.verbose:
                print(f"[TT] r1 already ≥ target ({r1_cur} ≥ {r1_target})")
            return
        
        # Helper function for replication along a specific dimension
        def _repeat_to(x: torch.Tensor, dim: int, target: int) -> torch.Tensor:
            cur = x.shape[dim]
            if cur == target:
                return x
            times = math.ceil(target / cur)
            reps = [1] * x.ndim
            reps[dim] = times
            x_rep = x.repeat(*reps)
            
            # Slice to exact target size
            slices = [slice(None)] * x.ndim
            slices[dim] = slice(0, target)
            return x_rep[tuple(slices)]
        
        # Expand with scaling to preserve tensor magnitude
        # Scaling by r1_cur/r1_target ensures the contraction remains unchanged
        scale = r1_cur / float(r1_target)
        c0_new = _repeat_to(c0, dim=2, target=r1_target) * scale
        c1_new = _repeat_to(c1, dim=0, target=r1_target)
        
        self.tt_tensor_gpu[0] = nn.Parameter(c0_new)
        self.tt_tensor_gpu[1] = nn.Parameter(c1_new)
        self._needs_opt_rebuild = True
        
        if self.verbose:
            print(f"[TT] Expanded r1: {r1_cur} → {r1_target}")


    # --------------------- ORDERING / SHAPE HELPERS ---------------------

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
        codes = TTUltraMIGSModule5Dopacity._hilbert_code(xyz01[:, 0], xyz01[:, 1], xyz01[:, 2], bits=bits) \
                if method == "hilbert" else TTUltraMIGSModule5Dopacity._morton_code_10bit(xyz01[:, 0], xyz01[:, 1], xyz01[:, 2])
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
        a, b, c = TTUltraMIGSModule5Dopacity._balanced_shape_for(G)
        perms = {(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)}
        return list(perms)

    @staticmethod
    def _pick_best_shape(order_t: torch.Tensor, xyz_t: torch.Tensor, candidates: list) -> tuple:
        order = order_t.cpu().numpy()
        xyz = xyz_t.detach().cpu().numpy()
        scored = [(sh, TTUltraMIGSModule5Dopacity._adjacency_cost(order, xyz, sh)) for sh in candidates]
        scored.sort(key=lambda t: t[1])
        return scored[0][0], scored


    # --------------------- RECONSTRUCTION ---------------------

    def recombine_core4(self):
        """
        En mode 'opacity-only', le dernier coeur TT (index 4) correspond déjà
        au bloc opacity complet (taille 1 sur le dernier mode).
        """
        return self.tt_tensor_gpu[4]


    def get_core0(self, idx):
        assert 0 <= idx < self.tt_tensor_gpu[0].shape[1], f"Invalid identity index {idx}"
        return self.tt_tensor_gpu[0][:, idx:idx+1, :]  # (1, 1, r1)

    @torch.no_grad()
    def expand_first_core(self, n_identities):
        """Duplique l'axe identité de core0 + des blocs (xyz, scaling, rotation, dc, rest) sans bruit."""
        if not len(self.tt_tensor_gpu):
            raise RuntimeError("TT cores must be initialized before expansion.")

        core0 = self.tt_tensor_gpu[0]   # (1, N, r1)
        r0, n_cur, r1 = core0.shape
        if n_cur >= n_identities:
            if self.verbose:
                print(f"[TT] identity axis already ≥ {n_identities}")
            return

        # --- 1) Dupliquer core0 sans bruit ---
        base_core0 = core0[:, 0:1, :].detach()                  # (1, 1, r1)
        new_core0  = base_core0.repeat(1, n_identities, 1)      # (1, N_id, r1)
        self.tt_tensor_gpu[0] = nn.Parameter(new_core0)

        # --- 2) Helper pour les paramètres par identité ---
        def _expand_param(param):
            # param peut être (G,d) (au tout début) ou (N_id,G,d)
            if param.dim() == 2:
                param = param.unsqueeze(0)   # (1,G,d)
            n_id_cur, G, d = param.shape
            if n_id_cur >= n_identities:
                return param
            base = param[0:1].detach()                      # (1,G,d)
            new_param = base.repeat(n_identities, 1, 1)     # (N_id,G,d)
            return new_param

        # --- 3) Dupliquer xyz/scaling/rotation/dc/rest pour toutes les identités ---
        # Note: opacity n'est plus ici car il est dans la TT
        self.xyz_param      = nn.Parameter(_expand_param(self.xyz_param))
        self.scaling_param  = nn.Parameter(_expand_param(self.scaling_param))
        self.rotation_param = nn.Parameter(_expand_param(self.rotation_param))
        self.dc_param       = nn.Parameter(_expand_param(self.dc_param))
        self.rest_param     = nn.Parameter(_expand_param(self.rest_param))

        self._needs_opt_rebuild = True
        if self.optimizer is not None:
            self._rebuild_optimizer_like_before()
        if self.verbose:
            print(f"[TT] expanded first core + per-identity params → {self.tt_tensor_gpu[0].shape}, "
                f"n_id={n_identities}")


    def reconstruct(self):
        """Full dense reconstruction as a tensorly TT → dense."""
        full = list(self.tt_tensor_gpu[:4]) + [self.recombine_core4()]
        return tt_to_tensor(full)

    def get_tt_tensor(self, idx=None):
        """
        Retourne la liste des cœurs TT (cores).
        Si idx est donné, on sélectionne le core0 correspondant à l'identité idx.
        """
        core0 = self.get_core0(idx) if idx is not None else self.tt_tensor_gpu[0]
        cores = [
            core0,
            self.tt_tensor_gpu[1],
            self.tt_tensor_gpu[2],
            self.tt_tensor_gpu[3],
            self.recombine_core4()
        ]
        return cores


    def get_W_for_identity(self, idx: int, original_order: bool = True) -> torch.Tensor:
        """
        Retourne (G, M) pour l'identité idx.
        
        - opacity est reconstruit via la TT (en ordre permuté)
        - les autres blocs viennent des paramètres par identité
        """
        # opacity via TT
        try:
            opacity_perm = self._contract_tt_identity_gemm(idx)  # (G, 1)
        except AssertionError:
            T = tt_to_tensor(self.get_tt_tensor(idx))   # fallback
            M_last = T.shape[-1]
            full_perm = T.squeeze(0).contiguous().view(-1, M_last)
            opacity_perm = full_perm[:, :1]

        device = opacity_perm.device

        if hasattr(self, "xyz_param"):
            # Paramètres par-identité : (N_id, G, d) → (G, d)
            xyz_perm      = self.xyz_param[idx].to(device)
            scaling_perm  = self.scaling_param[idx].to(device)
            rotation_perm = self.rotation_param[idx].to(device)
            dc_perm       = self.dc_param[idx].to(device)
            rest_perm     = self.rest_param[idx].to(device)

        # Reconstruire dans l'ordre : xyz(3) + scaling(3) + rotation(4) + dc(1) + rest(31) + opacity(1)
        W_perm = torch.cat(
            [xyz_perm, scaling_perm, rotation_perm, dc_perm, rest_perm, opacity_perm],
            dim=1
        )  # (G, 43)

        # Remettre l'ordre original si demandé
        if original_order and hasattr(self, "inv_perm"):
            return W_perm[self.inv_perm.to(device)]
        return W_perm


    def _contract_tt_identity_gemm(self, idx: int) -> torch.Tensor:
        """Fast GEMM path for (I=1) contraction, returning opacity (G, 1)."""
        c0 = self.get_core0(idx)          # (1, 1, r1)
        c1 = self.tt_tensor_gpu[1]        # (r1, n1, r2)
        c2 = self.tt_tensor_gpu[2]        # (r2, n2, r3)
        c3 = self.tt_tensor_gpu[3]        # (r3, n3, r4)
        c4 = self.recombine_core4()       # (r4, 1, r5)

        r1, n1, r2 = c1.shape
        _,  n2, r3 = c2.shape
        _,  n3, r4 = c3.shape
        r4_c4, M, r5 = c4.shape
        assert c0.shape == (1, 1, r1), "core0 incompatible with c1"
        assert r4_c4 == r4, "core4 r4 mismatch"
        assert r5 == 1, "last TT rank must be 1 for this path"
        assert M == 1, f"Expected M=1 (opacity), got {M}"

        X = c0.reshape(1, r1) @ c1.reshape(r1, n1 * r2)
        X = X.reshape(n1, r2) @ c2.reshape(r2, n2 * r3)
        X = X.reshape(n1 * n2, r3) @ c3.reshape(r3, n3 * r4)
        X = X.reshape(n1 * n2 * n3, r4)          # (G, r4)
        C4 = c4.squeeze(-1).reshape(r4, M)       # (r4, 1)
        return X @ C4




    # --------------------- TRAINING CONTROL ---------------------

    def optimize_parameters(self):
        """Retourne tous les paramètres à optimiser."""
        params = list(self.tt_tensor_gpu)
        params += [
            self.xyz_param,
            self.scaling_param,
            self.rotation_param,
            self.dc_param,
            self.rest_param,
        ]
        return params



    def freeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = False


    def unfreeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = True


    def set_optimizer(self, opt_cfg):
        """Create optimizer and per-group LR scheduler for TT cores + per-identity params (opacity-only TT)."""

        self._opt_cfg = dict(opt_cfg) if opt_cfg is not None else {}

        # ---------- 1) TT cores ----------
        n_cores = len(self.tt_tensor_gpu)  # expected: 5 (core0..core4)

        tt_lrs         = list(self._opt_cfg.get("tt_lrs",       [1.6e-4] * n_cores))
        tt_final_lrs   = list(self._opt_cfg.get("tt_final_lrs", [1.6e-6] * n_cores))
        tt_decay_iters = int(self._opt_cfg.get("tt_decay_iters", 50000))

        # extend lists if needed
        if len(tt_lrs) < n_cores:
            tt_lrs += [tt_lrs[-1]] * (n_cores - len(tt_lrs))
        if len(tt_final_lrs) < n_cores:
            tt_final_lrs += [tt_final_lrs[-1]] * (n_cores - len(tt_final_lrs))

        # Fixed LR for opacity TT core (core4)
        opacity_lr = float(self._opt_cfg.get("opacity_lr", 5e-2))

        # ---------- 2) Per-identity params ----------
        xyz_lr       = float(self._opt_cfg.get("xyz_lr",        1.6e-4))
        xyz_final_lr = float(self._opt_cfg.get("xyz_final_lr",  1.6e-6))

        scaling_lr   = float(self._opt_cfg.get("scaling_lr",    5e-3))
        rotation_lr  = float(self._opt_cfg.get("rotation_lr",   1e-3))
        dc_lr        = float(self._opt_cfg.get("dc_lr",         2.5e-3))
        rest_lr      = float(self._opt_cfg.get("rest_lr",       2.5e-3))

        param_groups = []

        # --- TT cores groups ---
        for i in range(n_cores):
            if i == n_cores - 1:
                # core4 == opacity (FIXE)
                param_groups.append({
                    "params": [self.tt_tensor_gpu[i]],
                    "lr": opacity_lr,
                    "name": f"tt_core_{i}_opacity_fixed",
                })
            else:
                lr_init  = float(tt_lrs[i])
                lr_final = float(tt_final_lrs[i])
                param_groups.append({
                    "params": [self.tt_tensor_gpu[i]],
                    "lr": lr_init,
                    "initial_lr": lr_init,
                    "final_lr": lr_final,
                    "decay_iters": tt_decay_iters,
                    "name": f"tt_core_{i}_decay",
                })

        # --- xyz_param (DECAY) ---
        param_groups.append({
            "params": [self.xyz_param],
            "lr": xyz_lr,
            "initial_lr": xyz_lr,
            "final_lr": xyz_final_lr,
            "decay_iters": tt_decay_iters,
            "name": "xyz_param_decay",
        })

        # --- fixed per-identity blocks ---
        param_groups += [
            {"params": [self.scaling_param],  "lr": scaling_lr,  "name": "scaling_param_fixed"},
            {"params": [self.rotation_param], "lr": rotation_lr, "name": "rotation_param_fixed"},
            {"params": [self.dc_param],       "lr": dc_lr,       "name": "dc_param_fixed"},
            {"params": [self.rest_param],     "lr": rest_lr,     "name": "rest_param_fixed"},
        ]

        # ---------- 3) Optimizer ----------
        self.optimizer = torch.optim.Adam(param_groups)

        # ---------- 4) Scheduler (par-groupe; les fixes restent fixes) ----------
        lr_lambdas = []
        for g in param_groups:
            if "final_lr" not in g:
                # LR strictement fixe
                lr_lambdas.append(lambda step: 1.0)
            else:
                init_lr = float(g["initial_lr"])
                final_lr = float(g["final_lr"])
                decay_iters = int(g.get("decay_iters", tt_decay_iters))

                init_lr = max(init_lr, 1e-12)
                final_lr = max(final_lr, 1e-12)
                ratio = final_lr / init_lr

                def make_lambda(ratio=ratio, decay_iters=decay_iters):
                    def f(step):
                        t = min(max(step, 0), decay_iters) / float(max(decay_iters, 1))
                        return ratio ** t
                    return f

                lr_lambdas.append(make_lambda())

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambdas)

        self._needs_opt_rebuild = False




    def _rebuild_optimizer_like_before(self):
        self.set_optimizer(self._opt_cfg)

    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, iteration=None):
        # Finetune fast-path (core0 slice + optional texture MLP lives elsewhere)
        if hasattr(self, "_ft_opt") and (self._ft_opt is not None):
            self.ft_step()
            return

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
        
        if (iteration is None) or (iteration >= self.tt_delay):
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update learning rate if scheduler exists
            if self.scheduler is not None:
                self.scheduler.step()

    # --------------------- IDENTITY FINETUNE ---------------------

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
            eps     = self._randn_like(mu, tag="add_identity_noise").expand_as(mu)
            new_row = mu + noise_scale * sig * eps
            norms = U.view(n_id, r1).norm(dim=1)
            target_norm = norms.median()
            cur_norm = new_row.view(-1).norm()
            new_row = new_row / (cur_norm + 1e-8) * float(target_norm)
        else:
            new_row = self._randn_like(core0[:, :1, :], tag="add_identity_boot") * 0.02
        self.tt_tensor_gpu[0] = nn.Parameter(torch.cat([core0, new_row], dim=1))

        # --- Étendre les blocs par identité en ajoutant une copie de la première identité ---
        def _append_param(param):
            if param.dim() == 2:
                param = param.unsqueeze(0)      # (1,G,d)
            base = param[0:1].detach()          # (1,G,d)  (identité 0)
            new_param = torch.cat([param.detach(), base], dim=0)  # (N_id+1,G,d)
            return nn.Parameter(new_param)

        self.xyz_param      = _append_param(self.xyz_param)
        self.scaling_param  = _append_param(self.scaling_param)
        self.rotation_param = _append_param(self.rotation_param)
        self.dc_param       = _append_param(self.dc_param)
        self.rest_param     = _append_param(self.rest_param)

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