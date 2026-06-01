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


class TTUltraMIGSModule5DPerBlock(nn.Module):
    """
    Per-parameter Tensor-Train MIGS (no MARS):
    - Builds one TT decomposition per parameter group (xyz, scaling, rotation, dc, rest, opacity).
    - All groups share the same spatial permutation (Hilbert) and (n1, n2, n3) tiling.
    - Ranks are determined once from a global target ratio (using total M=43),
      then applied to all blocks (with rM clamped to each M_block).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        tt_cfg = cfg.migs if not isinstance(cfg, dict) else cfg["migs"]
        self._base_seed = int(getattr(cfg, "seed", 123))
        # self._per_block_target_ranks = {

        #     "xyz":     [1, 64, 64, 64, 64, 1],
        #     "rest":    [1, 64, 64, 64, 64, 1],


        #     "rotation":[1, 48, 48, 48, 48, 1],
        #     "scaling": [1, 32, 32, 32, 32, 1],


        #     "dc":      [1, 16, 16, 16, 16, 1],
        #     "opacity": [1,  8,  8,  8,  8,  1],
        # }

        self._per_block_target_ranks = {
            "xyz":     [1, 24, 24, 16,  8, 1],
            "rest":    [1, 48, 48, 32, 16, 1],
            "rotation":[1, 24, 24, 16,  8, 1],
            "scaling": [1, 16, 16, 12,  8, 1],
            "dc":      [1,  8,  8,  4,  2, 1],
            "opacity": [1,  8,  8,  4,  2, 1],
        }


        # Training delay (TT-only)
        self.tt_delay = tt_cfg.get("delay", 1000)
        if self.tt_delay is None:
            self.tt_delay = cfg.model.gaussian.get("delay", 0)

        # Working shape (set during init_from_tensor)
        self.tt_rank = tt_cfg.get("rank")   # will be set by suggest_tt_ranks_weighted
        self.tt_shape = tt_cfg.get("tt_shape")
        self.verbose = bool(tt_cfg.get("verbose", False))

        # Optimizer machinery
        self.optimizer = None
        self.scheduler = None
        self._opt_cfg = None
        self._needs_opt_rebuild = False
        self._tt_unfrozen = False

        # Per-block TT cores (each is a nn.ParameterList of 5 cores)
        self.tt_blocks = nn.ModuleDict()

        # Shared buffers: permutation and its inverse
        self.register_buffer("perm", torch.tensor([], dtype=torch.long))
        self.register_buffer("inv_perm", torch.tensor([], dtype=torch.long))

        # Save dir for exports
        self.save_dir = getattr(self.cfg, "hilbert_vis_5d", "./exports")
        os.makedirs(self.save_dir, exist_ok=True)

        # Keep block meta
        self.block_specs = [
            ("xyz",       3),
            ("scaling",   3),
            ("rotation",  4),
            ("dc",        1),
            ("rest",     31),
            ("opacity",   1),
        ]
        self._core0_mask_hooks = {}   # for identity finetune

    # --------------------- RNG helpers ---------------------

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

    # --------------------- INITIALIZATION ---------------------

    def init_from_tensor(self, gaussian_model):
        """Build per-block TT from the current Gaussian parameters; initialize trainable cores."""
        device = gaussian_model._xyz.device
        G = gaussian_model._xyz.shape[0]
        if self.verbose:
            print("****************** G =", G)

        # Gather raw blocks (G, M_b)
        xyz           = gaussian_model._xyz
        scaling       = gaussian_model._scaling
        rotation      = gaussian_model._rotation
        features_dc   = gaussian_model._features_dc.squeeze(-1)
        features_rest = gaussian_model._features_rest.squeeze(-1)
        opacity       = gaussian_model._opacity

        def print_param_stats(name, tensor):
            t = tensor.detach().cpu()
            print(f"{name:12s} shape={tuple(t.shape)} | min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}")

        if self.verbose:
            print_param_stats("xyz", xyz)
            print_param_stats("scaling", scaling)
            print_param_stats("rotation", rotation)
            print_param_stats("features_dc", features_dc)
            print_param_stats("features_rest", features_rest)
            print_param_stats("opacity", opacity)

        # Compose full (G, M) ONLY for permutation + diagnostics
        all_params = [xyz, scaling, rotation, features_dc, features_rest, opacity]
        W_GM = torch.cat(
            [p if p.ndim == 2 else p.view(p.shape[0], -1) for p in all_params],
            dim=1
        )
        M_total = int(W_GM.shape[1])

        # Spatial permutation via Hilbert (based on xyz)
        perm = self._build_spatial_order_from_xyz(xyz, method="hilbert", bits=15)
        inv_perm = torch.empty_like(perm); inv_perm[perm] = torch.arange(G, device=perm.device)
        self.register_buffer("perm", perm.to(device))
        self.register_buffer("inv_perm", inv_perm.to(device))
        W_perm = W_GM[self.perm]  # (G, M_total)

        # -------- snapshot after perm (blockwise arrays for convenience) --------
        with torch.no_grad():
            idx = self.perm
            xyz_perm        = xyz[idx].detach().cpu().numpy()
            scaling_perm    = scaling[idx].detach().cpu().numpy()
            rotation_perm   = rotation[idx].detach().cpu().numpy()
            features_dc_p   = features_dc[idx].detach().cpu().numpy()
            features_rest_p = features_rest[idx].detach().cpu().numpy()
            opacity_perm    = opacity[idx].detach().cpu().numpy()

            use_sh = bool(getattr(gaussian_model, "use_sh", False))
            sh_deg = int(getattr(gaussian_model, "max_sh_degree", 0)) if use_sh else 0

            np.savez_compressed(
                os.path.join(self.save_dir, "snapshot_after_perm_full.npz"),
                xyz=xyz_perm, scaling=scaling_perm, rotation=rotation_perm,
                features_dc=features_dc_p, features_rest=features_rest_p, opacity=opacity_perm,
                perm=self.perm.detach().cpu().numpy(),
                use_sh=np.array([use_sh], dtype=np.bool_), sh_deg=np.array([sh_deg], dtype=np.int64)
            )
            if self.verbose:
                print("[EXPORT] snapshot_after_perm_full.npz")

        # Choose balanced (n1,n2,n3) tiling of G (sharing across blocks)
        candidates = self._candidate_shapes(G)
        best_shape, scored = self._pick_best_shape(self.perm, xyz, candidates)
        if self.verbose:
            print(f"[TT] adjacency scores: {scored}")
            print(f"[TT] picked shape: {best_shape}")
        assert best_shape[0] * best_shape[1] * best_shape[2] == G
        n1, n2, n3 = best_shape

        # snapshot after reshape (grid ijk)
        G_chk = xyz_perm.shape[0]
        assert n1 * n2 * n3 == G_chk, "n1*n2*n3 must equal G"
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
        if self.verbose:
            print("[EXPORT] snapshot_after_reshape_full.npz")

        # Final TT config target ranks computed ONCE globally (M_total)
        migs_cfg = self.cfg.migs if not isinstance(self.cfg, dict) else self.cfg["migs"]

        # # Build per-block TT using same ranks (with rM clamped per block)
        # self.tt_blocks = nn.ModuleDict()
        # block_inputs = {
        #     "xyz":      xyz,
        #     "scaling":  scaling,
        #     "rotation": rotation,
        #     "dc":       features_dc,
        #     "rest":     features_rest,
        #     "opacity":  opacity,
        # }

        # for name, Mb in self.block_specs:
        #     B = block_inputs[name]                                  # (G, Mb)
        #     Bp = B[self.perm]                                       # permute rows
        #     W_tt = Bp.unsqueeze(0).reshape(1, n1, n2, n3, Mb)

        #     # (A) Compute exact caps that TL will accept for this shape
        #     caps = self.tt_rank_caps_like_tensorly((1, n1, n2, n3, Mb))   # [r1_max, r2_max, r3_max, r4_max]

        #     # (B) Call TensorLy with those caps (fast; no extra surprises/capping)
        #     tt_tensor = tensor_train(W_tt, rank=caps, verbose=self.verbose)
        #     cores = [nn.Parameter(c.to(W_tt.device)) for c in tt_tensor.factors]

        #     # (C) Build your uniform scaffold target: [1, R, R, R, R, 1]
        #     R = int(self.cfg.migs.get("init_rank", 64))
        #     ranks_block_target = [1, R, R, R, R, 1]
        #     self._per_block_target_ranks[name] = ranks_block_target

        #     # (D) Expand r1..r4 up to R by zero-padding (preserve reconstruction)
        #     cores = self._expand_ranks_to_targets_preserve_block(cores, ranks_block_target)

        #     # (E) (Optional but recommended) add tiny noise only to the newly added zeros so they can learn
        #     with torch.no_grad():
        #         eps = 1e-4
        #         for p in cores:
        #             z = (p.data == 0)
        #             if z.any():
        #                 p.data[z] = eps * torch.randn_like(p.data[z])

        #     self.tt_blocks[name] = nn.ParameterList(cores)


        # # Persist global record
        # self.tt_shape = (1, n1, n2, n3, int(M_total))
        # self.tt_rank  = ranks_block_target

        # Build per-block TT using your custom per-block target ranks
        self.tt_blocks = nn.ModuleDict()
        block_inputs = {
            "xyz":      xyz,
            "scaling":  scaling,
            "rotation": rotation,
            "dc":       features_dc,
            "rest":     features_rest,
            "opacity":  opacity,
        }

        for name, Mb in self.block_specs:
            B = block_inputs[name]                      # (G, Mb)
            Bp = B[self.perm]                           # permute rows
            W_tt = Bp.unsqueeze(0).reshape(1, n1, n2, n3, Mb)

            # (A) Rangs cibles pour CE bloc (définis dans __init__)
            ranks_block_target = self._per_block_target_ranks[name]   # [r0,r1,r2,r3,r4,r5]

            # (B) TT-SVD avec ces rangs (TensorLy va les caper si nécessaire)
            tt_tensor = tensor_train(W_tt, rank=ranks_block_target, verbose=self.verbose)
            cores = [nn.Parameter(c.to(W_tt.device)) for c in tt_tensor.factors]
            # cores = [G0(1,I,r1), G1(r1,n1,r2), G2(r2,n2,r3), G3(r3,n3,r4), G4(r4,Mb,r5)]

            # (C) Étendre r1 par réplication (comme ta version globale)
            cores = self._expand_r1_by_replication_block(cores, r1_target=ranks_block_target[1])

            # (D) Étendre r2, r3, r4 par zero-pad + petit bruit (appris)
            cores = self._expand_ranks_to_targets_preserve_block(cores, ranks_block_target)

            self.tt_blocks[name] = nn.ParameterList(cores)

            if self.verbose:
                print("=== Final TT ranks per block ===")
                for name, cores in self.tt_blocks.items():
                    shapes = [tuple(c.shape) for c in cores]
                    # ranks = [r1,r2,r3,r4]
                    r1 = cores[0].shape[2]
                    r2 = cores[1].shape[2]
                    r3 = cores[2].shape[2]
                    r4 = cores[3].shape[2]
                    print(f"{name}: core shapes = {shapes} | ranks = [{r1}, {r2}, {r3}, {r4}]")


        # Persist global record (facultatif, juste pour debug / logs)
        self.tt_shape = (1, n1, n2, n3, int(M_total))
        self.tt_rank  = self._per_block_target_ranks["xyz"]



        # Optimizer rebuild if already created earlier
        if self.optimizer is not None and self._needs_opt_rebuild:
            self._rebuild_optimizer_like_before()
            self._needs_opt_rebuild = False

        # Diagnostics in original order: reconstruct per-block, concat, compare
        W_rec_perm = self.get_W_for_identity(0, original_order=False).to(W_GM.device)
        # if you want original order, uncomment:
        # W_rec = W_rec_perm[self.inv_perm]
        W_rec = W_rec_perm  # comparison in permuted order is fine if consistent for both

        if self.verbose:
            print(f"[TT] recon shape: {tuple(W_rec.shape)}")

        compare_reconstruction_per_block(
            W_perm, W_rec, split_sizes=[3, 3, 4, 1, 31, 1],
            names=['xyz', 'scaling', 'rotation', 'dc', 'rest', 'opacity']
        )
        plot_correlation_across_parameters(W_perm, W_rec)
        plot_pca_groupwise_xyz_auto(W_perm, W_rec, num_groups=10)

    
    # @staticmethod
    # def tt_rank_caps_like_tensorly(shape_5d):
    #     # shape_5d = (I, n1, n2, n3, Mb)
    #     I, n1, n2, n3, Mb = map(int, shape_5d)
    #     r0 = 1
    #     r1 = min(r0 * I, n1 * n2 * n3 * Mb)  # with I=1 => r1 = 1
    #     r2 = min(r1 * n1, n2 * n3 * Mb)
    #     r3 = min(r2 * n2, n3 * Mb)
    #     r4 = min(r3 * n3, Mb)                  # last internal rank ≤ Mb
    #     return [1,r1, r2, r3, r4,1]                       # internal ranks only


    # ---------- ZERO-PAD EXPANSION HELPERS (preserve tensor exactly) ----------

    # @torch.no_grad()
    # def _zero_pad_pair_preserve(self, left: torch.Tensor, right: torch.Tensor,
    #                             add: int, dim_left: int, dim_right: int):
    #     """
    #     Expand a shared rank by concatenating zeros on BOTH sides:
    #       - left is zero-padded along its 'outgoing' (last) rank axis,
    #       - right is zero-padded along its 'incoming' (first) rank axis.
    #     """
    #     if add <= 0:
    #         return left, right
    #     dev = left.device
    #     dl_shape = list(left.shape);  dl_shape[dim_left]  = add
    #     dr_shape = list(right.shape); dr_shape[dim_right] = add
    #     pad_left  = torch.zeros(dl_shape,  device=dev, dtype=left.dtype)
    #     pad_right = torch.zeros(dr_shape, device=dev, dtype=right.dtype)
    #     new_left  = torch.cat([left,  pad_left],  dim=dim_left)
    #     new_right = torch.cat([right, pad_right], dim=dim_right)
    #     return new_left, new_right

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
        scale = 1e-2  # 1e-3 si tu veux encore plus conservateur

        pad_left  = scale * left_std  * torch.randn(dl_shape, device=dev, dtype=left.dtype)
        pad_right = scale * right_std * torch.randn(dr_shape, device=dev, dtype=right.dtype)

        new_left  = torch.cat([left,  pad_left],  dim=dim_left)
        new_right = torch.cat([right, pad_right], dim=dim_right)
        return new_left, new_right


    # @torch.no_grad()
    # def _expand_ranks_to_targets_preserve_block(self, cores, ranks_target):
    #     """
    #     Expand ranks (r1..r4 and rM) for a single block's core list (5 cores):
    #       cores: [G0 (1,I,r1), G1 (r1,n1,r2), G2 (r2,n2,r3), G3 (r3,n3,r4), G4 (r4,M,1)]
    #     Only expands when current < target. Never shrinks.
    #     """
    #     # Core shapes
    #     c0, c1, c2, c3, c4 = cores

    #     # r1 between c0 (.., r1) and c1 (r1, ..) — for I=1 init we may want to reach target r1
    #     r1_cur = c0.shape[2]
    #     r1_tgt = int(ranks_target[1])
    #     if r1_tgt > r1_cur:
    #         add = r1_tgt - r1_cur
    #         # pad c0 on its last dim, c1 on its first dim
    #         new_c0, new_c1 = self._zero_pad_pair_preserve(c0, c1, add, dim_left=2, dim_right=0)
    #         cores[0], cores[1] = nn.Parameter(new_c0), nn.Parameter(new_c1)
    #         c0, c1 = cores[0], cores[1]

    #     # r2 between c1 and c2
    #     r2_cur = c1.shape[2]
    #     r2_tgt = int(ranks_target[2])
    #     if r2_tgt > r2_cur:
    #         add = r2_tgt - r2_cur
    #         new_c1, new_c2 = self._zero_pad_pair_preserve(c1, c2, add, dim_left=2, dim_right=0)
    #         cores[1], cores[2] = nn.Parameter(new_c1), nn.Parameter(new_c2)
    #         c1, c2 = cores[1], cores[2]

    #     # r3 between c2 and c3
    #     r3_cur = c2.shape[2]
    #     r3_tgt = int(ranks_target[3])
    #     if r3_tgt > r3_cur:
    #         add = r3_tgt - r3_cur
    #         new_c2, new_c3 = self._zero_pad_pair_preserve(c2, c3, add, dim_left=2, dim_right=0)
    #         cores[2], cores[3] = nn.Parameter(new_c2), nn.Parameter(new_c3)
    #         c2, c3 = cores[2], cores[3]

    #     # r4 between c3 and c4
    #     r4_cur = c3.shape[2]
    #     r4_tgt = int(ranks_target[4])
    #     if r4_tgt > r4_cur:
    #         add = r4_tgt - r4_cur
    #         new_c3, new_c4 = self._zero_pad_pair_preserve(c3, c4, add, dim_left=2, dim_right=0)
    #         cores[3], cores[4] = nn.Parameter(new_c3), nn.Parameter(new_c4)
    #         c3, c4 = cores[3], cores[4]

    #     # rM (last internal rank on M mode) = ranks_target[-2]
    #     rM_cur = c4.shape[2]
    #     rM_tgt = int(ranks_target[-2])  # already clamped to Mb at construction
    #     if rM_tgt > rM_cur:
    #         add = rM_tgt - rM_cur
    #         # pad c4 on last dim only (no paired core)
    #         pad = torch.zeros((c4.shape[0], c4.shape[1], add), device=c4.device, dtype=c4.dtype)
    #         c4_new = torch.cat([c4, pad], dim=2)
    #         cores[4] = nn.Parameter(c4_new)

    #     return cores

    @torch.no_grad()
    def _expand_ranks_to_targets_preserve_block(self, cores, ranks_target):
        """
        Expand ranks (r1..r4) pour un bloc:
          cores: [G0 (1,I,r1), G1 (r1,n1,r2), G2 (r2,n2,r3),
                  G3 (r3,n3,r4), G4 (r4,M, r5=1)]
        On n’essaie PAS de modifier r5 (on le laisse à 1).
        """
        c0, c1, c2, c3, c4 = cores

        # ---- r2: entre c1 (.., r2) et c2 (r2, ..) ----
        r2_cur = c1.shape[2]
        r2_tgt = int(ranks_target[2])
        if r2_tgt > r2_cur:
            add = r2_tgt - r2_cur
            new_c1, new_c2 = self._zero_pad_pair_preserve(c1, c2, add, dim_left=2, dim_right=0)
            cores[1] = nn.Parameter(new_c1)
            cores[2] = nn.Parameter(new_c2)
            c1, c2 = cores[1], cores[2]

        # ---- r3: entre c2 (.., r3) et c3 (r3, ..) ----
        r3_cur = c2.shape[2]
        r3_tgt = int(ranks_target[3])
        if r3_tgt > r3_cur:
            add = r3_tgt - r3_cur
            new_c2, new_c3 = self._zero_pad_pair_preserve(c2, c3, add, dim_left=2, dim_right=0)
            cores[2] = nn.Parameter(new_c2)
            cores[3] = nn.Parameter(new_c3)
            c2, c3 = cores[2], cores[3]

        # ---- r4: entre c3 (.., r4) et c4 (r4, ..) ----
        r4_cur = c3.shape[2]
        r4_tgt = int(ranks_target[4])
        if r4_tgt > r4_cur:
            add = r4_tgt - r4_cur
            new_c3, new_c4 = self._zero_pad_pair_preserve(c3, c4, add, dim_left=2, dim_right=0)
            cores[3] = nn.Parameter(new_c3)
            cores[4] = nn.Parameter(new_c4)

        self._needs_opt_rebuild = True
        return cores

    @torch.no_grad()
    def _expand_r1_by_replication_block(self, cores, r1_target: int):
        """
        Version par bloc de _expand_r1_by_replication :
        - cores: [c0, c1, c2, c3, c4]
        - on étend r1 (dim 2 de c0, dim 0 de c1) par réplication + scaling,
          pour préserver à peu près la même magnitude.
        """
        c0, c1, c2, c3, c4 = cores
        r1_cur = c0.shape[2]
        if r1_cur >= r1_target:
            return cores

        def _repeat_to(x: torch.Tensor, dim: int, target: int) -> torch.Tensor:
            cur = x.shape[dim]
            if cur == target:
                return x
            times = math.ceil(target / cur)
            reps = [1] * x.ndim
            reps[dim] = times
            x_rep = x.repeat(*reps)
            slices = [slice(None)] * x.ndim
            slices[dim] = slice(0, target)
            return x_rep[tuple(slices)]

        # Réplication + scaling pour garder la magnitude stable
        scale = r1_cur / float(r1_target)
        c0_new = _repeat_to(c0, dim=2, target=r1_target) * scale
        c1_new = _repeat_to(c1, dim=0, target=r1_target)

        cores[0] = nn.Parameter(c0_new)
        cores[1] = nn.Parameter(c1_new)
        self._needs_opt_rebuild = True
        if self.verbose:
            print(f"[TT-PerBlock] Expanded r1: {r1_cur} → {r1_target}")
        return cores



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
        codes = TTUltraMIGSModule5DPerBlock._hilbert_code(
            xyz01[:, 0], xyz01[:, 1], xyz01[:, 2], bits=bits
        ) if method == "hilbert" else TTUltraMIGSModule5DPerBlock._morton_code_10bit(
            xyz01[:, 0], xyz01[:, 1], xyz01[:, 2]
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
        a, b, c = TTUltraMIGSModule5DPerBlock._balanced_shape_for(G)
        perms = {(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)}
        return list(perms)

    @staticmethod
    def _pick_best_shape(order_t: torch.Tensor, xyz_t: torch.Tensor, candidates: list) -> tuple:
        order = order_t.cpu().numpy()
        xyz = xyz_t.detach().cpu().numpy()
        scored = [(sh, TTUltraMIGSModule5DPerBlock._adjacency_cost(order, xyz, sh)) for sh in candidates]
        scored.sort(key=lambda t: t[1])
        return scored[0][0], scored

    # --------------------- RECONSTRUCTION ---------------------

    def _contract_tt_identity_gemm_block(self, cores, idx_identity: int) -> torch.Tensor:
        """
        Fast GEMM path for (I, n1, n2, n3, M_b) with I>=1 but core0 indexed to one identity.
        cores: list of 5 cores for a single block.
        Returns (G, M_b) in permuted order.
        """
        c0, c1, c2, c3, c4 = cores
        # select the identity slice from core0 → (1,1,r1)
        assert c0.shape[1] >= 1, "core0 must have identity axis"
        assert 0 <= idx_identity < c0.shape[1], f"Invalid identity index {idx_identity}"
        core0 = c0[:, idx_identity:idx_identity+1, :]  # (1,1,r1)

        r1, n1, r2 = c1.shape
        _,  n2, r3 = c2.shape
        _,  n3, r4 = c3.shape
        r4_c4, Mb, rM = c4.shape
        assert core0.shape == (1, 1, r1), "core0 incompatible with c1"
        assert r4_c4 == r4, "last r4 mismatch"

        X = core0.reshape(1, r1) @ c1.reshape(r1, n1 * r2)
        X = X.reshape(n1, r2)    @ c2.reshape(r2, n2 * r3)
        X = X.reshape(n1 * n2, r3) @ c3.reshape(r3, n3 * r4)
        X = X.reshape(n1 * n2 * n3, r4)        # (G, r4)
        C4 = c4.reshape(r4, Mb * rM)           # (r4, Mb*rM)
        Y = X @ C4                              # (G, Mb*rM)
        return Y.reshape(-1, Mb, rM).sum(dim=2) # collapse last rank (rM) by sum (equivalent to rM=1 when padded zeros)

    def reconstruct_block(self, block_name: str, idx_identity: int = 0, original_order: bool = False) -> torch.Tensor:
        """Reconstruct a parameter block (G, M_block) for identity idx."""
        assert block_name in self.tt_blocks, f"Unknown block {block_name}"
        cores = [p for p in self.tt_blocks[block_name]]
        W_perm = self._contract_tt_identity_gemm_block(cores, idx_identity)
        if original_order:
            return W_perm[self.inv_perm]
        return W_perm

    def reconstruct_all(self, idx_identity: int = 0, original_order: bool = False) -> torch.Tensor:
        """Reconstruct all parameter blocks and concatenate into (G, M_total)."""
        mats = [self.reconstruct_block(name, idx_identity, original_order=False) for name, _ in self.block_specs]
        W_perm = torch.cat(mats, dim=1)
        if original_order:
            return W_perm[self.inv_perm]
        return W_perm

    def get_W_for_identity(self, idx: int, original_order: bool = True) -> torch.Tensor:
        """Alias to reconstruct_all (kept for compatibility with your downstream)."""
        return self.reconstruct_all(idx_identity=idx, original_order=original_order)

    # --------------------- IDENTITY / CORE0 MANAGEMENT ---------------------

    @torch.no_grad()
    def _repeat_to(self, x: torch.Tensor, dim: int, target: int) -> torch.Tensor:
        cur = x.shape[dim]
        if cur == target:
            return x
        times = math.ceil(target / cur)
        reps = [1] * x.dim(); reps[dim] = times
        x_rep = x.repeat(*reps)
        sl = [slice(None)] * x.dim(); sl[dim] = slice(0, target)
        return x_rep[tuple(sl)]

    @torch.no_grad()
    def expand_first_core(self, n_identities: int):
        """Duplicate identity axis (core0) in all blocks to the target size."""
        for name in self.tt_blocks:
            cores = self.tt_blocks[name]
            c0 = cores[0]  # (1, N_id, r1)
            r0, n_cur, r1 = c0.shape
            if n_cur >= n_identities:
                continue
            base = c0[:, 0:1, :].detach()
            rep   = base.repeat(1, n_identities, 1)
            noise = self._randn_like(rep, tag=f"core0_expand_noise_{name}") * 1e-3
            new   = rep + noise
            self.tt_blocks[name][0] = nn.Parameter(new)
        self._needs_opt_rebuild = True
        if self.optimizer is not None:
            self._rebuild_optimizer_like_before()
        if self.verbose:
            print(f"[TT] expanded first core across blocks → N_id={n_identities}")

    @torch.no_grad()
    def add_identity(self, noise_scale: float = 0.05, rebuild_optimizer: bool = True) -> int:
        """
        Append one identity to core0 in all blocks using a neutral init:
        mean + noise_scale * std * N(0,1), rescaled to the median row norm (per block).
        Returns the index of the newly added identity.
        """
        # assume all blocks share same N_id
        any_block = next(iter(self.tt_blocks.values()))
        r0, n_id, r1 = any_block[0].shape
        new_idx = n_id  # next slot

        for name, cores in self.tt_blocks.items():
            c0 = cores[0]  # (1, N, r1)
            if n_id > 0:
                U = c0.detach()
                mu  = U.mean(dim=1, keepdim=True)
                sig = U.std(dim=1, unbiased=False, keepdim=True).clamp_(min=1e-8)
                eps     = self._randn_like(mu, tag=f"add_identity_noise_{name}").expand_as(mu)
                new_row = mu + noise_scale * sig * eps
                norms = U.view(n_id, r1).norm(dim=1)
                target_norm = norms.median()
                cur_norm = new_row.view(-1).norm()
                new_row = new_row / (cur_norm + 1e-8) * float(target_norm)
            else:
                new_row = self._randn_like(c0[:, :1, :], tag=f"add_identity_boot_{name}") * 0.02
            self.tt_blocks[name][0] = nn.Parameter(torch.cat([c0, new_row], dim=1))

        self._needs_opt_rebuild = True
        if rebuild_optimizer and (self.optimizer is not None):
            self._rebuild_optimizer_like_before()

        if self.verbose:
            print(f"[TT] added identity → new index {new_idx}")
        return new_idx

    def enable_identity_finetune(self, idx: int, color_mlp: nn.Module = None,
                                 lr_id: float = 3e-3, lr_tex: float = 1e-3,
                                 include_color_in_ft_opt: bool = False):
        """
        Finetune only the identity slice `idx` of core0 across ALL blocks (mask gradients);
        optionally include a color MLP (kept for compatibility; pass None to ignore).
        """
        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

        # Enable only core0 across blocks
        self._core0_mask_hooks = {}
        params = []
        for name, cores in self.tt_blocks.items():
            c0 = cores[0]
            c0.requires_grad = True
            mask = torch.zeros_like(c0); mask[:, idx:idx+1, :] = 1.0
            if name in self._core0_mask_hooks and self._core0_mask_hooks[name] is not None:
                self._core0_mask_hooks[name].remove()
            self._core0_mask_hooks[name] = c0.register_hook(lambda g, m=mask: g * m)
            params.append({"params": [c0], "lr": lr_id})

        if color_mlp is not None:
            for p in color_mlp.parameters():
                p.requires_grad = include_color_in_ft_opt
            if include_color_in_ft_opt:
                params.append({"params": list(color_mlp.parameters()), "lr": lr_tex})

        self._ft_opt = torch.optim.Adam(params)

    def ft_step(self):
        if hasattr(self, "_ft_opt") and (self._ft_opt is not None):
            self._ft_opt.step()
            self._ft_opt.zero_grad()

    def disable_identity_finetune(self):
        for name, hook in self._core0_mask_hooks.items():
            if hook is not None:
                hook.remove()
        self._core0_mask_hooks = {}
        self._ft_opt = None
        for p in self.parameters():
            p.requires_grad = True

    # --------------------- TRAINING CONTROL / OPTIMIZER ---------------------

    def optimize_parameters(self):
        """Return all TT parameters from all blocks."""
        params = []
        for name, plist in self.tt_blocks.items():
            params += list(plist)
        return params

    def freeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = False

    def unfreeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = True

    def set_optimizer(self, opt_cfg):
        """
        Create optimizer and LR scheduler for TT cores in all blocks.
        - xyz:      all cores 0..4 decayed 1.6e-4 → 1.6e-6
        - scaling:  cores 0..3 decayed 1.6e-4 → 1.6e-6, core4 = 5e-3 (fixed)
        - rotation: cores 0..3 decayed 1.6e-4 → 1.6e-6, core4 = 1e-3 (fixed)
        - dc:       cores 0..3 decayed 1.6e-4 → 1.6e-6, core4 = 2.5e-3 (fixed)
        - rest:     cores 0..3 decayed 1.6e-4 → 1.6e-6, core4 = 2.5e-3 (fixed)
        - opacity:  cores 0..3 decayed 1.6e-4 → 1.6e-6, core4 = 5e-2 (fixed)
        """
        self._opt_cfg = dict(opt_cfg) if opt_cfg is not None else {}
        tt_decay_iters = self._opt_cfg.get("tt_decay_iters", 50000)

        block_lr_cfg = {
            "xyz":     {"decay": (1.6e-4, 1.6e-6), "core4": None},
            "scaling": {"decay": (1.6e-4, 1.6e-6), "core4": 5e-3},
            "rotation":{"decay": (1.6e-4, 1.6e-6), "core4": 1e-3},
            "dc":      {"decay": (1.6e-4, 1.6e-6), "core4": 2.5e-3},
            "rest":    {"decay": (1.6e-4, 1.6e-6), "core4": 2.5e-3},
            "opacity": {"decay": (1.6e-4, 1.6e-6), "core4": 5e-2},
        }

        param_groups = []
        lr_lambdas = []

        for name, cores in self.tt_blocks.items():
            cfg = block_lr_cfg[name]
            lr_init, lr_final = cfg["decay"]

            # gamma for this block’s decay
            gamma = (lr_final / lr_init) ** (1.0 / max(tt_decay_iters, 1))

            for i, core in enumerate(cores):
                if i < 4 or (i == 4 and cfg["core4"] is None):  # decayed group
                    group = {
                        "params": [core],
                        "lr": lr_init,
                        "initial_lr": lr_init,
                        "final_lr": lr_final,
                    }
                    param_groups.append(group)
                    lr_lambdas.append(lambda step, g=gamma: g**step)

                elif i == 4:  # special fixed core
                    group = {"params": [core], "lr": cfg["core4"]}
                    param_groups.append(group)
                    lr_lambdas.append(lambda step: 1.0)  # fixed

        self.optimizer = torch.optim.Adam(param_groups)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambdas)

        self._needs_opt_rebuild = False



    def _rebuild_optimizer_like_before(self):
        self.set_optimizer(self._opt_cfg)

    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, iteration=None):
        # Identity finetune fast path
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

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_learning_rate()
