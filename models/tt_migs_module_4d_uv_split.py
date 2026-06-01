import os
import math
import hashlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor

from omegaconf import ListConfig

tl.set_backend("pytorch")


class TTUltraMIGSModule4DUVGrid(nn.Module):
    """
    UV TT with TWO independent branches:
      - REST TT    : (I, Nu, Nv, 39)
          [scaling(3), rotation(4), dc(1), rest(31)]
      - OPACITY TT : (I, Nu, Nv, 1)
          [opacity(1)]

    Final sampled W:
      W = [scaling(3), rotation(4), dc(1), rest(31), opacity(1)] => 40

    Key property:
      TV loss on OPACITY branch → gradient goes ONLY into tt_opacity_gpu cores
      REST branch is NEVER contaminated by TV loss → no ring artifacts
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        tt_cfg = cfg.migs if not isinstance(cfg, dict) else cfg["migs"]

        self._base_seed = int(getattr(cfg, "seed", 123))

        self.tt_delay = tt_cfg.get("delay", 1000)
        if self.tt_delay is None:
            self.tt_delay = cfg.model.gaussian.get("delay", 0)

        self.verbose = bool(tt_cfg.get("verbose", False))
        self.tt_rank_rest = None
        self.tt_rank_opacity = None

        # Optim
        self.optimizer = None
        self.scheduler = None
        self._opt_cfg = None
        self._needs_opt_rebuild = False
        self._tt_unfrozen = False

        # Eval caches
        self._grid_cache_rest = {}
        self._grid_cache_opacity = {}

        # -------------------------
        # REST TT : (I,Nu,Nv,39)
        # -------------------------
        self.tt_rest_gpu = nn.ParameterList()

        self.core4_scaling  = nn.Parameter(torch.zeros(1, 3,  1))
        self.core4_rotation = nn.Parameter(torch.zeros(1, 4,  1))
        self.core4_dc       = nn.Parameter(torch.zeros(1, 1,  1))
        self.core4_rest     = nn.Parameter(torch.zeros(1, 31, 1))

        # -------------------------
        # OPACITY TT : (I,Nu,Nv,1)
        # -------------------------
        self.tt_opacity_gpu = nn.ParameterList()
        self.core4_opacity  = nn.Parameter(torch.zeros(1, 1,  1))

        # Backward-compat alias (external code that expects tt_tensor_gpu)
        self.tt_tensor_gpu = self.tt_rest_gpu

        self.save_dir = getattr(self.cfg, "exports_dir", "./exports")
        os.makedirs(self.save_dir, exist_ok=True)

    # =========================================================
    # RNG helpers
    # =========================================================

    def _stream(self, tag: str, device) -> torch.Generator:
        h = int.from_bytes(hashlib.md5(tag.encode("utf8")).digest()[:8], "little")
        g = torch.Generator(device=device)
        g.manual_seed(self._base_seed ^ h)
        return g

    def _randn_like(self, ref, tag: str):
        g = self._stream(tag, ref.device)
        return torch.randn(ref.shape, device=ref.device, dtype=ref.dtype, generator=g)

    # =========================================================
    # Rank normalization helper
    # =========================================================

    def _to_tt_ranks_4d(self, rank):
        """
        Normalize rank specification for 4D TT (I,Nu,Nv,C):
          - int R           -> [1, R, R, R, 1]
          - list/tuple len5 -> as-is
          - None            -> use cfg.migs.init_rank
        """
        if rank is None:
            R = int(self.cfg.migs.get("init_rank", 64))
            return [1, R, R, R, 1]

        if isinstance(rank, ListConfig):
            rank = list(rank)

        if isinstance(rank, (list, tuple)):
            rank = list(map(int, rank))
            assert len(rank) == 5, f"[TT-UV-SPLIT] rank list must have len=5, got {rank}"
            return rank

        R = int(rank)
        return [1, R, R, R, 1]

    # =========================================================
    # Shared padding primitive
    # =========================================================

    @torch.no_grad()
    def _zero_pad_pair_preserve(self, left, right, add, dim_left, dim_right):
        """
        Pad the bond dimension connecting two adjacent TT cores with small noise.
        """
        if add <= 0:
            return left, right

        dev = left.device

        dl_shape = list(left.shape);  dl_shape[dim_left]  = add
        dr_shape = list(right.shape); dr_shape[dim_right] = add

        left_std  = left.detach().std()
        right_std = right.detach().std()
        if (not torch.isfinite(left_std))  or left_std  < 1e-8:
            left_std  = left.detach().abs().mean()
        if (not torch.isfinite(right_std)) or right_std < 1e-8:
            right_std = right.detach().abs().mean()

        scale = 1e-2
        pad_left  = scale * left_std  * torch.randn(dl_shape, device=dev, dtype=left.dtype)
        pad_right = scale * right_std * torch.randn(dr_shape, device=dev, dtype=right.dtype)

        return torch.cat([left, pad_left], dim=dim_left), \
               torch.cat([right, pad_right], dim=dim_right)

    # =========================================================
    # r1 expansion (shared helper, works on any ParameterList)
    # =========================================================

    @torch.no_grad()
    def _expand_r1_for_branch(self, param_list: nn.ParameterList, r1_target: int):
        """
        Expand r1 bond (core0 dim2 ↔ core1 dim0) by replication.
        Works on any branch ParameterList.
        """
        c0 = param_list[0]
        c1 = param_list[1]

        r1_cur = c0.shape[2]
        r1_target = int(r1_target)
        if r1_cur >= r1_target:
            return

        def _repeat_to(x, dim, target):
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

        scale = r1_cur / float(r1_target)
        param_list[0] = nn.Parameter(_repeat_to(c0, dim=2, target=r1_target) * scale)
        param_list[1] = nn.Parameter(_repeat_to(c1, dim=0, target=r1_target))

    # =========================================================
    # REST branch rank expansion
    # =========================================================

    @torch.no_grad()
    def _expand_rest_ranks(self, rank_or_ranks):
        """
        Expand r2 and r3 bonds of the REST branch.
        """
        if isinstance(rank_or_ranks, (int, float)):
            R = int(rank_or_ranks)
            ranks_target = [1, R, R, R, 1]
        elif isinstance(rank_or_ranks, ListConfig):
            ranks_target = list(map(int, list(rank_or_ranks)))
        else:
            ranks_target = list(map(int, rank_or_ranks))

        c1 = self.tt_rest_gpu[1]
        c2 = self.tt_rest_gpu[2]

        # r2 : between core1 and core2
        r2_cur = c1.shape[2]
        r2_tgt = ranks_target[2]
        if r2_tgt > r2_cur:
            new_c1, new_c2 = self._zero_pad_pair_preserve(
                c1, c2, r2_tgt - r2_cur, dim_left=2, dim_right=0)
            self.tt_rest_gpu[1] = nn.Parameter(new_c1)
            self.tt_rest_gpu[2] = nn.Parameter(new_c2)

        c2 = self.tt_rest_gpu[2]

        # r3 : between core2 and core4_rest
        r3_cur = c2.shape[2]
        r3_tgt = ranks_target[3]
        if r3_tgt > r3_cur:
            core4_rest = self.recombine_core4_rest()  # (r3, 39, 1)
            new_c2, new_core4 = self._zero_pad_pair_preserve(
                c2, core4_rest, r3_tgt - r3_cur, dim_left=2, dim_right=0)
            self.tt_rest_gpu[2] = nn.Parameter(new_c2)

            # split back
            self.core4_scaling  = nn.Parameter(new_core4[:, 0:3,  :])
            self.core4_rotation = nn.Parameter(new_core4[:, 3:7,  :])
            self.core4_dc       = nn.Parameter(new_core4[:, 7:8,  :])
            self.core4_rest     = nn.Parameter(new_core4[:, 8:39, :])

        self.tt_rank_rest = ranks_target
        self._grid_cache_rest.clear()

    # =========================================================
    # OPACITY branch rank expansion
    # =========================================================

    @torch.no_grad()
    def _expand_opacity_ranks(self, rank_or_ranks):
        """
        Expand r2 and r3 bonds of the OPACITY branch.
        """
        if isinstance(rank_or_ranks, (int, float)):
            R = int(rank_or_ranks)
            ranks_target = [1, R, R, R, 1]
        elif isinstance(rank_or_ranks, ListConfig):
            ranks_target = list(map(int, list(rank_or_ranks)))
        else:
            ranks_target = list(map(int, rank_or_ranks))

        c1 = self.tt_opacity_gpu[1]
        c2 = self.tt_opacity_gpu[2]

        # r2
        r2_cur = c1.shape[2]
        r2_tgt = ranks_target[2]
        if r2_tgt > r2_cur:
            new_c1, new_c2 = self._zero_pad_pair_preserve(
                c1, c2, r2_tgt - r2_cur, dim_left=2, dim_right=0)
            self.tt_opacity_gpu[1] = nn.Parameter(new_c1)
            self.tt_opacity_gpu[2] = nn.Parameter(new_c2)

        c2 = self.tt_opacity_gpu[2]

        # r3 : between core2 and core4_opacity
        r3_cur = c2.shape[2]
        r3_tgt = ranks_target[3]
        if r3_tgt > r3_cur:
            new_c2, new_core4 = self._zero_pad_pair_preserve(
                c2, self.core4_opacity, r3_tgt - r3_cur, dim_left=2, dim_right=0)
            self.tt_opacity_gpu[2] = nn.Parameter(new_c2)
            self.core4_opacity = nn.Parameter(new_core4)  # (r3_new, 1, 1)

        self.tt_rank_opacity = ranks_target
        self._grid_cache_opacity.clear()

    # =========================================================
    # Debug prints
    # =========================================================

    def _debug_print_shapes(self, tag=""):
        if len(self.tt_rest_gpu) >= 3:
            c0 = self.tt_rest_gpu[0]
            c1 = self.tt_rest_gpu[1]
            c2 = self.tt_rest_gpu[2]
            c4 = self.recombine_core4_rest()
            print(f"[TT-UV-SPLIT]{tag} REST ranks={self.tt_rank_rest}")
            print(f"  rest core0: {tuple(c0.shape)}")
            print(f"  rest core1: {tuple(c1.shape)}")
            print(f"  rest core2: {tuple(c2.shape)}")
            print(f"  rest core4: {tuple(c4.shape)}")

        if len(self.tt_opacity_gpu) >= 3:
            c0 = self.tt_opacity_gpu[0]
            c1 = self.tt_opacity_gpu[1]
            c2 = self.tt_opacity_gpu[2]
            c4 = self.core4_opacity
            print(f"[TT-UV-SPLIT]{tag} OPACITY ranks={self.tt_rank_opacity}")
            print(f"  op core0: {tuple(c0.shape)}")
            print(f"  op core1: {tuple(c1.shape)}")
            print(f"  op core2: {tuple(c2.shape)}")
            print(f"  op core4: {tuple(c4.shape)}")

    # =========================================================
    # Init from gaussian model
    # =========================================================

    def init_from_tensor(self, gaussian_model):
        """
        Build TWO UV TTs from GaussianModel:
          - REST tensor    : (1,Nu,Nv,39)
          - OPACITY tensor : (1,Nu,Nv,1)

        Requires:
          gaussian_model._uv : (G,2) in [0,1]
        """
        with torch.no_grad():
            if hasattr(self.cfg, "migs") and getattr(self.cfg.migs, "skip_init_from_tensor", False):
                print("[TT-UV-SPLIT] skip_init_from_tensor=True, skipping")
                return

            device = gaussian_model._xyz.device
            G = gaussian_model._xyz.shape[0]
            print(f"[TT-UV-SPLIT] Initializing from {G} Gaussians using UV grid")

            # -------------------------
            # Assemble W blocks
            # -------------------------
            scaling      = gaussian_model._scaling.detach()                          # (G,3)
            rotation     = gaussian_model._rotation.detach()                         # (G,4)
            features_dc  = gaussian_model._features_dc.squeeze(-1).detach()          # (G,1)
            features_rest= gaussian_model._features_rest.squeeze(-1).detach()        # (G,31)
            opacity      = gaussian_model._opacity.detach()                          # (G,1)

            W_rest    = torch.cat([scaling, rotation, features_dc, features_rest], dim=1)  # (G,39)
            W_opacity = opacity                                                            # (G,1)

            assert W_rest.shape[1]    == 39, f"Expected 39, got {W_rest.shape[1]}"
            assert W_opacity.shape[1] == 1,  f"Expected 1, got {W_opacity.shape[1]}"

            # -------------------------
            # Priors
            # -------------------------
            W_prior_rest    = W_rest.mean(dim=0).clone()
            W_prior_opacity = W_opacity.mean(dim=0).clone()
            W_prior_opacity[0] = float(self.cfg.migs.get("empty_opacity_logit", -8.0))

            self.register_buffer("W_prior_rest",    W_prior_rest.detach())
            self.register_buffer("W_prior_opacity", W_prior_opacity.detach())

            # -------------------------
            # UV metadata
            # -------------------------
            if not (hasattr(gaussian_model, "_uv") and gaussian_model._uv is not None):
                raise RuntimeError("[TT-UV-SPLIT] gaussian_model has no _uv.")

            uv = gaussian_model._uv.detach().clone()
            if uv.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                uv = uv.float()
            self.register_buffer("gaussian_uv", uv)

            # -------------------------
            # Resolution
            # -------------------------
            if getattr(self.cfg.migs, "uv_resolution", None) is not None:
                res = self.cfg.migs.uv_resolution
                if isinstance(res, (list, tuple, ListConfig)):
                    Nu, Nv = map(int, list(res))
                else:
                    Nu = Nv = int(res)
            else:
                Nu = int(self.cfg.migs.get("uv_Nu", 256))
                Nv = int(self.cfg.migs.get("uv_Nv", 256))

            Nu = max(4, Nu)
            Nv = max(4, Nv)
            print(f"[TT-UV-SPLIT] Resolution: (Nu,Nv)=({Nu},{Nv}) cells={Nu*Nv}")

            # -------------------------
            # UV splatting
            # -------------------------
            uv01 = uv.clamp(0.0, 1.0)
            u = uv01[:, 0] * (Nu - 1)
            v = uv01[:, 1] * (Nv - 1)

            iu0 = torch.floor(u).long().clamp(0, Nu - 2)
            iv0 = torch.floor(v).long().clamp(0, Nv - 2)
            iu1 = iu0 + 1
            iv1 = iv0 + 1

            fu = (u - iu0.float()).clamp(0, 1)
            fv = (v - iv0.float()).clamp(0, 1)

            w00 = (1.0 - fu) * (1.0 - fv)
            w01 = (1.0 - fu) * fv
            w10 = fu          * (1.0 - fv)
            w11 = fu          * fv

            V_cells = Nu * Nv

            def lin_uv(iu, iv):
                return (iv * Nu + iu).long()

            def build_grid(W_feat):
                C = W_feat.shape[1]
                grid_flat   = torch.zeros(V_cells, C, device=device, dtype=W_feat.dtype)
                counts_flat = torch.zeros(V_cells, 1, device=device, dtype=torch.float32)
                for iu_c, iv_c, ww in [(iu0,iv0,w00),(iu0,iv1,w01),(iu1,iv0,w10),(iu1,iv1,w11)]:
                    idx = lin_uv(iu_c, iv_c)
                    ww2 = ww.view(-1, 1).to(torch.float32)
                    grid_flat.index_add_(0, idx, W_feat * ww2.to(W_feat.dtype))
                    counts_flat.index_add_(0, idx, ww2)
                grid   = grid_flat.view(1, Nu, Nv, C)
                counts = counts_flat.view(1, Nu, Nv, 1)
                grid   = grid / (counts.to(grid.dtype) + 1e-8)
                return grid, counts

            grid_rest,    counts = build_grid(W_rest)     # (1,Nu,Nv,39)
            grid_opacity, _      = build_grid(W_opacity)  # (1,Nu,Nv,1)

            occupied = (counts.squeeze(0).squeeze(-1) > 1e-6)  # (Nu,Nv)
            n_bleed  = int(self.cfg.migs.get("bleed_iters", 8))

            def bleed_grid(grid):
                occ_local = occupied.clone()
                C = grid.shape[-1]
                for _it in range(n_bleed):
                    empty = ~occ_local
                    if not empty.any():
                        break
                    g     = grid[0]
                    occ_f = occ_local.float()
                    val_sum = torch.zeros_like(g)
                    cnt_sum = torch.zeros(Nu, Nv, 1, device=device)
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        su = slice(max(-di,0), Nu + min(-di,0))
                        sv = slice(max(-dj,0), Nv + min(-dj,0))
                        tu = slice(max(di,0),  Nu + min(di,0))
                        tv = slice(max(dj,0),  Nv + min(dj,0))
                        mask_nb = occ_f[su, sv].unsqueeze(-1)
                        val_sum[tu, tv] += g[su, sv] * mask_nb
                        cnt_sum[tu, tv] += mask_nb
                    fillable = empty & (cnt_sum.squeeze(-1) > 0)
                    if fillable.any():
                        avg = val_sum / (cnt_sum + 1e-8)
                        grid[0][fillable] = avg[fillable]
                        occ_local[fillable] = True
                return grid, occ_local

            grid_rest,    occupied_rest    = bleed_grid(grid_rest)
            grid_opacity, occupied_opacity = bleed_grid(grid_opacity)
            occupied_final = occupied_rest | occupied_opacity

            print(
                f"[TT-UV-SPLIT] Bleed: {n_bleed} iters, "
                f"occupied {occupied_final.sum().item()}/{Nu*Nv} "
                f"({100*occupied_final.sum().item()/(Nu*Nv):.1f}%)"
            )

            # Fill remaining empty cells
            still_empty = ~occupied_final
            n_still = int(still_empty.sum().item())
            if n_still > 0:
                grid_rest[0][still_empty]    = self.W_prior_rest.unsqueeze(0).expand(n_still, -1)
                op_fill    = self.W_prior_opacity.clone()
                op_fill[0] = -2.0
                grid_opacity[0][still_empty] = op_fill.unsqueeze(0).expand(n_still, -1)
                print(f"[TT-UV-SPLIT] {n_still} cells still empty -> filled with priors")

            # Confidence map
            lambda_conf = float(self.cfg.migs.get("confidence_lambda", 2.0))
            conf = 1.0 - torch.exp(-lambda_conf * counts)
            bled_mask = occupied_final & (counts.squeeze(0).squeeze(-1) <= 1e-6)
            conf[0, bled_mask, :] = 1.0
            self.register_buffer("confidence_map", conf.detach())

            print(
                f"[TT-UV-SPLIT] Confidence map: "
                f"min={conf.min():.3f} max={conf.max():.3f} mean={conf.mean():.3f}"
            )

            # -------------------------
            # TT decomposition
            # -------------------------
            rank_rest    = self._to_tt_ranks_4d(
                self.cfg.migs.get("rank_rest",    self.cfg.migs.get("rank", None)))
            rank_opacity = self._to_tt_ranks_4d(
                self.cfg.migs.get("rank_opacity", self.cfg.migs.get("rank", None)))

            self.tt_rank_rest    = rank_rest
            self.tt_rank_opacity = rank_opacity

            tt_rest    = tensor_train(grid_rest,    rank=rank_rest,    verbose=self.verbose)
            tt_opacity = tensor_train(grid_opacity, rank=rank_opacity, verbose=self.verbose)

            # ---- REST branch ----
            self.tt_rest_gpu = nn.ParameterList(
                [nn.Parameter(c.to(device)) for c in tt_rest.factors[:3]])
            core4_rest_full = tt_rest.factors[3].to(device)  # (r3, 39, 1)

            self.core4_scaling  = nn.Parameter(core4_rest_full[:, 0:3,  :].detach().clone())
            self.core4_rotation = nn.Parameter(core4_rest_full[:, 3:7,  :].detach().clone())
            self.core4_dc       = nn.Parameter(core4_rest_full[:, 7:8,  :].detach().clone())
            self.core4_rest     = nn.Parameter(core4_rest_full[:, 8:39, :].detach().clone())

            # ---- OPACITY branch ----
            self.tt_opacity_gpu = nn.ParameterList(
                [nn.Parameter(c.to(device)) for c in tt_opacity.factors[:3]])
            self.core4_opacity = nn.Parameter(
                tt_opacity.factors[3].to(device).detach().clone())  # (r3, 1, 1)

            # alias for external compat
            self.tt_tensor_gpu = self.tt_rest_gpu

            self.register_buffer("NuNv", torch.tensor([Nu, Nv], device=device, dtype=torch.int64))

            self._debug_print_shapes(tag=" after_tt_init (before rank expansion)")

            # -------------------------
            # Rank expansion (BOTH branches independently)
            # -------------------------
            rank_cfg = self.cfg.migs.get("init_rank", None)
            if not isinstance(rank_cfg, (list, tuple, ListConfig)) and rank_cfg is not None:
                R = int(rank_cfg)

                # REST branch
                self._expand_r1_for_branch(self.tt_rest_gpu, R)
                self._expand_rest_ranks(R)

                # OPACITY branch
                self._expand_r1_for_branch(self.tt_opacity_gpu, R)
                self._expand_opacity_ranks(R)

                self._debug_print_shapes(tag=" after_rank_expansion")

            print(
                f"[TT-UV-SPLIT] Init complete.\n"
                f"  REST    tensor=(1,{Nu},{Nv},39), ranks={self.tt_rank_rest}\n"
                f"  OPACITY tensor=(1,{Nu},{Nv},1),  ranks={self.tt_rank_opacity}"
            )

            self._needs_opt_rebuild = True
            self._grid_cache_rest.clear()
            self._grid_cache_opacity.clear()

    # =========================================================
    # TT core accessors
    # =========================================================

    def recombine_core4_rest(self) -> torch.Tensor:
        """Return (r3, 39, 1) from REST block slices."""
        return torch.cat(
            [self.core4_scaling,
             self.core4_rotation,
             self.core4_dc,
             self.core4_rest],
            dim=1,
        )

    def get_core0_rest(self, identity_idx: int) -> torch.Tensor:
        core0 = self.tt_rest_gpu[0]
        assert 0 <= identity_idx < core0.shape[1]
        return core0[:, identity_idx:identity_idx + 1, :]

    def get_core0_opacity(self, identity_idx: int) -> torch.Tensor:
        core0 = self.tt_opacity_gpu[0]
        assert 0 <= identity_idx < core0.shape[1]
        return core0[:, identity_idx:identity_idx + 1, :]

    def get_tt_tensor_rest(self, identity_idx: int):
        return [
            self.get_core0_rest(identity_idx),
            self.tt_rest_gpu[1],
            self.tt_rest_gpu[2],
            self.recombine_core4_rest(),
        ]

    def get_tt_tensor_opacity(self, identity_idx: int):
        return [
            self.get_core0_opacity(identity_idx),
            self.tt_opacity_gpu[1],
            self.tt_opacity_gpu[2],
            self.core4_opacity,
        ]

    def _reconstruct_grid_rest(self, identity_idx: int) -> torch.Tensor:
        return tt_to_tensor(self.get_tt_tensor_rest(identity_idx))     # (1,Nu,Nv,39)

    def _reconstruct_grid_opacity(self, identity_idx: int) -> torch.Tensor:
        return tt_to_tensor(self.get_tt_tensor_opacity(identity_idx))  # (1,Nu,Nv,1)

    def _reconstruct_grid_for_identity(self, identity_idx: int) -> torch.Tensor:
        """Full (1,Nu,Nv,40) — for external compat."""
        grid_rest    = self._reconstruct_grid_rest(identity_idx)
        grid_opacity = self._reconstruct_grid_opacity(identity_idx)
        return torch.cat([grid_rest, grid_opacity], dim=-1)

    # =========================================================
    # Parameters / freeze / unfreeze
    # =========================================================

    def optimize_parameters(self):
        return [
            self.tt_rest_gpu[0],
            self.tt_rest_gpu[1],
            self.tt_rest_gpu[2],
            self.tt_opacity_gpu[0],
            self.tt_opacity_gpu[1],
            self.tt_opacity_gpu[2],
            self.core4_scaling,
            self.core4_rotation,
            self.core4_dc,
            self.core4_rest,
            self.core4_opacity,
        ]

    def freeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = False

    def unfreeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = True

    # =========================================================
    # Sampling
    # =========================================================

    def get_W_for_identity(self, identity_idx: int, uv_query: torch.Tensor = None) -> torch.Tensor:
        """
        Sample W at UV query points from REST and OPACITY grids.
        Returns (N, 40) = [scaling(3), rotation(4), dc(1), rest(31), opacity(1)]
        """
        if not hasattr(self, "NuNv"):
            raise RuntimeError("[TT-UV-SPLIT] init_from_tensor must be called before sampling.")

        if uv_query is None:
            if not hasattr(self, "gaussian_uv"):
                raise RuntimeError("[TT-UV-SPLIT] No uv_query and no stored gaussian_uv.")
            uv_query = self.gaussian_uv
        uv_query = uv_query.detach()

        Nu = int(self.NuNv[0].item())
        Nv = int(self.NuNv[1].item())

        # ---- Reconstruct grids ----
        if self.training:
            grid_rest    = self._reconstruct_grid_rest(identity_idx).squeeze(0)       # (Nu,Nv,39)
            grid_opacity = self._reconstruct_grid_opacity(identity_idx).squeeze(0)    # (Nu,Nv,1)
        else:
            if identity_idx not in self._grid_cache_rest:
                self._grid_cache_rest[identity_idx] = \
                    self._reconstruct_grid_rest(identity_idx).squeeze(0).detach()
            if identity_idx not in self._grid_cache_opacity:
                self._grid_cache_opacity[identity_idx] = \
                    self._reconstruct_grid_opacity(identity_idx).squeeze(0).detach()
            grid_rest    = self._grid_cache_rest[identity_idx]
            grid_opacity = self._grid_cache_opacity[identity_idx]

        # (1, C, Nv, Nu)
        inp_rest    = grid_rest.permute(2, 1, 0).unsqueeze(0).contiguous()
        inp_opacity = grid_opacity.permute(2, 1, 0).unsqueeze(0).contiguous()

        uv01 = uv_query.clamp(0.0, 1.0)
        x = uv01[:, 0] * 2.0 - 1.0
        y = uv01[:, 1] * 2.0 - 1.0
        grid_sample_coords = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)

        mode = str(self.cfg.migs.get("uv_sampling_mode", "baseline")).lower()

        def _gs(tensor, m):
            return F.grid_sample(
                tensor, grid_sample_coords,
                mode=m, padding_mode="border", align_corners=True)

        if mode == "all_nearest":
            rest_s = _gs(inp_rest,    "nearest").view(39, -1).permute(1, 0).contiguous()
            op_s   = _gs(inp_opacity, "nearest").view( 1, -1).permute(1, 0).contiguous()

        elif mode == "opacity_nearest":
            rest_s = _gs(inp_rest,    "bilinear").view(39, -1).permute(1, 0).contiguous()
            op_s   = _gs(inp_opacity, "nearest" ).view( 1, -1).permute(1, 0).contiguous()

        elif mode == "opacity_scale_nearest":
            scaling_inp   = inp_rest[:, 0:3,  :, :]
            rest_feat_inp = inp_rest[:, 3:39, :, :]
            scaling_s   = _gs(scaling_inp,   "nearest" ).view( 3, -1).permute(1, 0).contiguous()
            rest_feat_s = _gs(rest_feat_inp, "bilinear").view(36, -1).permute(1, 0).contiguous()
            rest_s = torch.cat([scaling_s, rest_feat_s], dim=1)
            op_s   = _gs(inp_opacity, "nearest").view(1, -1).permute(1, 0).contiguous()

        else:  # baseline: all bilinear
            rest_s = _gs(inp_rest,    "bilinear").view(39, -1).permute(1, 0).contiguous()
            op_s   = _gs(inp_opacity, "bilinear").view( 1, -1).permute(1, 0).contiguous()

        return torch.cat([rest_s, op_s], dim=1)  # (N, 40)

    # =========================================================
    # Identity expansion
    # =========================================================

    @torch.no_grad()
    def expand_first_core(self, n_identities: int):
        if len(self.tt_rest_gpu) == 0 or len(self.tt_opacity_gpu) == 0:
            raise RuntimeError("[TT-UV-SPLIT] TT cores must be initialized before expand_first_core.")

        def _expand_core0(param_list, tag):
            core0 = param_list[0]
            _, I, _ = core0.shape
            if I >= n_identities:
                return
            base  = core0[:, 0:1, :].detach()
            rep   = base.repeat(1, n_identities, 1)
            noise = self._randn_like(rep, tag=tag) * 1e-3
            param_list[0] = nn.Parameter(rep + noise)

        _expand_core0(self.tt_rest_gpu,    "core0_expand_noise_rest")
        _expand_core0(self.tt_opacity_gpu, "core0_expand_noise_opacity")

        self.tt_tensor_gpu = self.tt_rest_gpu
        self._grid_cache_rest.clear()
        self._grid_cache_opacity.clear()
        self._needs_opt_rebuild = True

        if self.optimizer is not None and self._opt_cfg is not None:
            self.set_optimizer(self._opt_cfg)

    @torch.no_grad()
    def add_identity(self, noise_scale: float = 0.05, rebuild_optimizer: bool = True) -> int:
        if len(self.tt_rest_gpu) == 0 or len(self.tt_opacity_gpu) == 0:
            raise RuntimeError("[TT-UV-SPLIT] TT cores not initialized.")

        def _add_to_core0(param_list, tag):
            core0 = param_list[0]
            _, I, _ = core0.shape
            U   = core0.detach()
            mu  = U.mean(dim=1, keepdim=True)
            sig = U.std(dim=1, unbiased=False, keepdim=True).clamp_(min=1e-8)
            eps = self._randn_like(mu, tag).expand_as(mu)
            new_row = mu + noise_scale * sig * eps
            param_list[0] = nn.Parameter(torch.cat([core0, new_row], dim=1))
            return I

        idx_rest = _add_to_core0(self.tt_rest_gpu,    "add_identity_rest")
        idx_op   = _add_to_core0(self.tt_opacity_gpu, "add_identity_opacity")
        assert idx_rest == idx_op

        self.tt_tensor_gpu = self.tt_rest_gpu
        self._grid_cache_rest.clear()
        self._grid_cache_opacity.clear()
        self._needs_opt_rebuild = True

        if rebuild_optimizer and (self.optimizer is not None) and (self._opt_cfg is not None):
            self.set_optimizer(self._opt_cfg)

        return idx_rest

    # =========================================================
    # Optimizer / step
    # =========================================================

    def set_optimizer(self, opt_cfg):
        self._opt_cfg = dict(opt_cfg) if opt_cfg is not None else {}

        lr_init    = float(self._opt_cfg.get("position_lr_init",    1.6e-4))
        lr_final   = float(self._opt_cfg.get("position_lr_final",   1.6e-6))
        decay_iters= int(  self._opt_cfg.get("position_lr_max_steps", 50000))

        param_groups = []
        decayed_idx  = []

        # REST branch TT cores (decayed LR)
        for core in [self.tt_rest_gpu[0], self.tt_rest_gpu[1], self.tt_rest_gpu[2]]:
            param_groups.append({"params": [core], "lr": lr_init,
                                  "initial_lr": lr_init, "final_lr": lr_final})
            decayed_idx.append(len(param_groups) - 1)

        # OPACITY branch TT cores (decayed LR)
        for core in [self.tt_opacity_gpu[0], self.tt_opacity_gpu[1], self.tt_opacity_gpu[2]]:
            param_groups.append({"params": [core], "lr": lr_init,
                                  "initial_lr": lr_init, "final_lr": lr_final})
            decayed_idx.append(len(param_groups) - 1)

        # REST heads (fixed LR)
        param_groups += [
            {"params": [self.core4_scaling],  "lr": float(self._opt_cfg.get("scaling_lr",  5e-3))},
            {"params": [self.core4_rotation], "lr": float(self._opt_cfg.get("rotation_lr", 1e-3))},
            {"params": [self.core4_dc],       "lr": float(self._opt_cfg.get("feature_lr",  2.5e-3))},
            {"params": [self.core4_rest],     "lr": float(self._opt_cfg.get("feature_lr",  2.5e-3))},
        ]

        # OPACITY head (fixed LR — can be higher since TV loss drives it)
        param_groups += [
            {"params": [self.core4_opacity],  "lr": float(self._opt_cfg.get("opacity_lr",  5e-2))},
        ]

        self.optimizer = torch.optim.Adam(param_groups)

        lr0   = max(lr_init,  1e-20)
        lrf   = max(lr_final, 1e-20)
        gamma = (lrf / lr0) ** (1.0 / max(decay_iters, 1))

        def make_lambda(i):
            return (lambda step: gamma ** step) if i in decayed_idx else (lambda step: 1.0)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[make_lambda(i) for i in range(len(self.optimizer.param_groups))]
        )

        self._needs_opt_rebuild = False

    def step(self, iteration=None):
        if self.optimizer is None:
            return

        if (iteration is not None) and (iteration < self.tt_delay):
            self.freeze_tt_parameters()
            self.optimizer.zero_grad(set_to_none=True)
            return

        if (iteration is not None) and (not self._tt_unfrozen) and (iteration >= self.tt_delay):
            self.unfreeze_tt_parameters()
            self._tt_unfrozen = True
            if self.verbose:
                print(f"[TT-UV-SPLIT] Unfrozen at iter {iteration}")

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self._grid_cache_rest.clear()
        self._grid_cache_opacity.clear()

        if self.scheduler is not None:
            self.scheduler.step()

    # =========================================================
    # TV loss — OPACITY branch ONLY
    # =========================================================

    def compute_tv_loss(self, identity_idx: int) -> torch.Tensor:
        """
        TV loss on OPACITY branch only.

        Gradient goes ONLY into:
          tt_opacity_gpu[0,1,2]  (opacity TT cores)
          core4_opacity           (opacity head)

        REST branch (tt_rest_gpu, core4_scaling, ...) is NEVER touched.
        → no ring artifacts in texture.
        """
        opacity_grid = self._reconstruct_grid_opacity(identity_idx)  # (1,Nu,Nv,1)

        diff_u = opacity_grid[:, 1:, :, :] - opacity_grid[:, :-1, :, :]
        diff_v = opacity_grid[:, :, 1:, :] - opacity_grid[:, :, :-1, :]

        return diff_u.abs().mean() + diff_v.abs().mean()