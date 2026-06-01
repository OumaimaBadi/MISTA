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
    Tensor-Train MIGS on a GEOMETRIC UV GRID (no 3D voxel grid).

    Factorizes a tensor: (I, Nu, Nv, M)
      - I   : identities
      - Nu  : U grid resolution
      - Nv  : V grid resolution
      - M   : gaussian parameter dim (here 40)

    Convention "core4 trick" (for Scene consistency):
      core0: (1, I,  r1)
      core1: (r1, Nu, r2)
      core2: (r2, Nv, r3)
      core4: (r3, M,  1)   <-- last core is called core4 (not core3)

    M = 40 with xyz EXCLUDED:
      W = [scaling(3), rotation(4), dc(1), rest(31), opacity(1)] => 40
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
        self.tt_rank = tt_cfg.get("rank", None)

        # Optim
        self.optimizer = None
        self.scheduler = None
        self._opt_cfg = None
        self._needs_opt_rebuild = False
        self._tt_unfrozen = False

        # Cache reconstructed grids (eval)
        self._grid_cache = {}

        # TT cores container (core0/core1/core2 + "core4 stored as tt_tensor_gpu[3]")
        self.tt_tensor_gpu = nn.ParameterList()

        # ---- Split last core "core4" by parameter blocks (M=40) ----

        self.core4_scaling  = nn.Parameter(torch.zeros(1, 3, 1))
        self.core4_rotation = nn.Parameter(torch.zeros(1, 4, 1))
        self.core4_dc       = nn.Parameter(torch.zeros(1, 1, 1))
        self.core4_rest     = nn.Parameter(torch.zeros(1, 31, 1))
        self.core4_opacity  = nn.Parameter(torch.zeros(1, 1, 1))

        # self.core4_dc      = nn.Parameter(torch.zeros(1, 1, 1))
        # self.core4_rest    = nn.Parameter(torch.zeros(1, 31, 1))
        # self.core4_opacity = nn.Parameter(torch.zeros(1, 1, 1))

        self.save_dir = getattr(self.cfg, "exports_dir", "./exports")
        os.makedirs(self.save_dir, exist_ok=True)

    # -------------------------- RNG helpers --------------------------

    def _stream(self, tag: str, device) -> torch.Generator:
        h = int.from_bytes(hashlib.md5(tag.encode("utf8")).digest()[:8], "little")
        g = torch.Generator(device=device)
        g.manual_seed(self._base_seed ^ h)
        return g

    def _randn_like(self, ref, tag: str):
        g = self._stream(tag, ref.device)
        return torch.randn(ref.shape, device=ref.device, dtype=ref.dtype, generator=g)


    # -------------------------- rank expansion helpers (UV 4D, core4 trick) --------------------------
    def _to_tt_ranks_4d(self, rank):
        """
        Normalize rank specification for 4D TT (I,Nu,Nv,M):
        - int R           -> [1, R, R, R, 1]
        - None            -> use cfg.migs.init_rank as R
        """
        if rank is None:
            R = int(self.cfg.migs.get("init_rank", 64))
            return [1, R, R, R, 1]

        # OmegaConf ListConfig
        if isinstance(rank, ListConfig):
            rank = list(rank)

        if isinstance(rank, (list, tuple)):
            rank = list(map(int, rank))
            assert len(rank) == 5, f"[TT-UV] rank list must have len=5, got {rank}"
            return rank

        # scalar R
        R = int(rank)
        return [1, R, R, R, 1]


    @torch.no_grad()
    def _zero_pad_pair_preserve(self, left, right, add, dim_left, dim_right):
        """
        Pad the bond dimension connecting two adjacent TT cores:
        - pad `left`  along dim_left
        - pad `right` along dim_right
        Initialize padded slices with small noise scaled by each tensor's std (fallback mean abs).
        """
        if add <= 0:
            return left, right

        dev = left.device

        dl_shape = list(left.shape)
        dr_shape = list(right.shape)
        dl_shape[dim_left] = add
        dr_shape[dim_right] = add

        left_std  = left.detach().std()
        right_std = right.detach().std()
        if (not torch.isfinite(left_std)) or left_std < 1e-8:
            left_std = left.detach().abs().mean()
        if (not torch.isfinite(right_std)) or right_std < 1e-8:
            right_std = right.detach().abs().mean()

        scale = 1e-2
        pad_left  = scale * left_std  * torch.randn(dl_shape, device=dev, dtype=left.dtype)
        pad_right = scale * right_std * torch.randn(dr_shape, device=dev, dtype=right.dtype)

        new_left  = torch.cat([left,  pad_left],  dim=dim_left)
        new_right = torch.cat([right, pad_right], dim=dim_right)
        return new_left, new_right


    @torch.no_grad()
    def _expand_ranks_to_targets_preserve(self, rank_or_ranks):
        """
        Expand TT bond ranks for UV 4D tensor (I,Nu,Nv,M) using "core4 trick".

        Accepts either:
        - scalar R (int) -> ranks_target = [1, R, R, R, 1]
        - list/tuple len=5 -> ranks_target used as-is

        We expand r2 and r3 by padding the connecting bonds.
        r1 is handled separately by _expand_r1_by_replication().
        """

        # ---- normalize input to ranks_target = [1,r1,r2,r3,1] ----
        if isinstance(rank_or_ranks, ListConfig):
            rank_or_ranks = list(rank_or_ranks)

        if isinstance(rank_or_ranks, (list, tuple)):
            ranks_target = list(map(int, rank_or_ranks))
            assert len(ranks_target) == 5, f"[TT-UV] ranks_target must be len=5, got {ranks_target}"
        else:
            R = int(rank_or_ranks)
            ranks_target = [1, R, R, R, 1]

        # current cores
        c0 = self.tt_tensor_gpu[0]  # (1,I,r1)
        c1 = self.tt_tensor_gpu[1]  # (r1,Nu,r2)
        c2 = self.tt_tensor_gpu[2]  # (r2,Nv,r3)

        # ------------------- expand r2: between core1 dim2 and core2 dim0 -------------------
        r2_cur = c1.shape[2]
        r2_tgt = int(ranks_target[2])
        if r2_tgt > r2_cur:
            add = r2_tgt - r2_cur
            new_c1, new_c2 = self._zero_pad_pair_preserve(c1, c2, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[1] = nn.Parameter(new_c1)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)

        # refresh after potential update
        c1 = self.tt_tensor_gpu[1]
        c2 = self.tt_tensor_gpu[2]

        # ------------------- expand r3: between core2 dim2 and core4(recombined) dim0 -------------------
        r3_cur = c2.shape[2]
        r3_tgt = int(ranks_target[3])
        if r3_tgt > r3_cur:
            add = r3_tgt - r3_cur

            core4_full = self.recombine_core4()  # (r3,M,1)
            new_c2, new_core4 = self._zero_pad_pair_preserve(c2, core4_full, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)

            # split back new_core4 into blocks (M=40 layout WITH xyz)
            r3n, M, r_last = new_core4.shape
            assert r_last == 1
            assert M == 40, f"[TT-UV] Expected M=40, got {M}"

            self.core4_scaling  = nn.Parameter(new_core4[:, 0:3,   :])
            self.core4_rotation = nn.Parameter(new_core4[:, 3:7,   :])
            self.core4_dc       = nn.Parameter(new_core4[:, 7:8,   :])
            self.core4_rest     = nn.Parameter(new_core4[:, 8:39,  :])
            self.core4_opacity  = nn.Parameter(new_core4[:, 39:40, :])

            # self.core4_dc       = nn.Parameter(new_core4[:, 0:1,   :])
            # self.core4_rest     = nn.Parameter(new_core4[:, 1:32,  :])
            # self.core4_opacity  = nn.Parameter(new_core4[:, 32:33, :])

            # Optional: keep a materialized core4 tensor for debugging consistency
            if len(self.tt_tensor_gpu) >= 4:
                self.tt_tensor_gpu[3] = nn.Parameter(new_core4.detach().clone())

        self.tt_rank = list(map(int, ranks_target))
        self._needs_opt_rebuild = True
        self._grid_cache.clear()


    @torch.no_grad()
    def _expand_r1_by_replication(self, r1_target: int):
        """
        Expand r1 (bond between core0 and core1) by replication to preserve structure.

        core0: (1, I, r1)
        core1: (r1, Nu, r2)
        """
        c0 = self.tt_tensor_gpu[0]
        c1 = self.tt_tensor_gpu[1]

        r1_cur = c0.shape[2]
        if r1_cur >= int(r1_target):
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

        r1_target = int(r1_target)

        # scale to keep energy similar (important!)
        scale = r1_cur / float(r1_target)

        c0_new = _repeat_to(c0, dim=2, target=r1_target) * scale
        c1_new = _repeat_to(c1, dim=0, target=r1_target)

        self.tt_tensor_gpu[0] = nn.Parameter(c0_new)
        self.tt_tensor_gpu[1] = nn.Parameter(c1_new)

        self._needs_opt_rebuild = True
        self._grid_cache.clear()

    def _debug_print_core_shapes(self, tag=""):
        if len(self.tt_tensor_gpu) < 3:
            print(f"[TT-UV]{tag} cores not initialized")
            return

        c0 = self.tt_tensor_gpu[0]
        c1 = self.tt_tensor_gpu[1]
        c2 = self.tt_tensor_gpu[2]
        c4 = self.recombine_core4()

        print(f"[TT-UV]{tag} target ranks = {self.tt_rank}")
        print(f"[TT-UV]{tag} core0: {tuple(c0.shape)}  # (1,I,r1)")
        print(f"[TT-UV]{tag} core1: {tuple(c1.shape)}  # (r1,Nu,r2)")
        print(f"[TT-UV]{tag} core2: {tuple(c2.shape)}  # (r2,Nv,r3)")
        print(f"[TT-UV]{tag} core4: {tuple(c4.shape)}  # (r3,M,1)")

    # -------------------------- init --------------------------

    def init_from_tensor(self, gaussian_model):
        """
        Build TT from UV grid.

        Requires:
          - gaussian_model._uv : (G,2) in [0,1]
          - gaussian_model params: _xyz, _scaling, _rotation, _features_dc, _features_rest, _opacity

        Stores:
          - gaussian_uv as buffer
          - W_prior as buffer
          - occ_grid as buffer: (1,Nu,Nv,1)
          - NuNv as buffer: (Nu,Nv)
        """
        with torch.no_grad():
            if hasattr(self.cfg, "migs") and getattr(self.cfg.migs, "skip_init_from_tensor", False):
                print("[TT-UV] skip_init_from_tensor=True, skipping")
                return

            device = gaussian_model._xyz.device
            G = gaussian_model._xyz.shape[0]
            print(f"[TT-UV] Initializing from {G} Gaussians using UV grid")

            # ---- Assemble W (G,40) ----
            scaling = gaussian_model._scaling.detach()
            rotation = gaussian_model._rotation.detach()

            # use_sh=False expected => _features_dc: (G,1,1), _features_rest: (G,31,1)
            features_dc = gaussian_model._features_dc.squeeze(-1).detach()      # (G,1)
            features_rest = gaussian_model._features_rest.squeeze(-1).detach()  # (G,31)
            opacity = gaussian_model._opacity.detach()                          # (G,1)

            W_GM = torch.cat([scaling, rotation, features_dc, features_rest, opacity], dim=1)  # (G,40)
            M = W_GM.shape[1]
            assert M == 40, f"[TT-UV] Expected M=40, got {M}"



            # ---- Prior for empty UV cells ----
            W_prior = W_GM.mean(dim=0).clone()
            opacity_idx = 39
            W_prior[opacity_idx] = float(self.cfg.migs.get("empty_opacity_logit", -8.0))
            self.register_buffer("W_prior", W_prior.detach())

            # ---- UV metadata ----
            if not (hasattr(gaussian_model, "_uv") and gaussian_model._uv is not None):
                raise RuntimeError("[TT-UV] gaussian_model has no _uv. Provide UV coords for UV-grid TT.")

            uv = gaussian_model._uv.detach().clone()
            if uv.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                uv = uv.float()
            self.register_buffer("gaussian_uv", uv)

            # ---- Choose UV grid resolution ----
            # cfg.migs.uv_resolution can be int or [Nu,Nv]
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
            print(f"[TT-UV] Resolution: (Nu,Nv)=({Nu},{Nv})  cells={Nu*Nv}")

            # ---- Clamp UV to [0,1] ----
            uv01 = uv.clamp(0.0, 1.0)
            u = uv01[:, 0] * (Nu - 1)
            v = uv01[:, 1] * (Nv - 1)

            # ---- Bilinear splatting into UV grid (4 corners) ----
            iu0 = torch.floor(u).long().clamp(0, Nu - 2)
            iv0 = torch.floor(v).long().clamp(0, Nv - 2)
            iu1 = iu0 + 1
            iv1 = iv0 + 1

            fu = (u - iu0.float()).clamp(0, 1)
            fv = (v - iv0.float()).clamp(0, 1)

            w00 = (1.0 - fu) * (1.0 - fv)
            w01 = (1.0 - fu) * fv
            w10 = fu * (1.0 - fv)
            w11 = fu * fv

            V = Nu * Nv
            grid_flat = torch.zeros(V, M, device=device, dtype=W_GM.dtype)
            counts_flat = torch.zeros(V, 1, device=device, dtype=torch.float32)

            def lin_uv(iu, iv):
                return (iv * Nu + iu).long()

            corners = [
                (iu0, iv0, w00),
                (iu0, iv1, w01),
                (iu1, iv0, w10),
                (iu1, iv1, w11),
            ]

            for iu_c, iv_c, ww in corners:
                idx = lin_uv(iu_c, iv_c)
                ww = ww.view(-1, 1).to(torch.float32)
                grid_flat.index_add_(0, idx, W_GM * ww.to(W_GM.dtype))
                counts_flat.index_add_(0, idx, ww)

            grid = grid_flat.view(1, Nu, Nv, M)           # (1,Nu,Nv,M)
            counts = counts_flat.view(1, Nu, Nv, 1)       # (1,Nu,Nv,1)

            eps = 1e-8
            grid = grid / (counts.to(grid.dtype) + eps)

            # ---- BLEED: propagate occupied values into empty neighbors ----
            occupied = (counts.squeeze(0).squeeze(-1) > 1e-6)  # (Nu, Nv) bool
            n_bleed = int(self.cfg.migs.get("bleed_iters", 8))

            for _it in range(n_bleed):
                empty = ~occupied
                if not empty.any():
                    break

                g = grid[0]           # (Nu, Nv, M)
                occ_f = occupied.float()

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
                    occupied[fillable] = True

            print(f"[TT-UV] Bleed: {n_bleed} iters, "
                  f"occupied {occupied.sum().item()}/{Nu*Nv} "
                  f"({100*occupied.sum().item()/(Nu*Nv):.1f}%)")

            # Fill remaining deep-interior gaps with W_prior
            still_empty = ~occupied  
            n_still = int(still_empty.sum().item()) 
            if n_still > 0:
                W_prior_fill = W_prior.clone()
                W_prior_fill[opacity_idx] = -2.0  # moins transparent que -8.0
                grid[0][still_empty] = W_prior_fill.unsqueeze(0).expand(n_still, -1)
                print(f"[TT-UV] {n_still} cells still empty -> W_prior with opacity=-2.0")

            # ---- Confidence map from counts (continuous, replaces binary occ) ----
            lambda_conf = float(self.cfg.migs.get("confidence_lambda", 2.0))
            # Start from counts-based confidence for original cells
            conf = 1.0 - torch.exp(-lambda_conf * counts)  # (1,Nu,Nv,1)
            # Bled cells get full confidence (they have valid interpolated values)
            bled_mask = occupied & (counts.squeeze(0).squeeze(-1) <= 1e-6)
            conf[0, bled_mask, :] = 1.0
            self.register_buffer("confidence_map", conf.detach())

            print(f"[TT-UV] Confidence map: min={conf.min():.3f} "
                  f"max={conf.max():.3f} mean={conf.mean():.3f}")
            # ---- TT Decomposition on (I,Nu,Nv,M) ----
            # init has only 1 identity => I=1
            # ---- TT Decomposition on (I,Nu,Nv,M) ----
            T = grid  # (1,Nu,Nv,40)

            ranks_target = self._to_tt_ranks_4d(self.cfg.migs.get("rank", None))
            self.tt_rank = ranks_target

            tt = tensor_train(T, rank=ranks_target, verbose=self.verbose)

            # 1) store TT cores FIRST
            self.tt_tensor_gpu = nn.ParameterList([nn.Parameter(c.to(device)) for c in tt.factors])

            # 2) split last core into blocks
            core4 = self.tt_tensor_gpu[3]  # (r3,40,1)

            self.core4_scaling  = nn.Parameter(core4[:, 0:3,   :].detach().clone())
            self.core4_rotation = nn.Parameter(core4[:, 3:7,   :].detach().clone())
            self.core4_dc       = nn.Parameter(core4[:, 7:8,   :].detach().clone())
            self.core4_rest     = nn.Parameter(core4[:, 8:39,  :].detach().clone())
            self.core4_opacity  = nn.Parameter(core4[:, 39:40, :].detach().clone())
            # 3) now prints are valid
            self._debug_print_core_shapes(tag=" after_tt_init")

            # 4) OPTIONAL: expansion only if you really need it (see note below)
            # If cfg.migs.rank is an int, that is your target R.
            rank_cfg = self.cfg.migs.get("init_rank", None)
            if not isinstance(rank_cfg, (list, tuple, ListConfig)) and rank_cfg is not None:
                R = int(rank_cfg)
                self._expand_r1_by_replication(R)
                self._expand_ranks_to_targets_preserve(R)
                self._debug_print_core_shapes(tag=" after_rank_expand")


            # ---- store resolution ----
            self.register_buffer("NuNv", torch.tensor([Nu, Nv], device=device, dtype=torch.int64))

            print(f"[TT-UV] Init complete. ranks={self.tt_rank}  tensor=(I,Nu,Nv,M)=(1,{Nu},{Nv},40)")

            self._needs_opt_rebuild = True
            self._grid_cache.clear()

    # -------------------------- TT core handling (core4 trick) --------------------------

    def recombine_core4(self) -> torch.Tensor:
        """Return the full last core (r3,40,1) reconstructed from block slices."""
        return torch.cat(
            [
                self.core4_scaling,
                self.core4_rotation,
                self.core4_dc,
                self.core4_rest,
                self.core4_opacity,
            ],
            dim=1,
        )

    def get_core0(self, identity_idx: int) -> torch.Tensor:
        """core0 slice: (1,1,r1) from core0 (1,I,r1)."""
        core0 = self.tt_tensor_gpu[0]
        assert 0 <= identity_idx < core0.shape[1], f"identity_idx out of range: {identity_idx}"
        return core0[:, identity_idx : identity_idx + 1, :]

    def get_tt_tensor(self, identity_idx: int):
        """
        Return TT cores for a single identity slice:
          core0: (1,1,r1)
          core1: (r1,Nu,r2)
          core2: (r2,Nv,r3)
          core4: (r3,M,1)
        """
        return [
            self.get_core0(identity_idx),
            self.tt_tensor_gpu[1],
            self.tt_tensor_gpu[2],
            self.recombine_core4(),
        ]

    def optimize_parameters(self):
        """List of all learnable params (cores 0..2 + core4 blocks)."""
        return [
            self.tt_tensor_gpu[0],
            self.tt_tensor_gpu[1],
            self.tt_tensor_gpu[2],
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

    # -------------------------- sampling --------------------------

    def _reconstruct_grid_for_identity(self, identity_idx: int) -> torch.Tensor:
        """Full reconstructed tensor: (1,Nu,Nv,40)."""
        cores = self.get_tt_tensor(identity_idx)
        return tt_to_tensor(cores)  # (1,Nu,Nv,40)



    def get_W_for_identity(self, identity_idx: int, uv_query: torch.Tensor = None) -> torch.Tensor:
        """
        Sample W at query UV points from reconstructed UV grid.

        Flags (Hydra/YAML under cfg.migs):
        - uv_sampling_mode: baseline | opacity_nearest | opacity_scale_nearest | all_nearest
        - uv_confidence_mode: bilinear | nearest

        Returns:
        (N, 40)
        """
        if not hasattr(self, "NuNv"):
            raise RuntimeError("[TT-UV] init_from_tensor must be called before sampling.")

        if uv_query is None:
            if not hasattr(self, "gaussian_uv"):
                raise RuntimeError("[TT-UV] No uv_query provided and no stored gaussian_uv.")
            uv_query = self.gaussian_uv
        uv_query = uv_query.detach()

        Nu = int(self.NuNv[0].item())
        Nv = int(self.NuNv[1].item())

        # ---- Get reconstructed grid (Nu,Nv,40) ----
        if self.training:
            grid_full = self._reconstruct_grid_for_identity(identity_idx).squeeze(0)  # (Nu,Nv,40)
        else:
            if identity_idx not in self._grid_cache:
                self._grid_cache[identity_idx] = self._reconstruct_grid_for_identity(identity_idx).squeeze(0).detach()
            grid_full = self._grid_cache[identity_idx]

        # grid_sample expects (N,C,H,W). We'll map H=Nv, W=Nu.
        inp = grid_full.permute(2, 1, 0).unsqueeze(0).contiguous()  # (1,40,Nv,Nu)

        # ---- Prepare sampling grid ----
        uv01 = uv_query.clamp(0.0, 1.0)
        x = uv01[:, 0] * 2.0 - 1.0  # U -> W axis
        y = uv01[:, 1] * 2.0 - 1.0  # V -> H axis
        grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)  # (1,N,1,2)

        opacity_idx = 39

        # ---- Config flags ----
        mode = str(self.cfg.migs.get("uv_sampling_mode", "baseline")).lower()
        conf_mode = str(self.cfg.migs.get("uv_confidence_mode", "bilinear")).lower()

        def _gs(x, m: str):
            return F.grid_sample(
                x, grid,
                mode=m,
                padding_mode="border",
                align_corners=True
            )

        # ---- Sample channels according to the test mode ----
        if mode == "all_nearest":
            sampled = _gs(inp, "nearest").view(40, -1).permute(1, 0).contiguous()

        elif mode == "opacity_nearest":
            feat_inp = inp[:, :opacity_idx, :, :]         # (1,39,Nv,Nu)
            op_inp   = inp[:, opacity_idx:opacity_idx+1, :, :]

            feat_sampled = _gs(feat_inp, "bilinear").view(39, -1).permute(1, 0).contiguous()
            op_sampled   = _gs(op_inp,   "nearest" ).view(1,  -1).permute(1, 0).contiguous()

            sampled = torch.cat([feat_sampled, op_sampled], dim=1)

        elif mode == "opacity_scale_nearest":
            # scaling et rotation n'existent plus → fallback bilinear
            sampled = _gs(inp, "bilinear").view(40, -1).permute(1, 0).contiguous()

        else:
            # baseline: all bilinear
            sampled = _gs(inp, "bilinear").view(40, -1).permute(1, 0).contiguous()

        # ---- Confidence-based opacity modulation in alpha-space ----
        # conf_full = self.confidence_map.squeeze(0)  # (Nu, Nv, 1)
        # conf_inp = conf_full.permute(2, 1, 0).unsqueeze(0).contiguous()  # (1,1,Nv,Nu)

        # cm = "nearest" if conf_mode == "nearest" else "bilinear"
        # C = F.grid_sample(
        #     conf_inp, grid,
        #     mode=cm,
        #     padding_mode="zeros",
        #     align_corners=True
        # ).view(-1, 1)  # (N,1)

        # # Modulate in alpha-space (physically correct transmittance)
        # opacity_logit = sampled[:, opacity_idx:opacity_idx + 1]  # (N,1)
        # alpha_tt = torch.sigmoid(opacity_logit)
        # alpha_mod = C * alpha_tt

        # # Back to logit
        # alpha_mod = alpha_mod.clamp(1e-7, 1.0 - 1e-7)
        # opacity_new = torch.log(alpha_mod / (1.0 - alpha_mod))

        # sampled = torch.cat([sampled[:, :opacity_idx], opacity_new], dim=1)

        # if self.verbose:
        #     print(
        #         f"[TT-UV] mode={mode} conf_mode={conf_mode} | "
        #         f"opacity: min={opacity_new.min():.2f} max={opacity_new.max():.2f} mean={opacity_new.mean():.2f} | "
        #         f"conf: min={C.min():.3f} max={C.max():.3f} mean={C.mean():.3f}"
        #     )

        return sampled

    # def get_W_for_identity(self, identity_idx: int, uv_query: torch.Tensor = None) -> torch.Tensor:
    #     """
    #     Sample W at query UV points from reconstructed UV grid.

    #     Args:
    #       identity_idx: identity index
    #       uv_query: (N,2) in [0,1]; if None uses stored gaussian_uv.

    #     Returns:
    #       (N,40)
    #     """
    #     if not hasattr(self, "NuNv"):
    #         raise RuntimeError("[TT-UV] init_from_tensor must be called before sampling.")

    #     if uv_query is None:
    #         if not hasattr(self, "gaussian_uv"):
    #             raise RuntimeError("[TT-UV] No uv_query provided and no stored gaussian_uv.")
    #         uv_query = self.gaussian_uv
    #     uv_query = uv_query.detach()

    #     Nu = int(self.NuNv[0].item())
    #     Nv = int(self.NuNv[1].item())

    #     # ---- Get reconstructed grid (Nu,Nv,40) ----
    #     if self.training:
    #         grid_full = self._reconstruct_grid_for_identity(identity_idx).squeeze(0)  # (Nu,Nv,40)
    #     else:
    #         if identity_idx not in self._grid_cache:
    #             self._grid_cache[identity_idx] = self._reconstruct_grid_for_identity(identity_idx).squeeze(0).detach()
    #         grid_full = self._grid_cache[identity_idx]

    #     # grid_sample expects (N,C,H,W). We'll map H=Nv, W=Nu.
    #     inp = grid_full.permute(2, 1, 0).unsqueeze(0).contiguous()  # (1,40,Nv,Nu)

    #     # ---- Prepare sampling grid ----
    #     uv01 = uv_query.clamp(0.0, 1.0)
    #     x = uv01[:, 0] * 2.0 - 1.0  # U -> W axis
    #     y = uv01[:, 1] * 2.0 - 1.0  # V -> H axis
    #     grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)  # (1,N,1,2)

    #     # ---- Sample ALL channels with bilinear (including opacity) ----
    #     opacity_idx = 39

    #     sampled = F.grid_sample(
    #         inp, grid,
    #         mode="bilinear",
    #         padding_mode="border",
    #         align_corners=True
    #     )  # (1, 40, N, 1)
    #     sampled = sampled.view(40, -1).permute(1, 0).contiguous()  # (N, 40)

    #     # ---- Confidence-based opacity modulation in alpha-space ----
    #     conf_full = self.confidence_map.squeeze(0)  # (Nu, Nv, 1)
    #     conf_inp = conf_full.permute(2, 1, 0).unsqueeze(0).contiguous()  # (1,1,Nv,Nu)

    #     C = F.grid_sample(
    #         conf_inp, grid,
    #         mode="bilinear",   # smooth — no hard edges
    #         padding_mode="zeros",
    #         align_corners=True
    #     ).view(-1, 1)  # (N, 1)

    #     # Modulate in alpha-space (physically correct transmittance)
    #     opacity_logit = sampled[:, opacity_idx:opacity_idx + 1]  # (N, 1)
    #     alpha_tt = torch.sigmoid(opacity_logit)
    #     alpha_mod = C * alpha_tt  # confidence × opacity

    #     # Back to logit
    #     alpha_mod = alpha_mod.clamp(1e-7, 1.0 - 1e-7)
    #     opacity_new = torch.log(alpha_mod / (1.0 - alpha_mod))

    #     sampled = torch.cat([sampled[:, :opacity_idx], opacity_new], dim=1)

    #     if self.verbose:
    #         print(
    #             f"[TT-UV] opacity: min={opacity_new.min():.2f} "
    #             f"max={opacity_new.max():.2f} mean={opacity_new.mean():.2f} "
    #             f"conf: min={C.min():.3f} max={C.max():.3f} mean={C.mean():.3f}"
    #         )

    #     return sampled

    # -------------------------- identity expansion (core0: (1,I,r1)) --------------------------

    @torch.no_grad()
    def expand_first_core(self, n_identities: int):
        """
        Expand identity core0 from (1,I,r1) to (1,n_identities,r1),
        by replicating identity 0 + small noise.
        """
        if len(self.tt_tensor_gpu) == 0:
            raise RuntimeError("[TT-UV] TT cores must be initialized before expand_first_core.")

        core0 = self.tt_tensor_gpu[0]  # (1,I,r1)
        _, I, _ = core0.shape
        if I >= n_identities:
            return

        base = core0[:, 0:1, :].detach()
        rep = base.repeat(1, n_identities, 1)
        noise = self._randn_like(rep, tag="core0_expand_noise") * 1e-3

        self.tt_tensor_gpu[0] = nn.Parameter(rep + noise)

        self._grid_cache.clear()
        self._needs_opt_rebuild = True
        if self.optimizer is not None and self._opt_cfg is not None:
            self.set_optimizer(self._opt_cfg)

    @torch.no_grad()
    def add_identity(self, noise_scale: float = 0.05, rebuild_optimizer: bool = True) -> int:
        """
        Append one identity to core0: (1,I,r1).
        Returns the new identity index.
        """
        if len(self.tt_tensor_gpu) == 0:
            raise RuntimeError("[TT-UV] TT cores not initialized.")

        core0 = self.tt_tensor_gpu[0]  # (1,I,r1)
        _, I, _ = core0.shape

        U = core0.detach()
        mu = U.mean(dim=1, keepdim=True)
        sig = U.std(dim=1, unbiased=False, keepdim=True).clamp_(min=1e-8)

        eps = self._randn_like(mu, "add_identity").expand_as(mu)
        new_row = mu + noise_scale * sig * eps  # (1,1,r1)

        self.tt_tensor_gpu[0] = nn.Parameter(torch.cat([core0, new_row], dim=1))

        self._grid_cache.clear()
        self._needs_opt_rebuild = True

        if rebuild_optimizer and (self.optimizer is not None) and (self._opt_cfg is not None):
            self.set_optimizer(self._opt_cfg)

        return I  # old I is the new index

    # -------------------------- optimizer / step --------------------------

    def set_optimizer(self, opt_cfg):
        """
        MIGS-style: decay for TT cores (core0/core1/core2), fixed LR for core4 block slices.
        """
        self._opt_cfg = dict(opt_cfg) if opt_cfg is not None else {}

        lr_init = float(self._opt_cfg.get("position_lr_init", 1.6e-4))
        lr_final = float(self._opt_cfg.get("position_lr_final", 1.6e-6))
        decay_iters = int(self._opt_cfg.get("position_lr_max_steps", 50000))

        param_groups = []
        decayed_idx = []

        # decayed TT cores: core0/core1/core2
        for core in [self.tt_tensor_gpu[0], self.tt_tensor_gpu[1], self.tt_tensor_gpu[2]]:
            param_groups.append(
                {
                    "params": [core],
                    "lr": lr_init,
                    "initial_lr": lr_init,
                    "final_lr": lr_final,
                }
            )
            decayed_idx.append(len(param_groups) - 1)

        # core4 slices: per-block lr
        param_groups += [
            {"params": [self.core4_scaling],  "lr": float(self._opt_cfg.get("scaling_lr", 5e-3))},
            {"params": [self.core4_rotation], "lr": float(self._opt_cfg.get("rotation_lr", 1e-3))},
            {"params": [self.core4_dc],       "lr": float(self._opt_cfg.get("feature_lr", 2.5e-3))},
            {"params": [self.core4_rest],     "lr": float(self._opt_cfg.get("feature_lr", 2.5e-3))},
            {"params": [self.core4_opacity],  "lr": float(self._opt_cfg.get("opacity_lr", 5e-2))},
        ]

        self.optimizer = torch.optim.Adam(param_groups)

        lr0 = max(lr_init, 1e-20)
        lrf = max(lr_final, 1e-20)
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

        # delay: freeze everything before tt_delay
        if (iteration is not None) and (iteration < self.tt_delay):
            self.freeze_tt_parameters()
            self.optimizer.zero_grad(set_to_none=True)
            return

        # unfreeze once when passing tt_delay
        if (iteration is not None) and (not self._tt_unfrozen) and (iteration >= self.tt_delay):
            self.unfreeze_tt_parameters()
            self._tt_unfrozen = True
            if self.verbose:
                print(f"[TT-UV] Unfrozen at iter {iteration}")

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self._grid_cache.clear()
        if self.scheduler is not None:
            self.scheduler.step()


    def compute_tv_loss(self, identity_idx: int) -> torch.Tensor:
        grid = self._reconstruct_grid_for_identity(identity_idx)  # (1,Nu,Nv,40)
        
        # Seulement l'opacité (canal 39) — pas les couleurs ni la géométrie
        opacity_grid = grid[:, :, :, 39:40]  # (1,Nu,Nv,1)
        
        diff_u = opacity_grid[:, 1:, :, :] - opacity_grid[:, :-1, :, :]
        diff_v = opacity_grid[:, :, 1:, :] - opacity_grid[:, :, :-1, :]
        
        return diff_u.abs().mean() + diff_v.abs().mean()
