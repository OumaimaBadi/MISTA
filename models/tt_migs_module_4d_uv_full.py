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


class TTUltraMIGSModule4DUVGridFull(nn.Module):
    """
    Tensor-Train MIGS on a GEOMETRIC UV GRID — FULL COMPRESSION (M=43).

    Factorizes a tensor: (I, Nu, Nv, M)
      - I   : identities
      - Nu  : U grid resolution
      - Nv  : V grid resolution
      - M   : 43 — ALL gaussian parameters compressed

    Parameter layout in W (M=43):
      cols 0:3   -> xyz      (3)
      cols 3:6   -> scaling  (3)
      cols 6:10  -> rotation (4)
      cols 10:11 -> dc       (1)
      cols 11:42 -> rest     (31)
      cols 42:43 -> opacity  (1)

    "core4 trick" (Scene compatibility):
      core0 : (1, I,  r1)
      core1 : (r1, Nu, r2)
      core2 : (r2, Nv, r3)
      core4 : (r3, 43, 1)   split into per-block nn.Parameters

    LR policy (reads from cfg.migs YAML keys):
      position_lr_init / position_lr_final / position_lr_max_steps
        -> applied with LambdaLR decay to:
              core0, core1, core2   (TT structural cores)
              core4_xyz             (xyz is a position-like parameter)
      scaling_lr  -> core4_scaling  (fixed LR)
      rotation_lr -> core4_rotation (fixed LR)
      feature_lr  -> core4_dc, core4_rest (fixed LR)
      opacity_lr  -> core4_opacity  (fixed LR)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg    = cfg
        tt_cfg      = cfg.migs if not isinstance(cfg, dict) else cfg["migs"]

        self._base_seed = int(getattr(cfg, "seed", 123))

        self.tt_delay = tt_cfg.get("delay", 1000)
        if self.tt_delay is None:
            self.tt_delay = cfg.model.gaussian.get("delay", 0)

        self.verbose  = bool(tt_cfg.get("verbose", False))
        self.tt_rank  = tt_cfg.get("rank", None)

        # Optimiser state
        self.optimizer          = None
        self.scheduler          = None
        self._opt_cfg           = None
        self._needs_opt_rebuild = False
        self._tt_unfrozen       = False

        # Eval cache (reconstructed grids per identity)
        self._grid_cache = {}

        # TT cores  (core0 / core1 / core2 / placeholder for core4)
        self.tt_tensor_gpu = nn.ParameterList()

        # core4 split by parameter block (M=43)
        # Initialised to zeros; real values filled in init_from_tensor
        self.core4_xyz      = nn.Parameter(torch.zeros(1, 3,  1))
        self.core4_scaling  = nn.Parameter(torch.zeros(1, 3,  1))
        self.core4_rotation = nn.Parameter(torch.zeros(1, 4,  1))
        self.core4_dc       = nn.Parameter(torch.zeros(1, 1,  1))
        self.core4_rest     = nn.Parameter(torch.zeros(1, 31, 1))
        self.core4_opacity  = nn.Parameter(torch.zeros(1, 1,  1))

        self.save_dir = getattr(self.cfg, "exports_dir", "./exports")
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------ RNG --

    def _stream(self, tag: str, device) -> torch.Generator:
        h = int.from_bytes(hashlib.md5(tag.encode("utf8")).digest()[:8], "little")
        g = torch.Generator(device=device)
        g.manual_seed(self._base_seed ^ h)
        return g

    def _randn_like(self, ref, tag: str):
        g = self._stream(tag, ref.device)
        return torch.randn(ref.shape, device=ref.device, dtype=ref.dtype, generator=g)

    # ----------------------------------------------------- rank helpers ------

    def _to_tt_ranks_4d(self, rank):
        """Normalise rank spec to [1, r1, r2, r3, 1]."""
        if rank is None:
            R = int(self.cfg.migs.get("init_rank", 64))
            return [1, R, R, R, 1]
        if isinstance(rank, ListConfig):
            rank = list(rank)
        if isinstance(rank, (list, tuple)):
            rank = list(map(int, rank))
            assert len(rank) == 5, f"[TT-UV-Full] rank list must be length 5, got {rank}"
            return rank
        return [1, int(rank), int(rank), int(rank), 1]

    @torch.no_grad()
    def _zero_pad_pair_preserve(self, left, right, add, dim_left, dim_right):
        """Expand a bond dimension between two adjacent TT cores with small noise."""
        if add <= 0:
            return left, right
        dev = left.device
        dl  = list(left.shape);  dl[dim_left]  = add
        dr  = list(right.shape); dr[dim_right] = add

        std_l = left.detach().std()
        std_r = right.detach().std()
        if not torch.isfinite(std_l) or std_l < 1e-8:
            std_l = left.detach().abs().mean()
        if not torch.isfinite(std_r) or std_r < 1e-8:
            std_r = right.detach().abs().mean()

        scale = 1e-2
        pad_l = scale * std_l * torch.randn(dl, device=dev, dtype=left.dtype)
        pad_r = scale * std_r * torch.randn(dr, device=dev, dtype=right.dtype)
        return (torch.cat([left, pad_l], dim=dim_left),
                torch.cat([right, pad_r], dim=dim_right))

    @torch.no_grad()
    def _expand_ranks_to_targets_preserve(self, rank_or_ranks):
        """Expand r2 and r3 bond dimensions by zero-padding with small noise."""
        if isinstance(rank_or_ranks, ListConfig):
            rank_or_ranks = list(rank_or_ranks)
        if isinstance(rank_or_ranks, (list, tuple)):
            rt = list(map(int, rank_or_ranks))
            assert len(rt) == 5
        else:
            R  = int(rank_or_ranks)
            rt = [1, R, R, R, 1]

        c0, c1, c2 = self.tt_tensor_gpu[0], self.tt_tensor_gpu[1], self.tt_tensor_gpu[2]

        # expand r2  (core1 dim2 <-> core2 dim0)
        if rt[2] > c1.shape[2]:
            add = rt[2] - c1.shape[2]
            new_c1, new_c2 = self._zero_pad_pair_preserve(c1, c2, add, 2, 0)
            self.tt_tensor_gpu[1] = nn.Parameter(new_c1)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)

        c1, c2 = self.tt_tensor_gpu[1], self.tt_tensor_gpu[2]

        # expand r3  (core2 dim2 <-> core4 dim0)
        if rt[3] > c2.shape[2]:
            add       = rt[3] - c2.shape[2]
            core4_old = self.recombine_core4()  # (r3, 43, 1)
            new_c2, new_core4 = self._zero_pad_pair_preserve(c2, core4_old, add, 2, 0)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)

            assert new_core4.shape[1] == 43, \
                f"[TT-UV-Full] core4 M mismatch: {new_core4.shape[1]} != 43"

            self.core4_xyz      = nn.Parameter(new_core4[:, 0:3,   :])
            self.core4_scaling  = nn.Parameter(new_core4[:, 3:6,   :])
            self.core4_rotation = nn.Parameter(new_core4[:, 6:10,  :])
            self.core4_dc       = nn.Parameter(new_core4[:, 10:11, :])
            self.core4_rest     = nn.Parameter(new_core4[:, 11:42, :])
            self.core4_opacity  = nn.Parameter(new_core4[:, 42:43, :])

            if len(self.tt_tensor_gpu) >= 4:
                self.tt_tensor_gpu[3] = nn.Parameter(new_core4.detach().clone())

        self.tt_rank = rt
        self._needs_opt_rebuild = True
        self._grid_cache.clear()

    @torch.no_grad()
    def _expand_r1_by_replication(self, r1_target: int):
        """Expand r1 by replication with energy scaling."""
        c0, c1  = self.tt_tensor_gpu[0], self.tt_tensor_gpu[1]
        r1_cur  = c0.shape[2]
        r1_tgt  = int(r1_target)
        if r1_cur >= r1_tgt:
            return

        def _repeat_to(x, dim, target):
            cur    = x.shape[dim]
            if cur == target: return x
            times  = math.ceil(target / cur)
            reps   = [1] * x.ndim; reps[dim] = times
            slices = [slice(None)] * x.ndim; slices[dim] = slice(0, target)
            return x.repeat(*reps)[tuple(slices)]

        scale  = r1_cur / float(r1_tgt)
        c0_new = _repeat_to(c0, dim=2, target=r1_tgt) * scale
        c1_new = _repeat_to(c1, dim=0, target=r1_tgt)
        self.tt_tensor_gpu[0] = nn.Parameter(c0_new)
        self.tt_tensor_gpu[1] = nn.Parameter(c1_new)
        self._needs_opt_rebuild = True
        self._grid_cache.clear()

    def _debug_print_core_shapes(self, tag=""):
        if len(self.tt_tensor_gpu) < 3:
            print(f"[TT-UV-Full]{tag} cores not yet initialized")
            return
        c4 = self.recombine_core4()
        print(f"[TT-UV-Full]{tag} target ranks = {self.tt_rank}")
        print(f"[TT-UV-Full]{tag} core0 : {tuple(self.tt_tensor_gpu[0].shape)}  # (1,I,r1)")
        print(f"[TT-UV-Full]{tag} core1 : {tuple(self.tt_tensor_gpu[1].shape)}  # (r1,Nu,r2)")
        print(f"[TT-UV-Full]{tag} core2 : {tuple(self.tt_tensor_gpu[2].shape)}  # (r2,Nv,r3)")
        print(f"[TT-UV-Full]{tag} core4 : {tuple(c4.shape)}  # (r3,43,1)")

    # ----------------------------------------------------------------- init --

    def init_from_tensor(self, gaussian_model):
        """
        Build TT from UV grid.

        W layout (M=43):
          [xyz(3) | scaling(3) | rotation(4) | dc(1) | rest(31) | opacity(1)]

        Requires:
          gaussian_model._uv  : (G,2) in [0,1]
          gaussian_model._xyz, _scaling, _rotation,
          _features_dc, _features_rest, _opacity
        """
        with torch.no_grad():
            if hasattr(self.cfg, "migs") and getattr(self.cfg.migs, "skip_init_from_tensor", False):
                print("[TT-UV-Full] skip_init_from_tensor=True — skipping")
                return

            device = gaussian_model._xyz.device
            G      = gaussian_model._xyz.shape[0]
            print(f"[TT-UV-Full] Initializing from {G} Gaussians (M=43, full compression)")

            # ---- Assemble W (G, 43) ----
            xyz           = gaussian_model._xyz.detach()
            scaling       = gaussian_model._scaling.detach()
            rotation      = gaussian_model._rotation.detach()
            features_dc   = gaussian_model._features_dc.squeeze(-1).detach()    # (G,1)
            features_rest = gaussian_model._features_rest.squeeze(-1).detach()  # (G,31)
            opacity       = gaussian_model._opacity.detach()                     # (G,1)

            W_GM = torch.cat([xyz, scaling, rotation, features_dc, features_rest, opacity], dim=1)
            M    = W_GM.shape[1]
            assert M == 43, f"[TT-UV-Full] Expected M=43, got {M}"

            # ---- Prior for empty UV cells ----
            opacity_idx = 42  # opacity is the last column
            W_prior     = W_GM.mean(dim=0).clone()
            W_prior[opacity_idx] = float(self.cfg.migs.get("empty_opacity_logit", -8.0))
            self.register_buffer("W_prior", W_prior.detach())

            # ---- UV coords ----
            if not (hasattr(gaussian_model, "_uv") and gaussian_model._uv is not None):
                raise RuntimeError("[TT-UV-Full] gaussian_model has no _uv attribute.")
            uv = gaussian_model._uv.detach().clone()
            if uv.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                uv = uv.float()
            self.register_buffer("gaussian_uv", uv)

            # ---- Grid resolution (from YAML: uv_resolution or uv_Nu/uv_Nv) ----
            if getattr(self.cfg.migs, "uv_resolution", None) is not None:
                res = self.cfg.migs.uv_resolution
                if isinstance(res, (list, tuple, ListConfig)):
                    Nu, Nv = map(int, list(res))
                else:
                    Nu = Nv = int(res)
            else:
                Nu = int(self.cfg.migs.get("uv_Nu", 256))
                Nv = int(self.cfg.migs.get("uv_Nv", 256))
            Nu = max(4, Nu); Nv = max(4, Nv)
            print(f"[TT-UV-Full] Resolution: (Nu,Nv)=({Nu},{Nv})  cells={Nu*Nv}")

            # ---- Bilinear splatting into UV grid ----
            uv01 = uv.clamp(0.0, 1.0)
            u    = uv01[:, 0] * (Nu - 1)
            v    = uv01[:, 1] * (Nv - 1)
            iu0  = torch.floor(u).long().clamp(0, Nu - 2)
            iv0  = torch.floor(v).long().clamp(0, Nv - 2)
            iu1  = iu0 + 1; iv1 = iv0 + 1
            fu   = (u - iu0.float()).clamp(0, 1)
            fv   = (v - iv0.float()).clamp(0, 1)
            w00  = (1. - fu) * (1. - fv); w01 = (1. - fu) * fv
            w10  = fu * (1. - fv);        w11 = fu * fv

            V          = Nu * Nv
            grid_flat  = torch.zeros(V, M, device=device, dtype=W_GM.dtype)
            count_flat = torch.zeros(V, 1, device=device, dtype=torch.float32)

            def lin(iu, iv):
                return (iv * Nu + iu).long()

            for iu_c, iv_c, ww in [(iu0,iv0,w00),(iu0,iv1,w01),(iu1,iv0,w10),(iu1,iv1,w11)]:
                idx = lin(iu_c, iv_c)
                ww  = ww.view(-1, 1).to(torch.float32)
                grid_flat.index_add_(0, idx, W_GM * ww.to(W_GM.dtype))
                count_flat.index_add_(0, idx, ww)

            grid   = grid_flat.view(1, Nu, Nv, M)
            counts = count_flat.view(1, Nu, Nv, 1)
            grid   = grid / (counts.to(grid.dtype) + 1e-8)

            # ---- Bleed: fill empty cells from neighbours ----
            occupied = (counts.squeeze(0).squeeze(-1) > 1e-6)   # (Nu, Nv)
            n_bleed  = int(self.cfg.migs.get("bleed_iters", 8))

            for _ in range(n_bleed):
                empty = ~occupied
                if not empty.any():
                    break
                g       = grid[0]
                occ_f   = occupied.float()
                val_sum = torch.zeros_like(g)
                cnt_sum = torch.zeros(Nu, Nv, 1, device=device)

                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    su = slice(max(-di,0), Nu+min(-di,0))
                    sv = slice(max(-dj,0), Nv+min(-dj,0))
                    tu = slice(max( di,0), Nu+min( di,0))
                    tv = slice(max( dj,0), Nv+min( dj,0))
                    m  = occ_f[su, sv].unsqueeze(-1)
                    val_sum[tu, tv] += g[su, sv] * m
                    cnt_sum[tu, tv] += m

                fill = empty & (cnt_sum.squeeze(-1) > 0)
                if fill.any():
                    grid[0][fill] = (val_sum / (cnt_sum + 1e-8))[fill]
                    occupied[fill] = True

            print(f"[TT-UV-Full] Bleed: {n_bleed} iters — "
                  f"occupied {occupied.sum().item()}/{Nu*Nv} "
                  f"({100*occupied.sum().item()/(Nu*Nv):.1f}%)")

            # Fill remaining empty cells with W_prior (opacity less negative than -8)
            still_empty = ~occupied
            n_still     = int(still_empty.sum().item())
            if n_still > 0:
                W_fill              = W_prior.clone()
                W_fill[opacity_idx] = -2.0
                grid[0][still_empty] = W_fill.unsqueeze(0).expand(n_still, -1)
                print(f"[TT-UV-Full] {n_still} cells still empty -> W_prior (opacity=-2.0)")

            # ---- Confidence map ----
            lam  = float(self.cfg.migs.get("confidence_lambda", 2.0))
            conf = 1.0 - torch.exp(-lam * counts)
            bled = occupied & (counts.squeeze(0).squeeze(-1) <= 1e-6)
            conf[0, bled, :] = 1.0
            self.register_buffer("confidence_map", conf.detach())
            print(f"[TT-UV-Full] Confidence: "
                  f"min={conf.min():.3f} max={conf.max():.3f} mean={conf.mean():.3f}")

            # ---- TT decomposition on (1, Nu, Nv, 43) ----
            T            = grid
            ranks_target = self._to_tt_ranks_4d(self.cfg.migs.get("rank", None))
            self.tt_rank = ranks_target

            tt = tensor_train(T, rank=ranks_target, verbose=self.verbose)

            # Store TT cores
            self.tt_tensor_gpu = nn.ParameterList(
                [nn.Parameter(c.to(device)) for c in tt.factors]
            )

            # Split last core (r3, 43, 1) into per-block Parameters
            core4 = self.tt_tensor_gpu[3]
            self.core4_xyz      = nn.Parameter(core4[:, 0:3,   :].detach().clone())
            self.core4_scaling  = nn.Parameter(core4[:, 3:6,   :].detach().clone())
            self.core4_rotation = nn.Parameter(core4[:, 6:10,  :].detach().clone())
            self.core4_dc       = nn.Parameter(core4[:, 10:11, :].detach().clone())
            self.core4_rest     = nn.Parameter(core4[:, 11:42, :].detach().clone())
            self.core4_opacity  = nn.Parameter(core4[:, 42:43, :].detach().clone())

            self._debug_print_core_shapes(tag=" after_tt_init")

            # Optional rank expansion to init_rank
            rank_cfg = self.cfg.migs.get("init_rank", None)
            if not isinstance(rank_cfg, (list, tuple, ListConfig)) and rank_cfg is not None:
                R = int(rank_cfg)
                self._expand_r1_by_replication(R)
                self._expand_ranks_to_targets_preserve(R)
                self._debug_print_core_shapes(tag=" after_rank_expand")

            self.register_buffer(
                "NuNv",
                torch.tensor([Nu, Nv], device=device, dtype=torch.int64)
            )
            print(f"[TT-UV-Full] Init complete. "
                  f"ranks={self.tt_rank}  tensor=(I=1, Nu={Nu}, Nv={Nv}, M=43)")

            self._needs_opt_rebuild = True
            self._grid_cache.clear()

    # ----------------------------------------------- core4 trick  -----------

    def recombine_core4(self) -> torch.Tensor:
        """Reconstruct full last core (r3, 43, 1) from per-block Parameters."""
        return torch.cat([
            self.core4_xyz,
            self.core4_scaling,
            self.core4_rotation,
            self.core4_dc,
            self.core4_rest,
            self.core4_opacity,
        ], dim=1)

    def get_core0(self, identity_idx: int) -> torch.Tensor:
        core0 = self.tt_tensor_gpu[0]
        assert 0 <= identity_idx < core0.shape[1], \
            f"[TT-UV-Full] identity_idx={identity_idx} out of range [0,{core0.shape[1]})"
        return core0[:, identity_idx : identity_idx + 1, :]

    def get_tt_tensor(self, identity_idx: int):
        return [
            self.get_core0(identity_idx),   # (1, 1,  r1)
            self.tt_tensor_gpu[1],           # (r1, Nu, r2)
            self.tt_tensor_gpu[2],           # (r2, Nv, r3)
            self.recombine_core4(),          # (r3, 43, 1)
        ]

    def optimize_parameters(self):
        """All learnable parameters."""
        return [
            self.tt_tensor_gpu[0],
            self.tt_tensor_gpu[1],
            self.tt_tensor_gpu[2],
            self.core4_xyz,
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

    # ------------------------------------------------------- sampling --------

    def _reconstruct_grid_for_identity(self, identity_idx: int) -> torch.Tensor:
        """Full reconstructed tensor: (1, Nu, Nv, 43)."""
        return tt_to_tensor(self.get_tt_tensor(identity_idx))

    def get_W_for_identity(self, identity_idx: int, uv_query: torch.Tensor = None) -> torch.Tensor:
        """
        Sample W at UV query points.

        Returns (N, 43):
          cols 0:3   xyz
          cols 3:6   scaling
          cols 6:10  rotation
          cols 10:11 dc
          cols 11:42 rest
          cols 42:43 opacity
        """
        if not hasattr(self, "NuNv"):
            raise RuntimeError("[TT-UV-Full] Call init_from_tensor before sampling.")

        if uv_query is None:
            if not hasattr(self, "gaussian_uv"):
                raise RuntimeError("[TT-UV-Full] No uv_query and no stored gaussian_uv.")
            uv_query = self.gaussian_uv
        uv_query = uv_query.detach()

        # Reconstruct grid (Nu, Nv, 43)
        if self.training:
            grid_full = self._reconstruct_grid_for_identity(identity_idx).squeeze(0)
        else:
            if identity_idx not in self._grid_cache:
                self._grid_cache[identity_idx] = (
                    self._reconstruct_grid_for_identity(identity_idx).squeeze(0).detach()
                )
            grid_full = self._grid_cache[identity_idx]

        # grid_sample: (1, 43, Nv, Nu)
        inp = grid_full.permute(2, 1, 0).unsqueeze(0).contiguous()

        uv01 = uv_query.clamp(0.0, 1.0)
        gx   = uv01[:, 0] * 2.0 - 1.0   # U -> W axis
        gy   = uv01[:, 1] * 2.0 - 1.0   # V -> H axis
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)

        opacity_idx = 42
        mode = str(self.cfg.migs.get("uv_sampling_mode", "baseline")).lower()

        def _gs(x, m):
            return F.grid_sample(x, grid, mode=m, padding_mode="border", align_corners=True)

        if mode == "all_nearest":
            sampled = _gs(inp, "nearest").view(43, -1).permute(1, 0).contiguous()

        elif mode == "opacity_nearest":
            # bilinear for all features, nearest for opacity only
            feat = _gs(inp[:, :opacity_idx, :, :],              "bilinear").view(42, -1).permute(1, 0)
            op   = _gs(inp[:, opacity_idx:opacity_idx+1, :, :], "nearest" ).view(1,  -1).permute(1, 0)
            sampled = torch.cat([feat, op], dim=1).contiguous()

        else:
            # baseline: bilinear for everything (xyz, opacity, all)
            sampled = _gs(inp, "bilinear").view(43, -1).permute(1, 0).contiguous()

        if self.verbose:
            print(
                f"[TT-UV-Full] mode={mode} | "
                f"xyz    : min={sampled[:,0:3].min():.3f} max={sampled[:,0:3].max():.3f} | "
                f"opacity: min={sampled[:,opacity_idx].min():.2f} "
                f"max={sampled[:,opacity_idx].max():.2f} "
                f"mean={sampled[:,opacity_idx].mean():.2f}"
            )

        return sampled  # (N, 43)

    # ----------------------------------------- identity expansion -----------

    @torch.no_grad()
    def expand_first_core(self, n_identities: int):
        """Expand core0 from (1,I,r1) to (1,n_identities,r1) by replication + noise."""
        if len(self.tt_tensor_gpu) == 0:
            raise RuntimeError("[TT-UV-Full] TT cores must be initialized first.")
        core0    = self.tt_tensor_gpu[0]
        _, I, _  = core0.shape
        if I >= n_identities:
            return
        base  = core0[:, 0:1, :].detach()
        rep   = base.repeat(1, n_identities, 1)
        noise = self._randn_like(rep, "core0_expand_noise") * 1e-3
        self.tt_tensor_gpu[0] = nn.Parameter(rep + noise)
        self._grid_cache.clear()
        self._needs_opt_rebuild = True
        if self.optimizer is not None and self._opt_cfg is not None:
            self.set_optimizer(self._opt_cfg)

    @torch.no_grad()
    def add_identity(self, noise_scale: float = 0.05, rebuild_optimizer: bool = True) -> int:
        """Append one new identity to core0. Returns the new identity index."""
        if len(self.tt_tensor_gpu) == 0:
            raise RuntimeError("[TT-UV-Full] TT cores not initialized.")
        core0   = self.tt_tensor_gpu[0]
        _, I, _ = core0.shape
        U       = core0.detach()
        mu      = U.mean(dim=1, keepdim=True)
        sig     = U.std(dim=1, unbiased=False, keepdim=True).clamp_(min=1e-8)
        eps     = self._randn_like(mu, "add_identity").expand_as(mu)
        new_row = mu + noise_scale * sig * eps
        self.tt_tensor_gpu[0] = nn.Parameter(torch.cat([core0, new_row], dim=1))
        self._grid_cache.clear()
        self._needs_opt_rebuild = True
        if rebuild_optimizer and self.optimizer is not None and self._opt_cfg is not None:
            self.set_optimizer(self._opt_cfg)
        return I  # old I = new index

    # -------------------------------------------------- optimizer / step -----

    def set_optimizer(self, opt_cfg):
        """
        Build Adam optimiser + LambdaLR scheduler.

        Decay schedule (LambdaLR, robust to checkpoint resume):
          lr(step) = lr_init * gamma^step
          gamma = (lr_final / lr_init) ^ (1 / position_lr_max_steps)

        Applied to (from YAML cfg.migs):
          position_lr_init        -> lr_init  for core0/core1/core2/core4_xyz
          position_lr_final       -> lr_final for core0/core1/core2/core4_xyz
          position_lr_max_steps   -> decay_iters (default 50000)

        Fixed LR groups (from YAML cfg.migs):
          scaling_lr  -> core4_scaling
          rotation_lr -> core4_rotation
          feature_lr  -> core4_dc, core4_rest
          opacity_lr  -> core4_opacity
        """
        self._opt_cfg = dict(opt_cfg) if opt_cfg is not None else {}

        # Read LR values from YAML (same keys used in scene.py / tt_migs_module_4d_uv.py)
        lr_init     = float(self._opt_cfg.get("position_lr_init",      1.6e-4))
        lr_final    = float(self._opt_cfg.get("position_lr_final",     1.6e-6))
        decay_iters = int  (self._opt_cfg.get("position_lr_max_steps", 50000))

        # gamma for LambdaLR: lr_init * gamma^step converges to lr_final at step=decay_iters
        gamma = (max(lr_final, 1e-20) / max(lr_init, 1e-20)) ** (1.0 / max(decay_iters, 1))

        param_groups = []
        decayed_idx  = []   # indices of groups that use the exponential decay

        # ---- Decayed groups: TT structural cores ----
        for core in [self.tt_tensor_gpu[0], self.tt_tensor_gpu[1], self.tt_tensor_gpu[2]]:
            param_groups.append({
                "params":     [core],
                "lr":         lr_init,
                "initial_lr": lr_init,
                "final_lr":   lr_final,
            })
            decayed_idx.append(len(param_groups) - 1)

        # ---- Decayed group: core4_xyz (position-like parameter) ----
        param_groups.append({
            "params":     [self.core4_xyz],
            "lr":         lr_init,
            "initial_lr": lr_init,
            "final_lr":   lr_final,
        })
        decayed_idx.append(len(param_groups) - 1)

        # ---- Fixed-LR groups: other core4 blocks ----
        param_groups += [
            {"params": [self.core4_scaling],  "lr": float(self._opt_cfg.get("scaling_lr",  5e-3))},
            {"params": [self.core4_rotation], "lr": float(self._opt_cfg.get("rotation_lr", 1e-3))},
            {"params": [self.core4_dc],       "lr": float(self._opt_cfg.get("feature_lr",  2.5e-3))},
            {"params": [self.core4_rest],     "lr": float(self._opt_cfg.get("feature_lr",  2.5e-3))},
            {"params": [self.core4_opacity],  "lr": float(self._opt_cfg.get("opacity_lr",  5e-2))},
        ]

        self.optimizer = torch.optim.Adam(param_groups)

        # LambdaLR  — more robust than ExponentialLR for checkpoint resume:
        # the multiplier is recomputed from the absolute step count each time,
        # so there is no drift if the scheduler state is not perfectly restored.
        def make_lambda(i):
            if i in decayed_idx:
                return lambda step: gamma ** step
            return lambda step: 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[make_lambda(i) for i in range(len(param_groups))]
        )

        self._needs_opt_rebuild = False
        print(
            f"[TT-UV-Full] Optimizer built — "
            f"lr_init={lr_init:.2e}  lr_final={lr_final:.2e}  "
            f"decay_iters={decay_iters}  gamma={gamma:.6f}  "
            f"decayed_groups={decayed_idx}"
        )

    def step(self, iteration=None):
        if self.optimizer is None:
            return

        # Freeze everything before tt_delay
        if iteration is not None and iteration < self.tt_delay:
            self.freeze_tt_parameters()
            self.optimizer.zero_grad(set_to_none=True)
            return

        # Unfreeze once when crossing tt_delay
        if iteration is not None and not self._tt_unfrozen and iteration >= self.tt_delay:
            self.unfreeze_tt_parameters()
            self._tt_unfrozen = True
            if self.verbose:
                print(f"[TT-UV-Full] Parameters unfrozen at iteration {iteration}")

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._grid_cache.clear()

        if self.scheduler is not None:
            self.scheduler.step()

    # -------------------------------------------------- TV regularisation ----

    def compute_tv_loss(self, identity_idx: int) -> torch.Tensor:
        """Total-variation on opacity channel only (col 42)."""
        grid     = self._reconstruct_grid_for_identity(identity_idx)  # (1,Nu,Nv,43)
        op_grid  = grid[:, :, :, 42:43]
        diff_u   = op_grid[:, 1:, :, :] - op_grid[:, :-1, :, :]
        diff_v   = op_grid[:, :, 1:, :] - op_grid[:, :, :-1, :]
        return diff_u.abs().mean() + diff_v.abs().mean()