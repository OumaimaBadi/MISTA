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


class TTUltraMIGSModule5DUVD(nn.Module):
    """
    Tensor-Train MIGS on a UV + Depth grid.

    Factorizes a tensor: (I, Nu, Nv, D, M)
      - I   : identities
      - Nu  : U grid resolution
      - Nv  : V grid resolution
      - D   : depth layers along SMPL surface normal
      - M   : gaussian parameter dim = 40

    Convention "core4 trick":
      core0: (1,  I,  r1)
      core1: (r1, Nu, r2)
      core2: (r2, Nv, r3)
      core3: (r3, D,  r4)   ← nouvelle dimension profondeur
      core4: (r4, M,  1)    ← M=40, tous les params dans le TT

    M = 40 (xyz EXCLUDED):
      W = [scaling(3), rotation(4), dc(1), rest(31), opacity(1)]

    Avantage vs UV 4D:
      - dos et ventre ont meme (u,v) mais D different → cellules separees
      - scaling/rotation stockes dans le TT → compression totale
      - ~100% des cellules utiles (surface SMPL uniquement)
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

        # TT cores: core0/core1/core2/core3 + core4 split
        self.tt_tensor_gpu = nn.ParameterList()

        # ---- Split last core "core4" by parameter blocks (M=40) ----
        self.core4_scaling  = nn.Parameter(torch.zeros(1, 3,  1))
        self.core4_rotation = nn.Parameter(torch.zeros(1, 4,  1))
        self.core4_dc       = nn.Parameter(torch.zeros(1, 1,  1))
        self.core4_rest     = nn.Parameter(torch.zeros(1, 31, 1))
        self.core4_opacity  = nn.Parameter(torch.zeros(1, 1,  1))

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

    # -------------------------- rank helpers --------------------------

    def _to_tt_ranks_5d(self, rank):
        """
        Normalize rank for 5D TT (I, Nu, Nv, D, M):
        - int R    -> [1, R, R, R, R, 1]
        - list len6 -> as-is
        - None     -> use cfg.migs.init_rank
        """
        if rank is None:
            R = int(self.cfg.migs.get("init_rank", 32))
            return [1, R, R, R, R, 1]

        if isinstance(rank, ListConfig):
            rank = list(rank)

        if isinstance(rank, (list, tuple)):
            rank = list(map(int, rank))
            assert len(rank) == 6, f"[TT-UVD] rank list must have len=6, got {rank}"
            return rank

        R = int(rank)
        return [1, R, R, R, R, 1]

    # -------------------------- depth computation --------------------------

    @torch.no_grad()
    def _compute_signed_depth(self, gaussian_model, smpl_verts, smpl_faces):
        """
        Compute signed distance of each gaussian to SMPL surface.
        Positive = exterior (clothes), Negative = interior (body).

        Args:
            gaussian_model: has _xyz (G,3), _face_ids (G,), _bary (G,3)
            smpl_verts: (V, 3) SMPL vertices
            smpl_faces: (F, 3) SMPL faces

        Returns:
            signed_dist: (G,) signed distance along surface normal
        """
        xyz      = gaussian_model._xyz.detach()      # (G, 3)
        face_ids = gaussian_model._face_ids          # (G,)
        bary     = gaussian_model._bary              # (G, 3)

        # Vertices du triangle de chaque gaussienne
        faces_g = smpl_faces[face_ids]               # (G, 3)
        v0 = smpl_verts[faces_g[:, 0]]              # (G, 3)
        v1 = smpl_verts[faces_g[:, 1]]              # (G, 3)
        v2 = smpl_verts[faces_g[:, 2]]              # (G, 3)

        # Point projete sur la surface SMPL
        p_surface = (bary[:, 0:1] * v0
                   + bary[:, 1:2] * v1
                   + bary[:, 2:3] * v2)             # (G, 3)

        # Normale du triangle
        e1 = v1 - v0
        e2 = v2 - v0
        normal = torch.cross(e1, e2, dim=1)
        normal = F.normalize(normal, dim=1)          # (G, 3)

        # Distance signee
        diff = xyz - p_surface                       # (G, 3)
        signed_dist = (diff * normal).sum(dim=1)     # (G,)

        return signed_dist

    # -------------------------- zero pad helper --------------------------

    @torch.no_grad()
    def _zero_pad_pair_preserve(self, left, right, add, dim_left, dim_right):
        if add <= 0:
            return left, right

        dev = left.device
        dl_shape = list(left.shape);  dl_shape[dim_left]  = add
        dr_shape = list(right.shape); dr_shape[dim_right] = add

        left_std  = left.detach().std()
        right_std = right.detach().std()
        if not torch.isfinite(left_std)  or left_std  < 1e-8:
            left_std  = left.detach().abs().mean()
        if not torch.isfinite(right_std) or right_std < 1e-8:
            right_std = right.detach().abs().mean()

        scale = 1e-2
        pad_left  = scale * left_std  * torch.randn(dl_shape, device=dev, dtype=left.dtype)
        pad_right = scale * right_std * torch.randn(dr_shape, device=dev, dtype=right.dtype)

        return (torch.cat([left,  pad_left],  dim=dim_left),
                torch.cat([right, pad_right], dim=dim_right))

    # -------------------------- rank expansion --------------------------

    @torch.no_grad()
    def _expand_r1_by_replication(self, r1_target: int):
        """Expand r1 (identity bond) by replication."""
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

        scale = r1_cur / float(r1_target)
        self.tt_tensor_gpu[0] = nn.Parameter(_repeat_to(c0, dim=2, target=int(r1_target)) * scale)
        self.tt_tensor_gpu[1] = nn.Parameter(_repeat_to(c1, dim=0, target=int(r1_target)))
        self._needs_opt_rebuild = True
        self._grid_cache.clear()

    @torch.no_grad()
    def _expand_ranks_to_targets_preserve(self, rank_or_ranks):
        """Expand bond ranks r2, r3, r4, r5 by zero-padding."""
        if isinstance(rank_or_ranks, ListConfig):
            rank_or_ranks = list(rank_or_ranks)

        if isinstance(rank_or_ranks, (list, tuple)):
            ranks_target = list(map(int, rank_or_ranks))
            assert len(ranks_target) == 6, f"[TT-UVD] ranks_target must be len=6, got {ranks_target}"
        else:
            R = int(rank_or_ranks)
            ranks_target = [1, R, R, R, R, 1]

        # r2: core1 dim2 <-> core2 dim0
        c1 = self.tt_tensor_gpu[1]
        c2 = self.tt_tensor_gpu[2]
        r2_cur = c1.shape[2]
        r2_tgt = ranks_target[2]
        if r2_tgt > r2_cur:
            new_c1, new_c2 = self._zero_pad_pair_preserve(c1, c2, r2_tgt - r2_cur, 2, 0)
            self.tt_tensor_gpu[1] = nn.Parameter(new_c1)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)

        # r3: core2 dim2 <-> core3 dim0
        c2 = self.tt_tensor_gpu[2]
        c3 = self.tt_tensor_gpu[3]
        r3_cur = c2.shape[2]
        r3_tgt = ranks_target[3]
        if r3_tgt > r3_cur:
            new_c2, new_c3 = self._zero_pad_pair_preserve(c2, c3, r3_tgt - r3_cur, 2, 0)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)
            self.tt_tensor_gpu[3] = nn.Parameter(new_c3)

        # r4: core3 dim2 <-> core4 dim0
        c3 = self.tt_tensor_gpu[3]
        r4_cur = c3.shape[2]
        r4_tgt = ranks_target[4]
        if r4_tgt > r4_cur:
            core4_full = self.recombine_core4()   # (r4, M, 1)
            new_c3, new_core4 = self._zero_pad_pair_preserve(c3, core4_full, r4_tgt - r4_cur, 2, 0)
            self.tt_tensor_gpu[3] = nn.Parameter(new_c3)

            r4n, M, _ = new_core4.shape
            assert M == 40, f"[TT-UVD] Expected M=40, got {M}"
            self.core4_scaling  = nn.Parameter(new_core4[:, 0:3,   :])
            self.core4_rotation = nn.Parameter(new_core4[:, 3:7,   :])
            self.core4_dc       = nn.Parameter(new_core4[:, 7:8,   :])
            self.core4_rest     = nn.Parameter(new_core4[:, 8:39,  :])
            self.core4_opacity  = nn.Parameter(new_core4[:, 39:40, :])

            if len(self.tt_tensor_gpu) >= 5:
                self.tt_tensor_gpu[4] = nn.Parameter(new_core4.detach().clone())

        self.tt_rank = ranks_target
        self._needs_opt_rebuild = True
        self._grid_cache.clear()

    # -------------------------- init --------------------------

    def init_from_tensor(self, gaussian_model,
                         smpl_verts: torch.Tensor = None,
                         smpl_faces: torch.Tensor = None):
        """
        Build TT from UVD grid.

        Args:
            gaussian_model: has _uv (G,2), _face_ids (G,), _bary (G,3), _xyz (G,3)
            smpl_verts: (V, 3) SMPL canonical vertices (from metadata['smpl_verts'])
            smpl_faces: (F, 3) SMPL faces (from metadata['faces'])
        """
        with torch.no_grad():
            if hasattr(self.cfg, "migs") and getattr(self.cfg.migs, "skip_init_from_tensor", False):
                print("[TT-UVD] skip_init_from_tensor=True, skipping")
                return

            device = gaussian_model._xyz.device
            G = gaussian_model._xyz.shape[0]
            print(f"[TT-UVD] Initializing from {G} Gaussians using UVD grid")

            # ---- Assemble W (G, 40) ----
            scaling      = gaussian_model._scaling.detach()                      # (G, 3)
            rotation     = gaussian_model._rotation.detach()                     # (G, 4)
            features_dc  = gaussian_model._features_dc.squeeze(-1).detach()     # (G, 1)
            features_rest= gaussian_model._features_rest.squeeze(-1).detach()   # (G, 31)
            opacity      = gaussian_model._opacity.detach()                      # (G, 1)

            W_GM = torch.cat([scaling, rotation, features_dc, features_rest, opacity], dim=1)
            M = W_GM.shape[1]
            assert M == 40, f"[TT-UVD] Expected M=40, got {M}"

            # ---- Prior for empty cells ----
            W_prior = W_GM.mean(dim=0).clone()
            opacity_idx = 39
            W_prior[opacity_idx] = float(self.cfg.migs.get("empty_opacity_logit", -8.0))
            self.register_buffer("W_prior", W_prior.detach())

            # ---- UV coords ----
            if not (hasattr(gaussian_model, "_uv") and gaussian_model._uv is not None):
                raise RuntimeError("[TT-UVD] gaussian_model has no _uv.")
            uv = gaussian_model._uv.detach().clone().float()
            self.register_buffer("gaussian_uv", uv)

            # ---- Depth D computation ----
            D = int(self.cfg.migs.get("uvd_D", 8))
            d_max = float(self.cfg.migs.get("uvd_d_max", 0.05))

            if (smpl_verts is not None and smpl_faces is not None
                    and gaussian_model._face_ids is not None
                    and gaussian_model._bary is not None):

                smpl_verts_t = smpl_verts.to(device=device, dtype=torch.float32)
                smpl_faces_t = smpl_faces.to(device=device, dtype=torch.long)

                signed_dist = self._compute_signed_depth(
                    gaussian_model, smpl_verts_t, smpl_faces_t
                )

                print(f"[TT-UVD] signed_dist: "
                      f"min={signed_dist.min():.4f}  max={signed_dist.max():.4f}  "
                      f"mean={signed_dist.mean():.4f}  "
                      f"p5={torch.quantile(signed_dist, 0.05):.4f}  "
                      f"p95={torch.quantile(signed_dist, 0.95):.4f}")

                # Si d_max auto: utilise le 95e percentile
                if self.cfg.migs.get("uvd_d_max_auto", False):
                    d_max = float(torch.quantile(signed_dist.abs(), 0.95).item())
                    print(f"[TT-UVD] Auto d_max = {d_max:.4f}")

                d_norm = (signed_dist + d_max) / (2.0 * d_max)  # [0, 1]
                d_norm = d_norm.clamp(0.0, 1.0)
                d_idx  = (d_norm * (D - 1)).long().clamp(0, D - 1)  # (G,)

                # Stocker d_norm pour le sampling
                self.register_buffer("gaussian_d_norm", d_norm.detach())

            else:
                print("[TT-UVD] Warning: no smpl_verts/faces or face_ids/bary → using D=0 for all")
                d_idx = torch.zeros(G, dtype=torch.long, device=device)
                self.register_buffer("gaussian_d_norm",
                                     torch.zeros(G, device=device, dtype=torch.float32))

            print(f"[TT-UVD] D={D}, d_max={d_max}")
            dist_counts = [(d_idx == i).sum().item() for i in range(D)]
            print(f"[TT-UVD] d_idx distribution: {dist_counts}")

            # ---- UV grid resolution ----
            if getattr(self.cfg.migs, "uv_resolution", None) is not None:
                res = self.cfg.migs.uv_resolution
                if isinstance(res, (list, tuple, ListConfig)):
                    Nu, Nv = map(int, list(res))
                else:
                    Nu = Nv = int(res)
            else:
                Nu = int(self.cfg.migs.get("uv_Nu", 128))
                Nv = int(self.cfg.migs.get("uv_Nv", 128))

            Nu = max(4, Nu)
            Nv = max(4, Nv)
            print(f"[TT-UVD] Resolution: (Nu,Nv,D)=({Nu},{Nv},{D})  cells={Nu*Nv*D}")

            # ---- Splatting 3D (u, v, d) ----
            uv01 = uv.clamp(0.0, 1.0)
            u = uv01[:, 0] * (Nu - 1)  # (G,)
            v = uv01[:, 1] * (Nv - 1)  # (G,)
            # d_idx deja calculé: entier [0, D-1]

            # Bilinear en UV (4 coins), nearest en D
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

            V_total = Nu * Nv * D
            grid_flat   = torch.zeros(V_total, M, device=device, dtype=W_GM.dtype)
            counts_flat = torch.zeros(V_total, 1,  device=device, dtype=torch.float32)

            def lin_uvd(iu, iv, id_):
                # ordre: (iu * Nv + iv) * D + id_
                return (iu * Nv * D + iv * D + id_).long()

            corners = [
                (iu0, iv0, w00),
                (iu0, iv1, w01),
                (iu1, iv0, w10),
                (iu1, iv1, w11),
            ]

            for iu_c, iv_c, ww in corners:
                idx = lin_uvd(iu_c, iv_c, d_idx)
                ww_ = ww.view(-1, 1).to(torch.float32)
                grid_flat.index_add_(0, idx, W_GM * ww_.to(W_GM.dtype))
                counts_flat.index_add_(0, idx, ww_)

            grid   = grid_flat.view(1, Nu, Nv, D, M)    # (1, Nu, Nv, D, M)
            counts = counts_flat.view(1, Nu, Nv, D, 1)  # (1, Nu, Nv, D, 1)

            eps = 1e-8
            grid = grid / (counts.to(grid.dtype) + eps)

            # ---- BLEED en UV (propagation dans les voisins UV vides) ----
            occupied = (counts.squeeze(0).squeeze(-1) > 1e-6)  # (Nu, Nv, D)
            n_bleed = int(self.cfg.migs.get("bleed_iters", 4))

            for _it in range(n_bleed):
                empty = ~occupied
                if not empty.any():
                    break
                g3 = grid[0]       # (Nu, Nv, D, M)
                occ_f = occupied.float()
                val_sum = torch.zeros_like(g3)
                cnt_sum = torch.zeros(Nu, Nv, D, 1, device=device)

                for di, dj, dd in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                    su = slice(max(-di,0), Nu + min(-di,0))
                    sv = slice(max(-dj,0), Nv + min(-dj,0))
                    sd = slice(max(-dd,0), D  + min(-dd,0))
                    tu = slice(max(di,0),  Nu + min(di,0))
                    tv = slice(max(dj,0),  Nv + min(dj,0))
                    td = slice(max(dd,0),  D  + min(dd,0))
                    mask_nb = occ_f[su, sv, sd].unsqueeze(-1)
                    val_sum[tu, tv, td, :] += g3[su, sv, sd, :] * mask_nb
                    cnt_sum[tu, tv, td, :] += mask_nb

                fillable = empty & (cnt_sum.squeeze(-1) > 0)
                if fillable.any():
                    avg = val_sum / (cnt_sum + 1e-8)
                    grid[0][fillable] = avg[fillable]
                    occupied[fillable] = True

            n_occ = occupied.sum().item()
            print(f"[TT-UVD] Bleed {n_bleed} iters: "
                  f"occupied {n_occ}/{Nu*Nv*D} ({100*n_occ/(Nu*Nv*D):.1f}%)")

            # Fill remaining empty cells with W_prior (opacity très negative)
            still_empty = ~occupied
            n_still = int(still_empty.sum().item())
            if n_still > 0:
                W_fill = W_prior.clone()
                W_fill[opacity_idx] = -8.0
                grid[0][still_empty] = W_fill.unsqueeze(0).expand(n_still, -1)
                print(f"[TT-UVD] {n_still} cells still empty → W_prior opacity=-8")

            # ---- TT Decomposition sur (I, Nu, Nv, D, M) ----
            ranks_target = self._to_tt_ranks_5d(self.cfg.migs.get("rank", None))
            self.tt_rank = ranks_target
            print(f"[TT-UVD] TT decomposition ranks={ranks_target} ...")

            tt = tensor_train(grid, rank=ranks_target, verbose=self.verbose)

            # Stocker les 5 cores
            self.tt_tensor_gpu = nn.ParameterList(
                [nn.Parameter(c.to(device)) for c in tt.factors]
            )

            # Split dernier core (index 4) en blocs M=40
            core4 = self.tt_tensor_gpu[4]   # (r4, 40, 1)
            self.core4_scaling  = nn.Parameter(core4[:, 0:3,   :].detach().clone())
            self.core4_rotation = nn.Parameter(core4[:, 3:7,   :].detach().clone())
            self.core4_dc       = nn.Parameter(core4[:, 7:8,   :].detach().clone())
            self.core4_rest     = nn.Parameter(core4[:, 8:39,  :].detach().clone())
            self.core4_opacity  = nn.Parameter(core4[:, 39:40, :].detach().clone())

            # Debug shapes
            self._debug_print_core_shapes(tag=" after_tt_init")

            # Expansion optionnelle
            rank_cfg = self.cfg.migs.get("init_rank", None)
            if not isinstance(rank_cfg, (list, tuple, ListConfig)) and rank_cfg is not None:
                R = int(rank_cfg)
                self._expand_r1_by_replication(R)
                self._expand_ranks_to_targets_preserve([1, R, R, R, R, 1])
                self._debug_print_core_shapes(tag=" after_rank_expand")

            # Stocker metadata
            self.register_buffer("NuNvD",
                torch.tensor([Nu, Nv, D], device=device, dtype=torch.int64))
            self.register_buffer("uvd_d_max",
                torch.tensor([d_max], device=device, dtype=torch.float32))

            print(f"[TT-UVD] Init complete. "
                  f"tensor=(I,Nu,Nv,D,M)=(1,{Nu},{Nv},{D},40)  "
                  f"ranks={self.tt_rank}")

            self._needs_opt_rebuild = True
            self._grid_cache.clear()

    # -------------------------- debug --------------------------

    def _debug_print_core_shapes(self, tag=""):
        if len(self.tt_tensor_gpu) < 4:
            print(f"[TT-UVD]{tag} cores not initialized")
            return
        c0 = self.tt_tensor_gpu[0]
        c1 = self.tt_tensor_gpu[1]
        c2 = self.tt_tensor_gpu[2]
        c3 = self.tt_tensor_gpu[3]
        c4 = self.recombine_core4()
        print(f"[TT-UVD]{tag} ranks={self.tt_rank}")
        print(f"[TT-UVD]{tag} core0: {tuple(c0.shape)}  # (1,I,r1)")
        print(f"[TT-UVD]{tag} core1: {tuple(c1.shape)}  # (r1,Nu,r2)")
        print(f"[TT-UVD]{tag} core2: {tuple(c2.shape)}  # (r2,Nv,r3)")
        print(f"[TT-UVD]{tag} core3: {tuple(c3.shape)}  # (r3,D,r4)")
        print(f"[TT-UVD]{tag} core4: {tuple(c4.shape)}  # (r4,M,1)")

    # -------------------------- core4 trick --------------------------

    def recombine_core4(self) -> torch.Tensor:
        """Return full last core (r4, 40, 1)."""
        return torch.cat([
            self.core4_scaling,
            self.core4_rotation,
            self.core4_dc,
            self.core4_rest,
            self.core4_opacity,
        ], dim=1)

    def get_core0(self, identity_idx: int) -> torch.Tensor:
        core0 = self.tt_tensor_gpu[0]
        assert 0 <= identity_idx < core0.shape[1]
        return core0[:, identity_idx:identity_idx + 1, :]

    def get_tt_tensor(self, identity_idx: int):
        return [
            self.get_core0(identity_idx),
            self.tt_tensor_gpu[1],
            self.tt_tensor_gpu[2],
            self.tt_tensor_gpu[3],
            self.recombine_core4(),
        ]

    def optimize_parameters(self):
        return [
            self.tt_tensor_gpu[0],
            self.tt_tensor_gpu[1],
            self.tt_tensor_gpu[2],
            self.tt_tensor_gpu[3],
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
        """Reconstruct (1, Nu, Nv, D, M) grid."""
        cores = self.get_tt_tensor(identity_idx)
        return tt_to_tensor(cores)

    def get_W_for_identity(self,
                           identity_idx: int,
                           uv_query: torch.Tensor = None,
                           d_query: torch.Tensor = None) -> torch.Tensor:
        """
        Sample W at query (UV, D) points.

        Args:
            identity_idx: identity index
            uv_query: (N, 2) in [0,1]. If None → uses stored gaussian_uv.
            d_query:  (N,)  in [0,1] (normalized depth). If None → uses stored gaussian_d_norm.

        Returns:
            (N, 40)
        """
        if not hasattr(self, "NuNvD"):
            raise RuntimeError("[TT-UVD] init_from_tensor must be called first.")

        if uv_query is None:
            uv_query = self.gaussian_uv
        if d_query is None:
            d_query = self.gaussian_d_norm

        uv_query = uv_query.detach()
        d_query  = d_query.detach()

        Nu = int(self.NuNvD[0].item())
        Nv = int(self.NuNvD[1].item())
        D  = int(self.NuNvD[2].item())

        # ---- Reconstruire la grille ----
        if self.training:
            grid_full = self._reconstruct_grid_for_identity(identity_idx).squeeze(0)
            # (Nu, Nv, D, M)
        else:
            if identity_idx not in self._grid_cache:
                self._grid_cache[identity_idx] = (
                    self._reconstruct_grid_for_identity(identity_idx)
                    .squeeze(0).detach()
                )
            grid_full = self._grid_cache[identity_idx]

        # ---- Préparer pour grid_sample 3D ----
        # grid_sample 3D attend (N, C, D, H, W)
        # On mappe : W=Nu (axe U), H=Nv (axe V), D=D (axe profondeur)
        # grid_full: (Nu, Nv, D, M)
        # → permute vers (M, D, Nv, Nu) puis unsqueeze batch
        inp = grid_full.permute(3, 2, 1, 0).unsqueeze(0).contiguous()
        # inp: (1, M, D, Nv, Nu)

        # ---- Coordonnées de sampling en [-1, 1] ----
        uv01 = uv_query.clamp(0.0, 1.0)
        x = uv01[:, 0] * 2.0 - 1.0   # U → W axis
        y = uv01[:, 1] * 2.0 - 1.0   # V → H axis
        z = d_query.clamp(0.0, 1.0) * 2.0 - 1.0   # D → D axis

        N = uv_query.shape[0]
        # grid_sample 3D attend (1, N, 1, 1, 3)
        grid_coords = torch.stack([x, y, z], dim=-1).view(1, N, 1, 1, 3)

        # ---- Sample ----
        sampled = F.grid_sample(
            inp,
            grid_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (1, M, N, 1, 1)

        sampled = sampled.view(40, N).permute(1, 0).contiguous()  # (N, 40)

        return sampled

    # -------------------------- identity expansion --------------------------

    @torch.no_grad()
    def expand_first_core(self, n_identities: int):
        if len(self.tt_tensor_gpu) == 0:
            raise RuntimeError("[TT-UVD] TT cores not initialized.")
        core0 = self.tt_tensor_gpu[0]
        _, I, _ = core0.shape
        if I >= n_identities:
            return
        base = core0[:, 0:1, :].detach()
        rep  = base.repeat(1, n_identities, 1)
        noise = self._randn_like(rep, tag="core0_expand_noise") * 1e-3
        self.tt_tensor_gpu[0] = nn.Parameter(rep + noise)
        self._grid_cache.clear()
        self._needs_opt_rebuild = True
        if self.optimizer is not None and self._opt_cfg is not None:
            self.set_optimizer(self._opt_cfg)

    @torch.no_grad()
    def add_identity(self, noise_scale: float = 0.05, rebuild_optimizer: bool = True) -> int:
        if len(self.tt_tensor_gpu) == 0:
            raise RuntimeError("[TT-UVD] TT cores not initialized.")
        core0 = self.tt_tensor_gpu[0]
        _, I, _ = core0.shape
        U   = core0.detach()
        mu  = U.mean(dim=1, keepdim=True)
        sig = U.std(dim=1, unbiased=False, keepdim=True).clamp_(min=1e-8)
        eps = self._randn_like(mu, "add_identity").expand_as(mu)
        new_row = mu + noise_scale * sig * eps
        self.tt_tensor_gpu[0] = nn.Parameter(torch.cat([core0, new_row], dim=1))
        self._grid_cache.clear()
        self._needs_opt_rebuild = True
        if rebuild_optimizer and self.optimizer is not None and self._opt_cfg is not None:
            self.set_optimizer(self._opt_cfg)
        return I

    # -------------------------- optimizer --------------------------

    def set_optimizer(self, opt_cfg):
        self._opt_cfg = dict(opt_cfg) if opt_cfg is not None else {}

        lr_init    = float(self._opt_cfg.get("position_lr_init",     1.6e-4))
        lr_final   = float(self._opt_cfg.get("position_lr_final",    1.6e-6))
        decay_iters= int  (self._opt_cfg.get("position_lr_max_steps", 50000))

        param_groups = []
        decayed_idx  = []

        # Cores 0..3 avec decay
        for core in [self.tt_tensor_gpu[0], self.tt_tensor_gpu[1],
                     self.tt_tensor_gpu[2], self.tt_tensor_gpu[3]]:
            param_groups.append({
                "params":     [core],
                "lr":         lr_init,
                "initial_lr": lr_init,
                "final_lr":   lr_final,
            })
            decayed_idx.append(len(param_groups) - 1)

        # core4 slices: LR fixes
        param_groups += [
            {"params": [self.core4_scaling],  "lr": float(self._opt_cfg.get("scaling_lr",  5e-3))},
            {"params": [self.core4_rotation], "lr": float(self._opt_cfg.get("rotation_lr", 1e-3))},
            {"params": [self.core4_dc],       "lr": float(self._opt_cfg.get("feature_lr",  2.5e-3))},
            {"params": [self.core4_rest],     "lr": float(self._opt_cfg.get("feature_lr",  2.5e-3))},
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

        if iteration is not None and iteration < self.tt_delay:
            self.freeze_tt_parameters()
            self.optimizer.zero_grad(set_to_none=True)
            return

        if iteration is not None and not self._tt_unfrozen and iteration >= self.tt_delay:
            self.unfreeze_tt_parameters()
            self._tt_unfrozen = True
            if self.verbose:
                print(f"[TT-UVD] Unfrozen at iter {iteration}")

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._grid_cache.clear()
        if self.scheduler is not None:
            self.scheduler.step()

    # -------------------------- TV loss --------------------------

    def compute_tv_loss(self, identity_idx: int) -> torch.Tensor:
        """Total variation loss on opacity channel."""
        grid = self._reconstruct_grid_for_identity(identity_idx)
        # (1, Nu, Nv, D, M) → opacity = index 39
        opacity_grid = grid[:, :, :, :, 39:40]   # (1, Nu, Nv, D, 1)

        diff_u = opacity_grid[:, 1:, :,  :, :] - opacity_grid[:, :-1, :,  :, :]
        diff_v = opacity_grid[:, :,  1:, :, :] - opacity_grid[:, :,  :-1, :, :]
        diff_d = opacity_grid[:, :,  :,  1:, :] - opacity_grid[:, :,  :,  :-1, :]

        return diff_u.abs().mean() + diff_v.abs().mean() + diff_d.abs().mean()