import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor
import hashlib
from omegaconf import ListConfig
import functools
tl.set_backend('pytorch')





class TTUltraMIGSModule5DGrid(nn.Module):
    """
    Tensor-Train MIGS with GEOMETRIC VOXEL GRID (no Hilbert, no arbitrary reshape).
    Factorizes a (I, Nx, Ny, Nz, M) tensor where Nx/Ny/Nz correspond to actual X/Y/Z axes.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        tt_cfg = cfg.migs if not isinstance(cfg, dict) else cfg["migs"]
        self._base_seed = int(getattr(cfg, "seed", 123))
        
        self.tt_delay = tt_cfg.get("delay", 1000)
        if self.tt_delay is None:
            self.tt_delay = cfg.model.gaussian.get("delay", 0)

        self.tt_rank = tt_cfg.get("rank")
        self.tt_shape = tt_cfg.get("tt_shape")
        self.verbose = bool(tt_cfg.get("verbose", False))

        self.optimizer = None
        self.scheduler = None
        self._opt_cfg = None
        self._needs_opt_rebuild = False
        self._tt_unfrozen = False
        self._grid_cache = {}

        self.tt_tensor_gpu = nn.ParameterList()

        # Last-mode split Parameters
        # self.core4_xyz      = nn.Parameter(torch.zeros(1, 3, 1))
        self.core4_scaling  = nn.Parameter(torch.zeros(1, 3, 1))
        self.core4_rotation = nn.Parameter(torch.zeros(1, 4, 1))
        self.core4_dc       = nn.Parameter(torch.zeros(1, 1, 1))
        self.core4_rest     = nn.Parameter(torch.zeros(1, 31, 1))
        self.core4_opacity  = nn.Parameter(torch.zeros(1, 1, 1))
        exp_dir = getattr(self.cfg, "exp_dir", None)
        if exp_dir is None:
            exp_dir = "./exports"  # fallback sûr
        self.save_dir = exp_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.fold_vis_dir = os.path.join(self.save_dir, "folding_vis")
        os.makedirs(self.fold_vis_dir, exist_ok=True)


    def _stream(self, tag: str, device) -> torch.Generator:
        h = int.from_bytes(hashlib.md5(tag.encode("utf8")).digest()[:8], 'little')
        g = torch.Generator(device=device)
        g.manual_seed(self._base_seed ^ h)
        return g

    def _randn_like(self, ref, tag):
        g = self._stream(tag, ref.device)
        return torch.randn(ref.shape, device=ref.device, dtype=ref.dtype, generator=g)

    def _compute_grid_resolution_A2(self, Lx, Ly, Lz, B, min_res=8, max_res=256):
        # Evite divisions foireuses
        Lx = float(max(Lx, 1e-12)); Ly = float(max(Ly, 1e-12)); Lz = float(max(Lz, 1e-12))
        B  = int(max(B, 8))

        s = (B / (Lx * Ly * Lz)) ** (1.0 / 3.0)

        Nx = int(round(s * Lx)) + 1
        Ny = int(round(s * Ly)) + 1
        Nz = int(round(s * Lz)) + 1

        Nx = max(2, min(Nx, max_res))
        Ny = max(2, min(Ny, max_res))
        Nz = max(2, min(Nz, max_res))

        # Optionnel: impose un min_res
        Nx = max(Nx, min_res)
        Ny = max(Ny, min_res)
        Nz = max(Nz, min_res)

        return Nx, Ny, Nz

    def _fold_pack_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        migs = self.cfg.migs
        use_fold_pack_legacy = bool(migs.get("use_fold_pack", False))
        use_folding = bool(migs.get("use_folding", use_fold_pack_legacy))
        use_packing = bool(migs.get("use_packing", use_fold_pack_legacy))

        if not use_folding:
            return xyz

        # Buffers MUST already exist (set once in init_from_tensor)
        if not (hasattr(self, "fold_x0") and hasattr(self, "fold_pack_axis") and hasattr(self, "fold_pack_delta")):
            raise RuntimeError("Folding enabled but fold buffers are missing. Call init_from_tensor() first.")

        x0 = float(self.fold_x0.item())
        axis = int(self.fold_pack_axis.item())
        delta = float(self.fold_pack_delta.item())

        left = (xyz[:, 0] < x0)
        xyz2 = xyz.clone()
        xyz2[left, 0] = x0 + (x0 - xyz2[left, 0])

        if use_packing:
            xyz2[left, axis] = xyz2[left, axis] + delta

        return xyz2


    def _save_folding_visualization(self, xyz_before: torch.Tensor, xyz_after: torch.Tensor):
        """
        Saves npz + 2D projections (XY, XZ, YZ) + histograms into exp_dir/folding_vis.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = os.path.join(self.fold_vis_dir, "init")  # ou timestamp si tu veux
        os.makedirs(out_dir, exist_ok=True)

        xb = xyz_before.detach().cpu().numpy()
        xa = xyz_after.detach().cpu().numpy()

        # 1) Save raw data
        np.savez_compressed(
            os.path.join(out_dir, "folding_xyz.npz"),
            xyz_before=xb,
            xyz_after=xa,
            fold_x0=float(self.fold_x0.item()),
            fold_pack_axis=int(self.fold_pack_axis.item()),
            fold_pack_delta=float(self.fold_pack_delta.item()),
            use_folding=bool(self.cfg.migs.get("use_folding", self.cfg.migs.get("use_fold_pack", False))),
            use_packing=bool(self.cfg.migs.get("use_packing", self.cfg.migs.get("use_fold_pack", False))),
        )

        def scatter2d(a, b, title, fname):
            plt.figure(figsize=(7, 7))
            plt.scatter(a[:, 0], a[:, 1], s=1, alpha=0.35, label="before")
            plt.scatter(b[:, 0], b[:, 1], s=1, alpha=0.35, label="after")
            plt.legend()
            plt.title(title)
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=200)
            plt.close()

        # XY
        scatter2d(xb[:, [0,1]], xa[:, [0,1]], "Folding: XY (x,y)", "fold_xy.png")
        # XZ
        scatter2d(xb[:, [0,2]], xa[:, [0,2]], "Folding: XZ (x,z)", "fold_xz.png")
        # YZ
        scatter2d(xb[:, [1,2]], xa[:, [1,2]], "Folding: YZ (y,z)", "fold_yz.png")

        # X histogram
        plt.figure(figsize=(7, 4))
        plt.hist(xb[:, 0], bins=200, alpha=0.5, label="before")
        plt.hist(xa[:, 0], bins=200, alpha=0.5, label="after")
        plt.axvline(float(self.fold_x0.item()), linestyle="--", linewidth=2, label="fold_x0")
        plt.legend()
        plt.title("Folding: X distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fold_x_hist.png"), dpi=200)
        plt.close()

        print(f"[FOLD-VIS] saved in: {out_dir}")


    # =====================================================
    # INITIALIZATION WITH GEOMETRIC VOXEL GRID
    # =====================================================

    def init_from_tensor(self, gaussian_model):
        with torch.no_grad():
            """Build TT from a regular 3D voxel grid (no Hilbert, no arbitrary reshape)."""
            
            if hasattr(self.cfg, 'migs') and getattr(self.cfg.migs, "skip_init_from_tensor", False):
                print("[TT-Grid] skip_init_from_tensor=True, skipping")
                return
            
            G = gaussian_model._xyz.shape[0]
            print(f"[TT-Grid] Initializing from {G} Gaussians using voxel grid")
            
            # ─────────────────────────────────────────────────────
            # STEP 1: Assemble (G, M=43)
            # ─────────────────────────────────────────────────────
            # xyz = gaussian_model._xyz
            # scaling = gaussian_model._scaling
            # rotation = gaussian_model._rotation
            # features_dc = gaussian_model._features_dc.squeeze(-1)
            # features_rest = gaussian_model._features_rest.squeeze(-1)
            # opacity = gaussian_model._opacity
            xyz = gaussian_model._xyz.detach()
            scaling = gaussian_model._scaling.detach()
            rotation = gaussian_model._rotation.detach()
            features_dc = gaussian_model._features_dc.squeeze(-1).detach()
            features_rest = gaussian_model._features_rest.squeeze(-1).detach()
            opacity = gaussian_model._opacity.detach()


            # ----------------------------
            # Fold+Pack parameters (buffers)
            # ----------------------------
            migs = self.cfg.migs
            use_fold_pack_legacy = bool(migs.get("use_fold_pack", False))
            use_folding = bool(migs.get("use_folding", use_fold_pack_legacy))

            if use_folding:
                x0 = 0.5 * (xyz[:, 0].min() + xyz[:, 0].max())
                pack_axis = int(migs.get("fold_pack_axis", 1))
                gap_ratio = float(migs.get("fold_pack_gap_ratio", 0.05))

                extent = (xyz[:, pack_axis].max() - xyz[:, pack_axis].min()).clamp(min=1e-6)
                delta = extent * (1.0 + gap_ratio)

                if not hasattr(self, "fold_x0"):
                    self.register_buffer("fold_x0", x0.detach().view(()))
                    self.register_buffer("fold_pack_axis", torch.tensor(pack_axis, device=xyz.device, dtype=torch.int64))
                    self.register_buffer("fold_pack_delta", delta.detach().view(()))


            
            #W_GM = torch.cat([xyz, scaling, rotation, features_dc, features_rest, opacity], dim=1)
            W_GM = torch.cat([scaling, rotation, features_dc, features_rest, opacity], dim=1)
            M = W_GM.shape[1]
            device = W_GM.device
            W_prior = W_GM.mean(dim=0)  # (M,)
            W_prior = W_prior.clone()
            opacity_idx = M - 1
            W_prior[opacity_idx] = float(self.cfg.migs.get("empty_opacity_logit", -8.0))

            self.register_buffer("W_prior", W_prior.detach())# (M,)

            # Apply fold+pack to xyz# ----------------------------
            # ----------------------------
            xyz_grid = self._fold_pack_xyz(xyz) 
            if bool(self.cfg.migs.get("save_foldpack_vis", False)):
                self._save_folding_visualization(xyz, xyz_grid)

            if bool(self.cfg.migs.get("save_foldpack_vis", False)):
                out_dir = os.path.join(self.save_dir, "foldpack_debug")
                os.makedirs(out_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(out_dir, "foldpack_xyz.npz"),
                    xyz_before=xyz.detach().cpu().numpy(),
                    xyz_after=xyz_grid.detach().cpu().numpy(),
                    fold_x0=float(self.fold_x0.item()),
                    fold_pack_axis=int(self.fold_pack_axis.item()),
                    fold_pack_delta=float(self.fold_pack_delta.item()),
                )
                print(f"[FoldPack] saved: {os.path.join(out_dir, 'foldpack_xyz.npz')}")
                    
                        # ─────────────────────────────────────────────────────
            # STEP 2: Bounding box
            # ─────────────────────────────────────────────────────
            coord_min = xyz_grid.min(0).values
            coord_max = xyz_grid.max(0).values

                        
            padding_ratio = self.cfg.migs.get("grid_padding", 0.05)
            if not hasattr(self.cfg.migs, "grid_padding"):
                print(f"[TT-Grid] Using default grid_padding={padding_ratio}")
            
            padding = (coord_max - coord_min) * padding_ratio
            coord_min = coord_min - padding
            coord_max = coord_max + padding
            L = (coord_max - coord_min)
            Lx, Ly, Lz = L.tolist()
            
            # ─────────────────────────────────────────────────────
            # STEP 3: Grid resolution
            # ─────────────────────────────────────────────────────
            # if hasattr(self.cfg.migs, "grid_resolution") and self.cfg.migs.grid_resolution is not None:
            #     res = self.cfg.migs.grid_resolution

            #     #  Hydra/OmegaConf safe: ListConfig behaves like a list but isn't `list`
            #     if isinstance(res, (list, tuple, ListConfig)):
            #         Nx, Ny, Nz = map(int, list(res))
            #     else:
            #         Nx = Ny = Nz = int(res)
            # else:
            #     target = G // 2
            #     side = int(round(target ** (1/3)))
            #     Nx = Ny = Nz = max(16, min(side, 64))
            #     print(f"[TT-Grid] Auto-computed resolution from G={G}")
            
            # print(f"[TT-Grid] Resolution: ({Nx}, {Ny}, {Nz}) = {Nx*Ny*Nz} voxels")
            if getattr(self.cfg.migs, "grid_resolution", None) is not None:
                res = self.cfg.migs.grid_resolution
                if isinstance(res, (list, tuple, ListConfig)):
                    Nx, Ny, Nz = map(int, list(res))
                else:
                    Nx = Ny = Nz = int(res)
            else:
                B = int(self.cfg.migs.get("grid_budget", 96_000))
                min_res = int(self.cfg.migs.get("grid_min_res", 16))
                max_res = int(self.cfg.migs.get("grid_max_res", 128))
                Nx, Ny, Nz = self._compute_grid_resolution_A2(Lx, Ly, Lz, B, min_res=min_res, max_res=max_res)

            print(f"[TT-Grid] Resolution: ({Nx}, {Ny}, {Nz}) = {Nx*Ny*Nz} voxels")
            dx = Lx / max(Nx - 1, 1)
            dy = Ly / max(Ny - 1, 1)
            dz = Lz / max(Nz - 1, 1)
            print(f"[TT-Grid] voxel steps: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
            print(f"[TT-Grid] anisotropy: dy/dx={dy/(dx+1e-12):.3f}, dz/dx={dz/(dx+1e-12):.3f}")
            
            # ─────────────────────────────────────────────────────
            # STEP 4: Assign Gaussians to voxels
            # ─────────────────────────────────────────────────────
            xyz_norm = (xyz_grid - coord_min) / (coord_max - coord_min)
            xyz_norm = xyz_norm.clamp(0, 1)
            
            # i_voxel = (xyz_norm[:, 0] * (Nx - 1)).long()
            # j_voxel = (xyz_norm[:, 1] * (Ny - 1)).long()
            # k_voxel = (xyz_norm[:, 2] * (Nz - 1)).long()
            
            # # ─────────────────────────────────────────────────────
            # # STEP 5: Populate grid
            # # ─────────────────────────────────────────────────────
            # grid = torch.zeros(1, Nx, Ny, Nz, M, device=device, dtype=W_GM.dtype)
            # counts = torch.zeros(1, Nx, Ny, Nz, device=device, dtype=torch.float32)
            
            # for g in range(G):
            #     i, j, k = i_voxel[g], j_voxel[g], k_voxel[g]
            #     grid[0, i, j, k, :] += W_GM[g, :]
            #     counts[0, i, j, k] += 1.0
                    
            # mask = (counts > 0).float().unsqueeze(-1)
            # grid = grid / (counts.unsqueeze(-1) + 1e-8) * mask
            # empty_mask = (counts == 0)  # (1, Nx, Ny, Nz)
            # n_empty = empty_mask.sum().item()

            # if n_empty > 0:
            #     # Valeurs neutres (PAS -10 pour opacity, sinon interpolation foireuse)
            #     W_neutral = W_prior.clone()
            #     W_neutral[opacity_idx] = -5.0  # ← Valeur intermédiaire (ajustable)
                
            #     # Broadcast et assign
            #     empty_indices = empty_mask.squeeze(0).nonzero(as_tuple=False)  # (n_empty, 3)
            #     for idx in range(n_empty):
            #         i, j, k = empty_indices[idx]
            #         grid[0, i, j, k, :] = W_neutral
                
            #     print(f"[TT-Grid] Filled {n_empty} empty voxels with neutral values")

            # Ligne 149 : GARDER
            # empty_mask = (counts == 0)

            # # Lignes 152-161 : GARDER
            # global_mean = W_GM.mean(0)
            # global_std = W_GM.std(0).clamp(min=1e-6)
            # empty_indices = empty_mask.nonzero(as_tuple=False)
            # n_empty = empty_indices.shape[0]

            # if n_empty > 0:
            #     noise = torch.randn(n_empty, M, device=device, dtype=W_GM.dtype) * 0.1 * global_std
            #     for idx in range(n_empty):
            #         b, i, j, k = empty_indices[idx]
            #         grid[b, i, j, k, :] = global_mean + noise[idx]
            
            # occupied = (counts > 0).sum().item()
            # total = Nx * Ny * Nz
            # print(f"[TT-Grid] Occupied: {occupied}/{total} ({100*occupied/total:.1f}%)")
            # # Occupancy (binaire)
            # occ_grid = (counts > 0).float()  # (1, Nx, Ny, Nz)
            # occ = occ_grid.unsqueeze(1)            # (1,1,Nx,Ny,Nz) mais attention à l'ordre
            # occ = occ.permute(0,1,4,3,2)           # (1,1,Nz,Ny,Nx) pour conv3d/pool3d
            # k = 5
            # p = k // 2
            # occ = F.max_pool3d(occ, kernel_size=k, stride=1, padding=p)
            # occ = occ.permute(0,1,4,3,2)           # retour (1,1,Nx,Ny,Nz)
            # occ_grid = occ.squeeze(1)              # (1,Nx,Ny,Nz)
            # print("[TT-Grid] occ raw mean:", float(((counts>0).float()).mean()))
            # print("[TT-Grid] occ dilated mean:", float((occ_grid>0).float().mean()))



            # self.register_buffer("occ_grid", occ_grid.detach().unsqueeze(-1))

            # ─────────────────────────────────────────────────────
            # STEP 4–5 (NEW): Trilinear splatting into grid
            # ─────────────────────────────────────────────────────

            # 1) Coordonnées continues en indices voxel
            x = xyz_norm[:, 0] * (Nx - 1)  # (G,)
            y = xyz_norm[:, 1] * (Ny - 1)
            z = xyz_norm[:, 2] * (Nz - 1)

            # 2) Indices base (floor) et fraction
            i0 = torch.floor(x).long()
            j0 = torch.floor(y).long()
            k0 = torch.floor(z).long()

            # clamp pour sécurité (éviter i1 hors grille)
            i0 = i0.clamp(0, Nx - 2)
            j0 = j0.clamp(0, Ny - 2)
            k0 = k0.clamp(0, Nz - 2)

            i1 = i0 + 1
            j1 = j0 + 1
            k1 = k0 + 1

            fx = (x - i0.float()).clamp(0, 1)  # (G,)
            fy = (y - j0.float()).clamp(0, 1)
            fz = (z - k0.float()).clamp(0, 1)

            wx0 = 1.0 - fx
            wy0 = 1.0 - fy
            wz0 = 1.0 - fz
            wx1 = fx
            wy1 = fy
            wz1 = fz

            # 3) Les 8 coins et leurs poids (G,)
            w000 = wx0 * wy0 * wz0
            w001 = wx0 * wy0 * wz1
            w010 = wx0 * wy1 * wz0
            w011 = wx0 * wy1 * wz1
            w100 = wx1 * wy0 * wz0
            w101 = wx1 * wy0 * wz1
            w110 = wx1 * wy1 * wz0
            w111 = wx1 * wy1 * wz1

            # 4) Préparer un grid et counts en flat pour accumuler efficacement
            V = Nx * Ny * Nz
            grid_flat   = torch.zeros(V, M, device=device, dtype=W_GM.dtype)
            counts_flat = torch.zeros(V, 1, device=device, dtype=torch.float32)

            def lin_index(ii, jj, kk):
                # order: (i, j, k) with sizes (Nx, Ny, Nz)
                return (ii * (Ny * Nz) + jj * Nz + kk).long()  # (G,)

            # 5) Accumulation par index_add_ (boucle sur 8 coins, pas sur G)
            corners = [
                (i0, j0, k0, w000),
                (i0, j0, k1, w001),
                (i0, j1, k0, w010),
                (i0, j1, k1, w011),
                (i1, j0, k0, w100),
                (i1, j0, k1, w101),
                (i1, j1, k0, w110),
                (i1, j1, k1, w111),
            ]

            for ii, jj, kk, ww in corners:
                idx_lin = lin_index(ii, jj, kk)  # (G,)
                ww = ww.view(-1, 1).to(torch.float32)  # (G,1) pour counts
                # grid accumule en dtype W_GM.dtype
                grid_flat.index_add_(0, idx_lin, W_GM * ww.to(W_GM.dtype))
                # counts accumule en float
                counts_flat.index_add_(0, idx_lin, ww)

            # 6) Reshape back to grid
            grid   = grid_flat.view(1, Nx, Ny, Nz, M)
            counts = counts_flat.view(1, Nx, Ny, Nz, 1)

            # 7) Normalisation (moyenne pondérée)
            eps = 1e-8
            grid = grid / (counts.to(grid.dtype) + eps)

            # 8) Masques "vides" basés sur counts (plus robuste qu'égalité à 0)
            empty_mask = (counts <= 1e-6).squeeze(-1)  # (1,Nx,Ny,Nz)
            n_empty = empty_mask.sum().item()

            if n_empty > 0:
                W_neutral = W_prior.clone()
                W_neutral[opacity_idx] = W_prior[opacity_idx]  # valeur neutre pour opacity logit

                empty_indices = empty_mask.squeeze(0).nonzero(as_tuple=False)  # (n_empty,3)
                # assign
                for t in range(empty_indices.shape[0]):
                    i, j, k = empty_indices[t]
                    grid[0, i, j, k, :] = W_neutral

                print(f"[TT-Grid] Filled {n_empty} empty voxels with neutral values")

            # 9) Occupancy (counts > 0) et dilatation identique à ton code
            occ_grid = (counts > 1e-6).float().squeeze(-1)  # (1,Nx,Ny,Nz)
            occ = occ_grid.unsqueeze(1).permute(0,1,4,3,2)  # (1,1,Nz,Ny,Nx)
            k = 5
            p = k // 2
            occ = F.max_pool3d(occ, kernel_size=k, stride=1, padding=p)
            occ = occ.permute(0,1,4,3,2).squeeze(1)  # (1,Nx,Ny,Nz)

            self.register_buffer("occ_grid", occ.detach().unsqueeze(-1))



                    
            # ─────────────────────────────────────────────────────
            # STEP 6: TT decomposition AVANT le test
            # ─────────────────────────────────────────────────────
            self.tt_shape = (1, Nx, Ny, Nz, int(M))
            
            rank_override = self.cfg.migs.get("rank", None)
            if rank_override is not None:
                ranks_target = list(map(int, rank_override))
                print(f"[TT-Grid] Using explicit ranks: {ranks_target}")
            else:
                R = int(self.cfg.migs.get("init_rank", 64))
                ranks_target = [1, R, R, R, R, 1]
                print(f"[TT-Grid] Using init_rank={R} → ranks={ranks_target}")
            
            self.tt_rank = ranks_target
            tt_tensor = tensor_train(grid, rank=ranks_target, verbose=self.verbose)
            
            self.tt_tensor_gpu = nn.ParameterList([
                nn.Parameter(c.to(device)) for c in tt_tensor.factors
            ])
            
            # Split last core
            core4 = self.tt_tensor_gpu[4]
            #self.core4_xyz      = nn.Parameter(core4[:, 0:3, :].detach().clone())
            # self.core4_scaling  = nn.Parameter(core4[:, 3:6, :].detach().clone())
            # self.core4_rotation = nn.Parameter(core4[:, 6:10, :].detach().clone())
            # self.core4_dc       = nn.Parameter(core4[:, 10:11, :].detach().clone())
            # self.core4_rest     = nn.Parameter(core4[:, 11:42, :].detach().clone())
            # self.core4_opacity  = nn.Parameter(core4[:, 42:43, :].detach().clone())
            core4 = self.tt_tensor_gpu[4]
            self.core4_scaling  = nn.Parameter(core4[:, 0:3,   :].detach().clone())
            self.core4_rotation = nn.Parameter(core4[:, 3:7,   :].detach().clone())
            self.core4_dc       = nn.Parameter(core4[:, 7:8,   :].detach().clone())
            self.core4_rest     = nn.Parameter(core4[:, 8:39,  :].detach().clone())  # 31 dims
            self.core4_opacity  = nn.Parameter(core4[:, 39:40, :].detach().clone())

            
            # Rank expansion
            self._expand_r1_by_replication(ranks_target[1])
            self._expand_ranks_to_targets_preserve(ranks_target)
            if self.verbose:
                shapes = [tuple(c.shape) for c in self.tt_tensor_gpu[:4]]
                core4_shape = tuple(self.recombine_core4().shape)
                print("[TT-Grid] Core shapes after expand:")
                print(f"  core0: {shapes[0]}  # (1, I, r1)")
                print(f"  core1: {shapes[1]}  # (r1, Nx, r2)")
                print(f"  core2: {shapes[2]}  # (r2, Ny, r3)")
                print(f"  core3: {shapes[3]}  # (r3, Nz, r4)")
                print(f"  core4(recombined): {core4_shape}  # (r4, M=43, 1)")
                print(f"  ranks target: {self.tt_rank}")

            # ─────────────────────────────────────────────────────
            # STEP 7: Store grid metadata
            # ─────────────────────────────────────────────────────
            self.register_buffer("grid_coord_min", coord_min.detach())
            self.register_buffer("grid_coord_max", coord_max.detach())
            self.grid_resolution = (Nx, Ny, Nz)
            self.register_buffer("gaussian_xyz", xyz.detach().clone())
            
            # ─────────────────────────────────────────────────────
            # STEP 8: Export metadata (optional)
            # ─────────────────────────────────────────────────────
            snapshot_dir = os.path.join(self.save_dir, "grid_init")
            os.makedirs(snapshot_dir, exist_ok=True)
            total = Nx * Ny * Nz
            occupied = (counts > 1e-6).sum().item()  # counts est (1,Nx,Ny,Nz,1)

            np.savez_compressed(
                os.path.join(snapshot_dir, "grid_metadata.npz"),
                coord_min=coord_min.detach().cpu().numpy(),
                coord_max=coord_max.detach().cpu().numpy(),

                grid_resolution=np.array([Nx, Ny, Nz], dtype=np.int64),
                occupied_voxels=occupied,
                total_voxels=total,
                occupancy_rate=occupied/total
            )
            
            # ─────────────────────────────────────────────────────
            # STEP 9: Test reconstruction  APRÈS la décomposition
            # ─────────────────────────────────────────────────────
            if self.verbose:
                print("[TT-Grid] Testing reconstruction...")
                W_rec = self.get_W_for_identity(0)  # Uses stored gaussian_xyz
                
                print(f"  Original: {W_GM.shape}")
                print(f"  Reconstructed: {W_rec.shape}")
                
                # Import diagnostic functions
                # from utils.migs_utils import (
                #     compare_reconstruction_per_block,
                #     plot_correlation_across_parameters,
                #     plot_pca_groupwise_xyz_auto
                # )
                
                # compare_reconstruction_per_block(
                #     W_GM, W_rec,
                #     # split_sizes=[3, 3, 4, 1, 31, 1],
                #     # names=['xyz', 'scaling', 'rotation', 'dc', 'rest', 'opacity']
                #     split_sizes=[3,4,1,31,1],
                #     names=['scaling','rotation','dc','rest','opacity']

                # )
                
                # plot_correlation_across_parameters(W_GM, W_rec)
                # plot_pca_groupwise_xyz_auto(W_GM, W_rec, num_groups=10)

            
            print(f"[TT-Grid]  Init complete. Grid: ({Nx},{Ny},{Nz}), Ranks: {self.tt_rank}")

            if bool(self.cfg.migs.get("save_foldpack_vis", False)):
                stats_dir = os.path.join(self.fold_vis_dir, "init")
                os.makedirs(stats_dir, exist_ok=True)

                total = int(Nx * Ny * Nz)
                occupied = int((counts > 1e-6).sum().item())  # counts: (1,Nx,Ny,Nz,1)
                empty = total - occupied

                np.savez_compressed(
                    os.path.join(stats_dir, "grid_stats_after_folding.npz"),
                    Nx=Nx, Ny=Ny, Nz=Nz,
                    total_voxels=total,
                    occupied_voxels=occupied,
                    empty_voxels=empty,
                    occupancy_rate=float(occupied / max(total, 1)),
                    coord_min=coord_min.detach().cpu().numpy(),
                    coord_max=coord_max.detach().cpu().numpy(),
                )

                print(f"[GRID-STATS] occupied={occupied}/{total} ({100*occupied/max(total,1):.2f}%), empty={empty}")
                print(f"[GRID-STATS] saved: {os.path.join(stats_dir, 'grid_stats_after_folding.npz')}")

            
        # =====================================================
        # RECONSTRUCTION: Sample grid at query points
        # =====================================================

    def get_W_for_identity(self, idx: int, xyz_query: torch.Tensor = None) -> torch.Tensor:
        """
        Sample parameters from TT grid for query points.
        
        Args:
            idx: identity index
            xyz_query: (N, 3) query positions. If None, uses stored gaussian_xyz.
        
        Returns:
            (N, M=40) parameters
        """
        
        # Cache management
        if self.training:
            cores = self.get_tt_tensor(idx)
            grid_full = tt_to_tensor(cores).squeeze(0)
        else:
            if idx not in self._grid_cache:
                cores = self.get_tt_tensor(idx)
                grid_reconstructed = tt_to_tensor(cores).squeeze(0)
                self._grid_cache[idx] = grid_reconstructed.detach()
            
            grid_full = self._grid_cache[idx]
        
        # Query points
        if xyz_query is None:
            if not hasattr(self, "gaussian_xyz"):
                raise RuntimeError("No query points provided and no stored gaussian_xyz")
            xyz_query = self.gaussian_xyz
        
        xyz_query = xyz_query.detach()
        
        # Normalize to [-1, 1]
        xyz_query_mapped = self._fold_pack_xyz(xyz_query)

        xyz_norm = (xyz_query_mapped - self.grid_coord_min) / (self.grid_coord_max - self.grid_coord_min)

        xyz_norm = xyz_norm * 2 - 1
        xyz_norm = xyz_norm.clamp(-1, 1)
        
        N_pts = xyz_query.shape[0]
        M = grid_full.shape[-1]
        opacity_idx_grid = M - 1

        # Create xyz_for_sample
        xyz_for_sample = xyz_norm.view(1, 1, 1, N_pts, 3)  # (1, 1, 1, N, 3)

        # Séparer features et opacity
        grid_features = grid_full[..., :opacity_idx_grid]  # (Nx, Ny, Nz, M-1)
        grid_opacity = grid_full[..., opacity_idx_grid:opacity_idx_grid+1]  # (Nx, Ny, Nz, 1)

        # Grid sample FEATURES avec bilinear (smooth)
        features_sampled = F.grid_sample(
            grid_features.permute(3, 2, 1, 0).unsqueeze(0),  # (1, M-1, Nz, Ny, Nx)
            xyz_for_sample,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (1, M-1, 1, 1, N)

        # Grid sample OPACITY avec nearest (sharp)
        opacity_sampled = F.grid_sample(
            grid_opacity.permute(3, 2, 1, 0).unsqueeze(0),  # (1, 1, Nz, Ny, Nx)
            xyz_for_sample,
            mode='nearest',
            padding_mode='border',
            align_corners=True
        )  # (1, 1, 1, 1, N)

        # ✅ CORRECTION : Reshape direct (plus robuste)
        features_sampled = features_sampled.view(M-1, N_pts).permute(1, 0)  # (N, M-1)
        opacity_sampled = opacity_sampled.view(1, N_pts).permute(1, 0)      # (N, 1)

        # Recombiner
        sampled = torch.cat([features_sampled, opacity_sampled], dim=1)  # (N, M)

        # Debug prints
        if self.verbose:
            print(f"[DEBUG SAMPLE] Opacity: min={sampled[:, -1].min():.2f}, "
                f"max={sampled[:, -1].max():.2f}, mean={sampled[:, -1].mean():.2f}")
            halo_count = ((sampled[:, -1] > -2) & (sampled[:, -1] < 0)).sum().item()
            print(f"[DEBUG SAMPLE] Halo range [-2, 0]: {halo_count}/{N_pts} points")

        # Occupancy sample
        occ_full = self.occ_grid.squeeze(0)  # (Nx, Ny, Nz, 1)
        occ_for_sample = occ_full.permute(3, 2, 1, 0).unsqueeze(0)  # (1, 1, Nz, Ny, Nx)

        occ_sampled = F.grid_sample(
            occ_for_sample,
            xyz_for_sample,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True
        )
        # ✅ CORRECTION : Reshape direct
        occ_sampled = occ_sampled.view(N_pts, 1)  # (N, 1)

        if self.verbose:
            print("[TT-Grid] occ_sampled mean/min/max:",
                float(occ_sampled.mean()), float(occ_sampled.min()), float(occ_sampled.max()))

        # Blend opacity with prior
        W_prior = self.W_prior.view(1, -1)  # (1, M)
        opacity_idx = sampled.shape[1] - 1

        opacity_old = sampled[:, opacity_idx:opacity_idx+1]

        # 1) binariser l’occupation (tue le halo)
        thr = float(self.cfg.migs.get("occ_threshold", 0.5))  # tu peux tester 0.3 / 0.7
        occ_hard = (occ_sampled > thr).to(occ_sampled.dtype)

        # 2) mélange dur : soit plein, soit vide
        opacity_new = occ_hard * opacity_old + (1.0 - occ_hard) * W_prior[:, opacity_idx:opacity_idx+1]


        # Rebuild sampled
        sampled = torch.cat(
            [sampled[:, :opacity_idx], opacity_new, sampled[:, opacity_idx+1:]],
            dim=1
        )

        return sampled

    # =====================================================
    # RESTE DU CODE (inchangé)
    # =====================================================

    @torch.no_grad()
    def _zero_pad_pair_preserve(self, left, right, add, dim_left, dim_right):
        if add <= 0:
            return left, right
        dev = left.device
        dl_shape = list(left.shape);  dl_shape[dim_left]  = add
        dr_shape = list(right.shape); dr_shape[dim_right] = add
        
        left_std  = left.detach().std()
        right_std = right.detach().std()
        if not torch.isfinite(left_std) or left_std < 1e-8:
            left_std = left.detach().abs().mean()
        if not torch.isfinite(right_std) or right_std < 1e-8:
            right_std = right.detach().abs().mean()
        
        scale = 1e-2
        pad_left  = scale * left_std  * torch.randn(dl_shape, device=dev, dtype=left.dtype)
        pad_right = scale * right_std * torch.randn(dr_shape, device=dev, dtype=right.dtype)
        
        new_left  = torch.cat([left,  pad_left],  dim=dim_left)
        new_right = torch.cat([right, pad_right], dim=dim_right)
        return new_left, new_right

    @torch.no_grad()
    def _expand_ranks_to_targets_preserve(self, ranks_target):
        c0 = self.tt_tensor_gpu[0]
        c1 = self.tt_tensor_gpu[1]
        c2 = self.tt_tensor_gpu[2]
        c3 = self.tt_tensor_gpu[3]
        
        # r2
        r2_cur = c1.shape[2]
        r2_tgt = int(ranks_target[2])
        if r2_tgt > r2_cur:
            add = r2_tgt - r2_cur
            new_c1, new_c2 = self._zero_pad_pair_preserve(c1, c2, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[1] = nn.Parameter(new_c1)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)
        
        c1 = self.tt_tensor_gpu[1]
        c2 = self.tt_tensor_gpu[2]
        c3 = self.tt_tensor_gpu[3]
        
        # r3
        r3_cur = c2.shape[2]
        r3_tgt = int(ranks_target[3])
        if r3_tgt > r3_cur:
            add = r3_tgt - r3_cur
            new_c2, new_c3 = self._zero_pad_pair_preserve(c2, c3, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)
            self.tt_tensor_gpu[3] = nn.Parameter(new_c3)
        
        c2 = self.tt_tensor_gpu[2]
        c3 = self.tt_tensor_gpu[3]
        
        # r4
        r4_cur = c3.shape[2]
        r4_tgt = int(ranks_target[4])
        if r4_tgt > r4_cur:
            add = r4_tgt - r4_cur
            core4_full = self.recombine_core4()
            new_c3, new_core4 = self._zero_pad_pair_preserve(c3, core4_full, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[3] = nn.Parameter(new_c3)
            
            r4n, M, r5 = new_core4.shape
            assert r5 == 1
            #self.core4_xyz      = nn.Parameter(new_core4[:, 0:3,   :])
            # self.core4_scaling  = nn.Parameter(new_core4[:, 3:6,   :])
            # self.core4_rotation = nn.Parameter(new_core4[:, 6:10,  :])
            # self.core4_dc       = nn.Parameter(new_core4[:, 10:11, :])
            # self.core4_rest     = nn.Parameter(new_core4[:, 11:42, :])
            # self.core4_opacity  = nn.Parameter(new_core4[:, 42:43, :])
            self.core4_scaling  = nn.Parameter(new_core4[:, 0:3,   :])
            self.core4_rotation = nn.Parameter(new_core4[:, 3:7,   :])
            self.core4_dc       = nn.Parameter(new_core4[:, 7:8,   :])
            self.core4_rest     = nn.Parameter(new_core4[:, 8:39,  :])
            self.core4_opacity  = nn.Parameter(new_core4[:, 39:40, :])

        
        self._needs_opt_rebuild = True

    @torch.no_grad()
    def _expand_r1_by_replication(self, r1_target):
        c0 = self.tt_tensor_gpu[0]
        c1 = self.tt_tensor_gpu[1]
        
        r1_cur = c0.shape[2]
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
        c0_new = _repeat_to(c0, dim=2, target=r1_target) * scale
        c1_new = _repeat_to(c1, dim=0, target=r1_target)
        
        self.tt_tensor_gpu[0] = nn.Parameter(c0_new)
        self.tt_tensor_gpu[1] = nn.Parameter(c1_new)
        self._needs_opt_rebuild = True

    def recombine_core4(self):
        # return torch.cat([
        #     self.core4_xyz, self.core4_scaling, self.core4_rotation,
        #     self.core4_dc, self.core4_rest, self.core4_opacity
        # ], dim=1)
        return torch.cat([
            self.core4_scaling,
            self.core4_rotation,
            self.core4_dc,
            self.core4_rest,
            self.core4_opacity
        ], dim=1)


    def get_core0(self, idx):
        assert 0 <= idx < self.tt_tensor_gpu[0].shape[1]
        return self.tt_tensor_gpu[0][:, idx:idx+1, :]

    def expand_first_core(self, n_identities):
        if not len(self.tt_tensor_gpu):
            raise RuntimeError("TT cores must be initialized")
        first = self.tt_tensor_gpu[0]
        r0, n_cur, r1 = first.shape
        if n_cur >= n_identities:
            return
        base = first[:, 0:1, :].detach()
        rep = base.repeat(1, n_identities, 1)
        noise = self._randn_like(rep, tag="core0_expand_noise") * 1e-3
        new = rep + noise
        self.tt_tensor_gpu[0] = nn.Parameter(new)
        self._needs_opt_rebuild = True
        if self.optimizer is not None:
            self._rebuild_optimizer_like_before()

    def get_tt_tensor(self, idx=None):
        core0 = self.get_core0(idx) if idx is not None else self.tt_tensor_gpu[0]
        return [core0, self.tt_tensor_gpu[1], self.tt_tensor_gpu[2], 
                self.tt_tensor_gpu[3], self.recombine_core4()]

    def optimize_parameters(self):
        return list(self.tt_tensor_gpu[:4]) + [
            self.core4_scaling, self.core4_rotation,
            self.core4_dc, self.core4_rest, self.core4_opacity
        ]

    def freeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = False

    def unfreeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = True

    def set_optimizer(self, opt_cfg):
        """Create optimizer and LR scheduler for TT cores and core4 slices."""
        self._opt_cfg = dict(opt_cfg) if opt_cfg is not None else {}
        tt_lrs = self._opt_cfg.get("tt_lrs", [1.6e-4] * 4)
        tt_final_lrs = self._opt_cfg.get("tt_final_lrs", [1.6e-6] * 4)
        tt_decay_iters = int(self._opt_cfg.get("tt_decay_iters", 50000))

        param_groups = []
        decayed_group_indices = []  # indices of groups that should decay

        # cores 0..3 with decay (assumes tt_tensor_gpu has 5 cores total including last split core)
        for i in range(len(self.tt_tensor_gpu) - 1):
            lr_init  = float(tt_lrs[i])
            lr_final = float(tt_final_lrs[i])
            param_groups.append({
                "params": [self.tt_tensor_gpu[i]],
                "lr": lr_init,
                "initial_lr": lr_init,
                "final_lr": lr_final,
            })
            decayed_group_indices.append(len(param_groups) - 1)

        # xyz slice with decay
        # param_groups.append({
        #     "params": [self.core4_xyz],
        #     "lr": 1.6e-4, "initial_lr": 1.6e-4, "final_lr": 1.6e-6
        # })
        #decayed_group_indices.append(len(param_groups) - 1)

        # other slices fixed LR (NO decay)
        param_groups += [
            {"params": [self.core4_scaling],  "lr": 5e-3},
            {"params": [self.core4_rotation], "lr": 1e-3},
            {"params": [self.core4_dc],       "lr": 2.5e-3},
            {"params": [self.core4_rest],     "lr": 2.5e-3},
            {"params": [self.core4_opacity],  "lr": 5e-2},
        ]

        self.optimizer = torch.optim.Adam(param_groups)

        # ---- Scheduler: decay ONLY decayed groups, keep others at 1.0 ----
        if len(decayed_group_indices) > 0:
            # Use exponential schedule: lr(t)=lr0 * gamma^t, where gamma matches (final/init)^(1/T)
            # We assume all decayed groups share the same ratio; if you want per-group ratios, we can do that too.
            # Protect against weird config
            lr0 = float(param_groups[decayed_group_indices[0]].get("initial_lr", param_groups[0]["lr"]))
            lrf = float(param_groups[decayed_group_indices[0]].get("final_lr", lr0))
            lrf = max(lrf, 1e-20)
            lr0 = max(lr0, 1e-20)

            gamma = (lrf / lr0) ** (1.0 / max(tt_decay_iters, 1))

            def make_lambda_for_group(g_idx: int):
                if g_idx in decayed_group_indices:
                    return lambda step: gamma ** step
                else:
                    return lambda step: 1.0

            lr_lambdas = [make_lambda_for_group(i) for i in range(len(self.optimizer.param_groups))]
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambdas)
        else:
            self.scheduler = None

        self._needs_opt_rebuild = False





    def _rebuild_optimizer_like_before(self):
        self.set_optimizer(self._opt_cfg)

    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, iteration=None):
        # Finetune fast-path
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
            
            #  AJOUT : Vider le cache après modification des cores
            self._grid_cache.clear()
            
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
            eps     = self._randn_like(mu, tag="add_identity_noise").expand_as(mu)  # même shape
            new_row = mu + noise_scale * sig * eps
            norms = U.view(n_id, r1).norm(dim=1)
            target_norm = norms.median()
            cur_norm = new_row.view(-1).norm()
            new_row = new_row / (cur_norm + 1e-8) * float(target_norm)
        else:
            new_row = self._randn_like(core0[:, :1, :], tag="add_identity_boot") * 0.02
        self.tt_tensor_gpu[0] = nn.Parameter(torch.cat([core0, new_row], dim=1))
        self._grid_cache.clear()
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