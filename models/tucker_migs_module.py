import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import tucker
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import math
tl.set_backend("pytorch")


class TuckerMIGSModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        migs_cfg = cfg["migs"] if isinstance(cfg, dict) else cfg.migs

        self.tucker_rank = migs_cfg.get("rank", [10,10,10])  # [R1_param, R2_id, R3_gaussian]
        self.verbose = migs_cfg.get("verbose", 0)

        if "dataset" in cfg and "names" in cfg["dataset"]:
            self.n_identities = len(cfg["dataset"]["names"])
        else:
            raise ValueError("Missing dataset.names")

        self.n_gaussians = cfg.model.gaussian.get("n_gaussians", 50000)
        self.n_params = 43
        self.delay = cfg.model.gaussian.get("delay", 0)

        # === Tucker factors ===
        r1, r2, r3 = self.tucker_rank

        # split U1 by block (same as in CPMIGS)
        self.U1_xyz      = nn.Parameter(torch.zeros(3, r1))
        self.U1_scaling  = nn.Parameter(torch.zeros(3, r1))
        self.U1_rotation = nn.Parameter(torch.zeros(4, r1))
        self.U1_dc       = nn.Parameter(torch.zeros(1, r1))
        self.U1_rest     = nn.Parameter(torch.zeros(31, r1))
        self.U1_opacity  = nn.Parameter(torch.zeros(1, r1))

        # Identity and Gaussian factors
        self.U2 = nn.Parameter(torch.zeros(self.n_identities, r2))
        self.U3 = nn.Parameter(torch.zeros(self.n_gaussians, r3))

        # Tucker core
        self.core = nn.Parameter(torch.zeros(r2, r3, r1))  

        self.optimizer = None
        self.scheduler = None

    def get_U1(self):
        return torch.cat([
            self.U1_xyz,
            self.U1_scaling,
            self.U1_rotation,
            self.U1_dc,
            self.U1_rest,
            self.U1_opacity
        ], dim=0)  # (43, r1)

    @staticmethod
    def compare_reconstruction_per_block(W_original: torch.Tensor, W_reconstructed: torch.Tensor):
        split_sizes = [3, 3, 4, 1, 31, 1]
        names = ['xyz', 'scaling', 'rotation', 'dc', 'rest', 'opacity']

        orig_parts = torch.split(W_original, split_sizes, dim=1)
        recon_parts = torch.split(W_reconstructed, split_sizes, dim=1)

        print("🔎 [Tucker DIAGNOSTIC] Block-wise reconstruction errors:")
        for name, orig, recon in zip(names, orig_parts, recon_parts):
            mse = ((orig - recon) ** 2).mean().item()
            l2 = torch.norm(orig - recon).item()
            print(f"    → {name.upper()} | MSE: {mse:.6e} | L2 Norm: {l2:.6e}")


    # def init_from_tensor(self, gaussian_model):
    #     G = gaussian_model._xyz.shape[0]

    #     xyz = gaussian_model._xyz
    #     scaling = gaussian_model._scaling
    #     rotation = gaussian_model._rotation
    #     features_dc = gaussian_model._features_dc.squeeze(-1)
    #     features_rest = gaussian_model._features_rest.squeeze(-1)
    #     opacity = gaussian_model._opacity

    #     all_params = [xyz, scaling, rotation, features_dc, features_rest, opacity]
    #     W_GM = torch.cat([x if x.ndim == 2 else x.view(x.shape[0], -1) for x in all_params], dim=1)
    #     W_identity = W_GM.unsqueeze(0)  # shape: (1, G, P)
    #     device = W_identity.device
    #     W_original = W_identity.squeeze(0).to(device)
    #     print(f"[TUCKER] Input tensor shape: {W_identity.shape}")  # (1, G, P)

    #     # === Tucker decomposition ===
    #     core, [U2, U3, U1] = tucker(W_identity, rank=self.tucker_rank, verbose=self.verbose)

    #     print(f"[TUCKER] Decomposition shapes:")
    #     print(f"         core: {core.shape}")         # Should be (1, R2, R3, R1)
    #     print(f"         U2 (identity): {U2.shape}")   # (1, R2)
    #     print(f"         U3 (gaussians): {U3.shape}")  # (G, R3)
    #     print(f"         U1 (params): {U1.shape}")     # (P, R1)
        
    #     dc_dim = features_dc.shape[1] if features_dc.ndim > 1 else 1
    #     dc_end = 10 + dc_dim
    #     self.U1_xyz = nn.Parameter(U1[0:3].detach())
    #     self.U1_scaling = nn.Parameter(U1[3:6].detach())
    #     self.U1_rotation = nn.Parameter(U1[6:10].detach())
    #     self.U1_dc = nn.Parameter(U1[10:dc_end].detach())
    #     self.U1_rest = nn.Parameter(U1[dc_end:-1].detach())
    #     self.U1_opacity = nn.Parameter(U1[-1:].detach())
    #     self.U2 = nn.Parameter(U2.detach())
    #     self.U3 = nn.Parameter(U3.detach())
    #     self.core = nn.Parameter(core.squeeze(0).detach())
    #     W_reconstructed = self.get_W_for_identity(0).to(device) 
    #     TuckerMIGSModule.compare_reconstruction_per_block(W_original, W_reconstructed)

    #     self.freeze_tucker_parameters()
    def init_from_tensor(self, gaussian_model):
        # ---- Récupération et concat des blocs → W_GM ∈ ℝ^{G×P} ----
        xyz = gaussian_model._xyz
        scaling = gaussian_model._scaling
        rotation = gaussian_model._rotation
        features_dc = gaussian_model._features_dc.squeeze(-1)
        features_rest = gaussian_model._features_rest.squeeze(-1)
        opacity = gaussian_model._opacity

        all_params = [xyz, scaling, rotation, features_dc, features_rest, opacity]
        W_GM = torch.cat(
            [x if x.ndim == 2 else x.view(x.shape[0], -1) for x in all_params],
            dim=1
        )  # (G, P)
        G, P = W_GM.shape
        device, dtype = W_GM.device, W_GM.dtype
        print(f"[TUCKER] Input matrix W_GM shape: {W_GM.shape}")  # (G, P)

        # ---- Rangs (ta convention : [R1_param, R2_id, R3_gaussian]) ----
        r1, r2, r3 = self.tucker_rank

        # ---- SVD compacte torch sur (G×P) : W_GM = U Σ V^T ----
        # 100% torch, pas de SciPy, pas d’allocation géante.
        U, S, Vh = torch.linalg.svd(W_GM, full_matrices=False)  # U:(G,P), S:(P,), Vh:(P,P)

        # Facteurs Tucker pour gaussiens et paramètres
        U3 = U[:, :r3].contiguous()                           # (G, r3)
        U1 = Vh.transpose(0, 1)[:, :r1].contiguous()          # (P, r1)

        # ---- Cœur : core[0] = U3^T @ W_GM @ U1 ; core ∈ ℝ^{r2×r3×r1} ----
        core = torch.zeros(r2, r3, r1, device=device, dtype=dtype)
        core0 = (U3.transpose(0, 1) @ W_GM) @ U1               # (r3, r1)
        core[0] = core0

        # ---- Identité : U2 one-hot initial (1×r2) ----
        U2 = torch.zeros(1, r2, device=device, dtype=dtype)
        U2[0, 0] = 1.0

        # ---- Découpage sémantique de U1 (P×r1) ----
        dc_dim = features_dc.shape[1] if features_dc.ndim > 1 else 1
        dc_end = 10 + dc_dim
        self.U1_xyz      = nn.Parameter(U1[0:3].detach())          # (3, r1)
        self.U1_scaling  = nn.Parameter(U1[3:6].detach())          # (3, r1)
        self.U1_rotation = nn.Parameter(U1[6:10].detach())         # (4, r1)
        self.U1_dc       = nn.Parameter(U1[10:dc_end].detach())    # (dc_dim, r1)
        self.U1_rest     = nn.Parameter(U1[dc_end:-1].detach())    # (11, r1)
        self.U1_opacity  = nn.Parameter(U1[-1:].detach())          # (1, r1)

        # ---- Enregistrer U2/U3/core comme paramètres entraînables ----
        self.U2   = nn.Parameter(U2.detach())     # (1, r2) au départ
        self.U3   = nn.Parameter(U3.detach())     # (G, r3)
        self.core = nn.Parameter(core.detach())   # (r2, r3, r1)
        print("shape of U1 is : ", U1.shape)
        print("shape of U2 is : ", self.U2.shape)
        print("shape of U3 is : ", self.U3.shape)
        print("shape of core is : ", self.core.shape)
        # ---- Sanity check reconstruction (identité 0) ----
        with torch.no_grad():
            W_original = W_GM
            W_reconstructed = self.get_W_for_identity(0).to(device)  # (G, P)
            TuckerMIGSModule.compare_reconstruction_per_block(W_original, W_reconstructed)
            TuckerMIGSModule.plot_pca_groupwise_xyz_auto(W_original, W_reconstructed, num_groups=10)

        # (Optionnel) geler au début selon ta stratégie
        self.freeze_tucker_parameters()

    @staticmethod
    def plot_pca_groupwise_xyz_auto(W_original, W_reconstructed, num_groups=10):
        """
        Compare XYZ embeddings (original vs reconstructed) in PCA space,
        coloring gaussians by cluster group (unsupervised, arbitrary) using KMeans.

        Args:
        - W_original: tensor [N, D] of original latent parameters.
        - W_reconstructed: tensor [N, D] of reconstructed latent parameters.
        - num_groups: number of clusters/groups to assign (default: 10)
        """
        os.makedirs("diagnostic", exist_ok=True)

        # Extract XYZ positions
        xyz_orig = W_original[:, :3].detach().cpu().numpy()
        xyz_recon = W_reconstructed[:, :3].detach().cpu().numpy()

        # Cluster the original positions (you could also try using recon instead, or both)
        part_labels = KMeans(n_clusters=num_groups, random_state=0).fit_predict(xyz_orig)

        # PCA projections
        pca_orig = PCA(n_components=2).fit_transform(xyz_orig)
        pca_recon = PCA(n_components=2).fit_transform(xyz_recon)

        # Colormap
        cmap = plt.cm.get_cmap("tab20", num_groups)

        # Plot side by side
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        for i in range(num_groups):
            mask = part_labels == i
            plt.scatter(pca_orig[mask, 0], pca_orig[mask, 1], alpha=0.6, s=10, color=cmap(i))
        plt.title("Original XYZ PCA (Colored by Cluster)")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        for i in range(num_groups):
            mask = part_labels == i
            plt.scatter(pca_recon[mask, 0], pca_recon[mask, 1], alpha=0.6, s=10, color=cmap(i))
        plt.title("Reconstructed XYZ PCA (Colored by Cluster)")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("diagnostic/pca_groupwise_xyz_auto.png")
        plt.close()



    def reconstruct_W(self):
        U1 = self.get_U1()
        return tl.tucker_to_tensor((self.core, [self.U2, self.U3, U1]))  # shape: (I, G, P)

    def get_W_for_identity(self, idx):
        U2_i = self.U2[idx:idx+1]  # (1, R2)
        U1 = self.get_U1()
        W_i = tl.tucker_to_tensor((self.core, [U2_i, self.U3, U1]))  # (1, G, P)
        return W_i.squeeze(0)

    def freeze_tucker_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = False

    def unfreeze_tucker_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = True

    def optimize_parameters(self):
        return [
            self.U1_xyz,
            self.U1_scaling,
            self.U1_rotation,
            self.U1_dc,
            self.U1_rest,
            self.U1_opacity,
            self.U2,
            self.U3,
            self.core
        ]

    def set_optimizer(self, opt_cfg):
        lr_pos = opt_cfg.get("position_lr_init", 1.6e-4)
        lr_pos_final = opt_cfg.get("position_lr_final", 1.6e-6)
        iters = opt_cfg.get("iterations", 50000)

        self.optim_groups = [
            {"params": [self.U1_xyz, self.U2, self.U3, self.core], "lr": lr_pos, "initial_lr": lr_pos, "final_lr": lr_pos_final},
            {"params": [self.U1_scaling], "lr": 5e-3},
            {"params": [self.U1_rotation], "lr": 1e-3},
            {"params": [self.U1_dc],      "lr": 2.5e-3},
            {"params": [self.U1_rest],    "lr": 2.5e-3},
            {"params": [self.U1_opacity], "lr": 5e-2}
        ]

        self.optimizer = torch.optim.Adam(self.optim_groups)

        decay_groups = [g for g in self.optim_groups if "final_lr" in g]
        if decay_groups:
            gamma = (lr_pos_final / lr_pos) ** (1. / iters)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        else:
            self.scheduler = None

    def step(self, iteration=None):
        if self.optimizer is None:
            return

        if iteration is not None:
            if iteration < self.delay:
                self.freeze_tucker_parameters()
                if iteration == self.delay - 1:
                    print(f"[TUCKER] frozen until iter {iteration}")
                return
            elif iteration == self.delay:
                self.unfreeze_tucker_parameters()
                print(f"[TUCKER] unfrozen at iter {iteration}")

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_learning_rate()

    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def add_identity(self):
        new_row = torch.randn(1, self.U2.shape[1], device=self.U2.device) * 0.01
        self.U2 = nn.Parameter(torch.cat([self.U2, new_row], dim=0))
        return self.U2.shape[0] - 1

    def expand_U2(self, n_identities):
        """Duplicate the first U2 row exactly for all identities (no noise)."""
        assert self.U2 is not None, "Call init_from_tensor first"
        base = self.U2.detach()[0].unsqueeze(0)  # shape (1, R)
        new_U2 = base.repeat(n_identities, 1)  # Pure duplication
        self.U2 = nn.Parameter(new_U2)
        print("✔️ expand_U2: new U2 shape =", self.U2.shape)
