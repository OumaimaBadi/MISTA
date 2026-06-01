import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# Ensure PyTorch backend is used for tensorly
tl.set_backend('pytorch')

class TTUltraMIGSModule(nn.Module):
    """
    TTUltraMIGSModule performs Tensor Train decomposition on a tensor of Gaussian parameters.
    The tensor is reshaped and factorized into TT cores that are learned during training.
    """
    def __init__(self, cfg):
        super().__init__()

        tt_cfg = cfg.migs if isinstance(cfg, dict) else cfg["migs"]

        self.tt_rank = tt_cfg.get("rank", [1,10,10,1]) 
        self.tt_shape = tuple(tt_cfg.get("tt_shape", [1, 50000, 43]))  
        self.verbose = tt_cfg.get("verbose", False)


        self.delay = cfg.model.gaussian.get("delay", 0)
        self.optimizer = None
        self.scheduler = None

        assert len(self.tt_shape) == len(self.tt_rank) - 1, \
            f"tt_shape ({len(self.tt_shape)}) must match len(tt_rank) - 1 ({len(self.tt_rank) - 1})"

        # === Allocate dummy TT cores to enable loading ===
        self.tt_cores = nn.ParameterList()
        for i in range(len(self.tt_shape)-1):
            r1 = self.tt_rank[i]
            n = self.tt_shape[i]
            r2 = self.tt_rank[i + 1]
            core = nn.Parameter(torch.zeros(r1, n, r2))
            self.tt_cores.append(core)

        # === Allocate core2 with correct shapes ===
        r2= self.tt_rank[2]
        r3 = self.tt_rank[3]

        self.core2_xyz      = nn.Parameter(torch.zeros(r2, 3,  r3))  # 0:3
        self.core2_scaling  = nn.Parameter(torch.zeros(r2, 3,  r3))  # 3:6
        self.core2_rotation = nn.Parameter(torch.zeros(r2, 4,  r3))  # 6:10
        self.core2_dc       = nn.Parameter(torch.zeros(r2, 1, r3))
        self.core2_rest     = nn.Parameter(torch.zeros(r2, 31, r3)) 
        self.core2_opacity  = nn.Parameter(torch.zeros(r2, 1,  r3))  # 42:43


    def init_from_tensor(self, gaussian_model):
        G = gaussian_model._xyz.shape[0]
        xyz = gaussian_model._xyz
        scaling = gaussian_model._scaling
        rotation = gaussian_model._rotation
        features_dc = gaussian_model._features_dc.squeeze(-1)
        features_rest = gaussian_model._features_rest.squeeze(-1)
        opacity = gaussian_model._opacity
        print("----------------- INITIALISATION --------------------")
        print(f"[DEBUG] XYZ stats     : min={xyz.min().item():.4f}, max={xyz.max().item():.4f}, mean={xyz.mean().item():.4f}, std={xyz.std().item():.4f}")
        print(f"[DEBUG] Scaling stats : min={scaling.min().item():.4f}, max={scaling.max().item():.4f}, mean={scaling.mean().item():.4f}, std={scaling.std().item():.4f}")
        print(f"[DEBUG] Rotation stats: min={rotation.min().item():.4f}, max={rotation.max().item():.4f}, mean={rotation.mean().item():.4f}, std={rotation.std().item():.4f}")
        print(f"[DEBUG] DC stats      : min={features_dc.min().item():.4f}, max={features_dc.max().item():.4f}, mean={features_dc.mean().item():.4f}, std={features_dc.std().item():.4f}")
        print(f"[DEBUG] rest stats : min={features_rest.min().item():.4f}, max={features_rest.max().item():.4f}, mean={features_rest.mean().item():.4f}, std={features_rest.std().item():.4f}")
        print(f"[DEBUG] Opacity stats : min={opacity.min().item():.4f}, max={opacity.max().item():.4f}, mean={opacity.mean().item():.4f}, std={opacity.std().item():.4f}")



        all_params = [xyz, scaling, rotation, features_dc, features_rest, opacity]
        W_GM = torch.cat([x if x.ndim == 2 else x.view(x.shape[0], -1) for x in all_params], dim=1)
       
        W_identity = W_GM.unsqueeze(0)  # shape: (1, G, M)

        #xpected_shape = tuple(int(x) for x in self.tt_shape)
        #W_tt = W_identity.reshape(expected_shape)
        print(f"[DEBUG] W_identity.shape avant tt = {W_identity.shape}, total = {W_identity.numel()}")
        print(f"[DEBUG] W_identity avant tt = {W_identity}")

        # TT-SVD decomposition
        tt_tensor = tensor_train(W_identity, rank=self.tt_rank, verbose=self.verbose)
        # Recréer l’objet TTTensor avec tous les cores sur GPU
        tt_tensor_on_gpu = type(tt_tensor)([core.to(W_identity.device) for core in tt_tensor.factors])

        # Stocker dans le module
        self.tt_tensor = tt_tensor_on_gpu

        #print(tt_tensor.factors)

        # Inject sliced core2 components
        core2 = self.tt_tensor[2]  # shape: (r2, 43, r3)

        with torch.no_grad():
            self.core2_xyz.copy_(core2[:, 0:3, :])
            self.core2_scaling.copy_(core2[:, 3:6, :])
            self.core2_rotation.copy_(core2[:, 6:10, :])
            self.core2_dc.copy_(core2[:, 10:11, :])
            self.core2_rest.copy_(core2[:, 11:42, :])
            self.core2_opacity.copy_(core2[:, 42:43, :])

        # 1. Choisir un device commun (par exemple celui de W_identity)
        device = W_identity.device

        # # 2. Mettre tous les tenseurs sur le même device
        #W_original = W_identity.squeeze(0).to(device)                    # (G, M)
        # W_tensorly_direct = tt_to_tensor(self.tt_tensor.factors).squeeze(0).to(device)
        #W_reconstructed = self.get_W_for_identity(0).to(device)         # (G, M)
        #TTUltraMIGSModule.compare_reconstruction_per_block(
        #     W_original, W_reconstructed,
        #     split_sizes=[3,3,4,1,31,1],
        #     names=['xyz','scaling','rotation','dc','rest','opacity']
        # )
        #TTUltraMIGSModule.plot_correlation_across_parameters(W_original, W_reconstructed)
        #TTUltraMIGSModule.plot_pca_groupwise_xyz_auto(W_original, W_reconstructed, num_groups=10)
        #TTUltraMIGSModule.plot_tsne_per_block(W_original, W_reconstructed, max_points=1000)
        # TTUltraMIGSModule.plot_per_block_errors(W_original, W_reconstructed)
        # # 3. Comparaisons correctes
        # print("Max diff (get_W_for_identity vs full):", (W_reconstructed - W_tensorly_direct).abs().max().item())
        # print("Max diff (reconstructed vs W_original):", (W_tensorly_direct - W_original).abs().max().item())



        # print(f"[DEBUG] W_reconstructed.shape = {W_reconstructed.shape}")
        # print(f"[DEBUG] W_original.shape = {W_original.shape}")

        # # === Vérification avec reconstruction TensorLy ===
        
        # W_custom = W_reconstructed

        # diff = W_original.to(W_custom.device) - W_custom
        # mse = (diff ** 2).mean().item()
        # l2 = torch.norm(diff).item()
        # print(f"[COMPARE] W_custom vs W_original → MSE: {mse:.6e} | L2 Norm: {l2:.6e}")

        # diff_direct = W_tensorly_direct.to(W_original.device) - W_original
        # mse_direct = (diff_direct ** 2).mean().item()
        # l2_direct = torch.norm(diff_direct).item()
        
        # print("[COMPARE] TensorLy ", W_tensorly_direct)
        # print("[COMPARE] W_original ", W_original)

        # print(f"[COMPARE] TensorLy vs W_original → MSE: {mse_direct:.6e} | L2 Norm: {l2_direct:.6e}")

        # # === Comparaison bloc par bloc (xyz, scaling, etc.) ===
        # split_sizes = [3, 3, 4, 1, 31, 1]  # tailles par bloc : xyz, scaling, rotation, dc, rest, opacity
        # names = ['xyz', 'scaling', 'rotation', 'dc', 'rest', 'opacity']

        # recon_parts = torch.split(W_tensorly_direct, split_sizes, dim=1)
        # orig_parts = torch.split(W_original, split_sizes, dim=1)

        # for name, recon, orig in zip(names, recon_parts, orig_parts):
        #     recon = recon.to(orig.device)
        #     mse = ((recon - orig) ** 2).mean().item()
        #     l2 = torch.norm(recon - orig).item()
        #     print(f"[ERROR] {name.upper()} MSE: {mse:.6e} | L2 Norm: {l2:.6e}")



        r1 = self.tt_rank[1]
        d1 = self.tt_shape[1]
        r2 = self.tt_rank[2]

        # === Core 0: shape (1, 1, r1) ===
        base_value = self.tt_tensor[0]  # (1, 1, 1)
        core0_new = base_value.repeat(1, 1, r1) / r1  # (1, 1, r1)

        # === Core 1: shape (r1, d1, r2) ===
        base_core1 = self.tt_tensor[1]  # (1, d1, r2)
        core1_new = base_core1.repeat(r1, 1, 1)  # (r1, d1, r2)

        # === Assign to model ===
        with torch.no_grad():
            self.tt_cores[0].copy_(core0_new)
            self.tt_cores[1].copy_(core1_new)



        print("📦 TT Cores Shapes & Stats:")
        for i, core in enumerate(self.tt_cores):
            print(f"Core {i} apres duplicate: = {core}")

        # W_reconstructed = self.get_W_for_identity(0).to(device)         # (G, M)
        # print("W_reconstructed ",W_reconstructed)
        # W_custom = W_reconstructed

        # diff = W_original.to(W_custom.device) - W_custom
        # mse = (diff ** 2).mean().item()
        # l2 = torch.norm(diff).item()
        # print(f"[COMPARE] W_custom vs W_original → MSE: {mse:.6e} | L2 Norm: {l2:.6e}")

    @staticmethod
    def compare_reconstruction_per_block(
        W_original: torch.Tensor,
        W_reconstructed: torch.Tensor,
        split_sizes: list = None,
        names: list = None,
    ):
        """
        Affiche erreurs (MSE, RMSE, MAE, L2, cos, Pearson) + stats (min/max/mean)
        pour chaque bloc: xyz, scaling, rotation, dc, rest, opacity.

        Args:
            W_original:   (G, M)
            W_reconstructed: (G, M)
            split_sizes:  ex. [3, 3, 4, 1, 31, 1] (optionnel)
            names:        ex. ['xyz','scaling','rotation','dc','rest','opacity'] (optionnel)
        """
        if names is None:
            names = ['xyz', 'scaling', 'rotation', 'dc', 'rest', 'opacity']
        if split_sizes is None:
            split_sizes = [3, 3, 4, 1, 31, 1]

        M = W_original.shape[1]
        assert M == W_reconstructed.shape[1], \
            f"Column mismatch: W_original={M}, W_reconstructed={W_reconstructed.shape[1]}"

        # Tentative d’auto-ajustement: si la somme ne colle pas, on recalcule 'rest'
        total = sum(split_sizes)
        if total != M and 'rest' in names:
            rest_idx = names.index('rest')
            other_sum = total - split_sizes[rest_idx]
            new_rest = M - other_sum
            if new_rest <= 0:
                raise ValueError(f"Impossible d'ajuster 'rest': M={M}, autres={other_sum}")
            split_sizes = split_sizes.copy()
            split_sizes[rest_idx] = new_rest

        assert sum(split_sizes) == M, \
            f"Split sizes ({sum(split_sizes)}) != tensor width ({M}). " \
            f"split_sizes={split_sizes}, names={names}"

        orig_parts  = torch.split(W_original,     split_sizes, dim=1)
        recon_parts = torch.split(W_reconstructed, split_sizes, dim=1)

        def stats(t: torch.Tensor):
            return t.min().item(), t.max().item(), t.mean().item()

        def mae_rmse(a: torch.Tensor, b: torch.Tensor):
            d = a - b
            mae  = d.abs().mean().item()
            rmse = torch.sqrt((d * d).mean()).item()
            return mae, rmse

        def pearson_corr(a: torch.Tensor, b: torch.Tensor):
            a = a.flatten(); b = b.flatten()
            a_mu, b_mu = a.mean(), b.mean()
            a_c, b_c = a - a_mu, b - b_mu
            a_std = a_c.std(unbiased=False); b_std = b_c.std(unbiased=False)
            denom = a_std * b_std
            if denom.item() == 0.0:
                return float('nan')
            return float((a_c * b_c).mean().item() / denom.item())

        print("\n[TT DIAGNOSTIC] Erreurs de reconstruction + stats par bloc:\n")

        for name, orig, recon in zip(names, orig_parts, recon_parts):
            mse   = torch.mean((orig - recon) ** 2).item()
            rmse  = torch.sqrt(torch.mean((orig - recon) ** 2)).item()
            mae   = torch.mean(torch.abs(orig - recon)).item()
            l2    = torch.norm(orig - recon).item()
            cos   = torch.nn.functional.cosine_similarity(
                        orig.reshape(1, -1), recon.reshape(1, -1)
                    ).item()
            rho   = pearson_corr(orig, recon)

            o_min, o_max, o_mean = stats(orig)
            r_min, r_max, r_mean = stats(recon)

            print(f" → {name.upper():8s} | "
                f"MSE: {mse:.6e} | RMSE: {rmse:.6e} | MAE: {mae:.6e} | L2: {l2:.6e} | "
                f"cos: {cos:.6f} | ρ: {rho:.6f}")
            print(f"    ORIG[min:{o_min:.6f}, max:{o_max:.6f}, mean:{o_mean:.6f}]  |  "
                f"RECON[min:{r_min:.6f}, max:{r_max:.6f}, mean:{r_mean:.6f}]")
        print()



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

    @staticmethod
    def plot_correlation_across_parameters(W_original, W_reconstructed):
        os.makedirs("diagnostic", exist_ok=True)

        # Définir les noms des 43 paramètres
        names = []
        names += ['x', 'y', 'z']                       # 3
        names += ['scale_x', 'scale_y', 'scale_z']     # 3
        names += ['rot_w', 'rot_x', 'rot_y', 'rot_z']  # 4
        names += ['dc']                                # 1
        names += [f'feat_{i}' for i in range(31)]      # 31
        names += ['opacity']                           # 1

        def compute_corr_full(tensor):
            arr = tensor.detach().cpu().numpy()
            df = pd.DataFrame(arr, columns=names)

            # Calculer la variance pour debug
            var = df.var()
            plt.figure(figsize=(12, 4))
            var.plot(kind='bar')
            plt.title("Variance de chaque paramètre")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig("diagnostic/param_variances.png")
            plt.close()

            # Corrélation complète, NaNs remplacés par 0 pour éviter les erreurs
            corr = df.corr().reindex(index=names, columns=names).fillna(0)

            return corr.values, names

        corr_orig, labels_orig = compute_corr_full(W_original)
        corr_recon, labels_recon = compute_corr_full(W_reconstructed)

        # Même labels (tous les paramètres)
        labels = names

        # Créer DataFrames pour bien aligner les matrices
        df_orig = pd.DataFrame(corr_orig, index=labels, columns=labels)
        df_recon = pd.DataFrame(corr_recon, index=labels, columns=labels)

        # Calcul de la différence
        corr_diff = df_orig.values - df_recon.values
        frob = np.linalg.norm(corr_diff, ord='fro')

        for mat, title, fname in [
            (df_orig.values, "Original", "param_corr_orig.png"),
            (df_recon.values, "Reconstructed", "param_corr_recon.png"),
            (corr_diff, f"Difference (Frobenius = {frob:.4f})", "param_corr_diff.png")
        ]:
            plt.figure(figsize=(14, 12))
            sns.heatmap(mat, xticklabels=labels, yticklabels=labels,
                        cmap='coolwarm' if "Difference" not in title else 'bwr', center=0,
                        annot=False)
            plt.title(f"Correlation Matrix – {title}")
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f"diagnostic/{fname}")
            plt.close()

        print(f"📊 Frobenius norm of correlation matrix difference: {frob:.6f}")



    def recombine_core2(self):
        return torch.cat([
            self.core2_xyz,
            self.core2_scaling,
            self.core2_rotation,
            self.core2_dc,
            self.core2_rest,
            self.core2_opacity
        ], dim=1)  # shape: (r2, 43, r3)


    def get_core0(self, idx):
        assert 0 <= idx < self.tt_cores[0].shape[1], f"Invalid identity index {idx}"
        return self.tt_cores[0][:, idx:idx+1, :]  # shape: (1, 1, r1)


    def expand_first_core(self, n_identities):
        """
        Duplique exactement la première identité du core TT0 pour toutes les identités.
        """
        if self.tt_cores is None:
            raise RuntimeError("TT cores must be initialized before expansion.")

        first_core = self.tt_cores[0]  # shape: (1, current_n_id, r1)
        r0, current_n_id, r1 = first_core.shape
        assert r0 == 1, f"Expected first TT rank r0 = 1, got {r0}"

        if current_n_id >= n_identities:
            print(f"ℹFirst core already has {current_n_id} identities (≥ {n_identities}), no expansion needed.")
            return

        print(f"Expanding first core from {current_n_id} to {n_identities} identities by duplication...")

        base = first_core[:, 0:1, :].detach()  # shape (1, 1, r1)
        new_core = base.repeat(1, n_identities, 1)  # shape (1, n_identities, r1)

        self.tt_cores[0] = nn.Parameter(new_core)
        print(f"✔️ First core duplicated → new shape: {self.tt_cores[0].shape}")


    def reconstruct(self):
        return tt_to_tensor([
            self.tt_cores[0],     # Core 0
            self.tt_cores[1],     # Core 1
            self.recombine_core2() # Core 2
        ])

        

    def get_tt_tensor(self, idx=None):
        """
        Builds the list of TT cores to feed into tt_to_tensor().
        If idx is provided, slices the first core to reconstruct only that identity.
        """
        if idx is not None:
            core0 = self.get_core0(idx)  # shape: (1, 1, r1)
        else:
            core0 = self.tt_cores[0]     # shape: (1, N, r1)

        device = core0.device  # Prend le device du core0 pour cohérence

        return [
            core0.to(device),
            self.tt_cores[1].to(device),
            self.recombine_core2().to(device)
        ]



    def get_W_for_identity(self, idx) -> torch.Tensor:
        """
        Reconstruct the Gaussian parameter matrix W[idx] for a specific identity
        using the Tensor Train (TT) decomposition and only the relevant TT core slice.

        Args:
            idx (int): Identity index to reconstruct.

        Returns:
            torch.Tensor: Reconstructed Gaussian parameters of shape (num_gaussians, num_params)
        """
        return tt_to_tensor(self.get_tt_tensor(idx)).squeeze(0)   # shape: (G, M)


    def optimize_parameters(self):
        return list(self.tt_cores[:2]) + [  # core 0 et core 1
            self.core2_xyz,
            self.core2_scaling,
            self.core2_rotation,
            self.core2_dc,
            self.core2_rest,
            self.core2_opacity,
        ]



    def freeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = False

    def unfreeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = True

    def set_optimizer(self, opt_cfg):
        tt_lrs = opt_cfg.get("tt_lrs", [1.6e-4] * 2)
        tt_final_lrs = opt_cfg.get("tt_final_lrs", [1.6e-6] * 2)
        tt_decay_iters = opt_cfg.get("tt_decay_iters", 50000)

        param_groups = []

        # TT cores 0 to 3 — with decay
        for i in range(len(self.tt_shape) - 1):
            param_groups.append({
                "params": [self.tt_cores[i]],
                "lr": tt_lrs[i],
                "initial_lr": tt_lrs[i],
                "final_lr": tt_final_lrs[i]
            })

        # core2_xyz — with decay
        param_groups.append({
            "params": [self.core2_xyz],
            "lr": 1.6e-4,
            "initial_lr": 1.6e-4,
            "final_lr": 1.6e-6
        })

        # Other core2 slices — constant LR (no decay)
        param_groups += [
            {"params": [self.core2_scaling], "lr": 5e-3},
            {"params": [self.core2_rotation], "lr": 1e-3},
            {"params": [self.core2_dc],      "lr": 2.5e-3},
            {"params": [self.core2_rest],    "lr": 2.5e-3},
            {"params": [self.core2_opacity], "lr": 5e-2},
        ]

        self.optimizer = torch.optim.Adam(param_groups)

        # Scheduler only for groups with decay
        decay_groups = [g for g in param_groups if "final_lr" in g]
        if decay_groups:
            # Use same gamma for all decayed groups
            gamma = (1.6e-6 / 1.6e-4) ** (1. / tt_decay_iters)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        else:
            self.scheduler = None



    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, iteration=None):
        if self.optimizer is None:
            return

        # Bloquer complètement tout avant delay
        if iteration is not None and iteration < self.delay:
            if iteration == self.delay - 1:
                print(f"[TTUltra] TT cores frozen until iteration {iteration}")
            self.freeze_tt_parameters()
            
            # Supprimer tous les gradients accidentellement calculés
            self.optimizer.zero_grad()
            return

        if iteration == self.delay:
            self.unfreeze_tt_parameters()
            print(f"[TTUltra] TT cores unfrozen at iteration {iteration}")

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_learning_rate()


    def add_identity(self):
        if self.tt_cores is None:
            raise RuntimeError("TT cores must be initialized before adding an identity.")

        first_core = self.tt_cores[0]  # shape: (1, n_identities, r1)
        r0, n_id, r1 = first_core.shape
        new_identity = torch.randn(1, 1, r1, device=first_core.device) * 0.01
        expanded_core = torch.cat([first_core, new_identity], dim=1)
        self.tt_cores[0] = nn.Parameter(expanded_core)
        print(f"🆕 Added identity. New first core shape: {self.tt_cores[0].shape}")
        return self.tt_cores[0].shape[1] - 1
