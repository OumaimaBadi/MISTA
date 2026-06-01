import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import CPPower
import os
import numpy as np
import pandas as pd
from utils.migs_utils import (
    compare_reconstruction_per_block,
    plot_correlation_across_parameters,
    plot_pca_groupwise_xyz_auto,
)


# Use the PyTorch backend for TensorLy ops
tl.set_backend('pytorch')


class CPMIGSModule(nn.Module):
    """
    CP-based MIGS module:
    factorizes a stacked Gaussian-parameter tensor into U1 (parameter blocks),
    U2 (identities), and U3 (Gaussians), then learns these factors.
    """
    def __init__(self, cfg):
        super().__init__()
        migs_cfg = cfg.migs if not isinstance(cfg, dict) else cfg["migs"]

        self.rank = migs_cfg.get("rank", 10)
        self.n_repeat = migs_cfg.get("n_repeat", 5)
        self.n_iteration = migs_cfg.get("n_iteration", 5)
        self.verbose = migs_cfg.get("verbose", 0)

        if "dataset" in cfg and "names" in cfg["dataset"]:
            self.n_identities = len(cfg["dataset"]["names"])
        else:
            raise ValueError("Could not determine number of identities.")

        # U1 is split by parameter type to allow different learning rates
        self.U1_xyz      = nn.Parameter(torch.zeros(3, self.rank))
        self.U1_scaling  = nn.Parameter(torch.zeros(3, self.rank))
        self.U1_rotation = nn.Parameter(torch.zeros(4, self.rank))
        self.U1_dc       = nn.Parameter(torch.zeros(1, self.rank))   # updated in init_from_tensor
        self.U1_rest     = nn.Parameter(torch.zeros(31, self.rank))  # default for SH features; updated later
        self.U1_opacity  = nn.Parameter(torch.zeros(1, self.rank))

        # U2: identities; U3: gaussians (rows)
        self.U2 = nn.Parameter(torch.zeros(self.n_identities, self.rank))
        self.U3 = nn.Parameter(torch.zeros(50000, self.rank))

        self.weights = None
        self.scheduler = None
        self.optimizer = None
        self._opt_cfg = None
        self._needs_opt_rebuild = False

        # Optional training delay for U1/U2/U3
        self.delay = cfg.model.gaussian.get("delay", 0)

    def init_from_tensor(self, gaussian_model):
        """
        CP init robust to large ranks, with sanity prints:
        - randomize copies of near-zero feature blocks only for CP init,
        - sanitize NaNs for large ranks (>=100),
        - fold CP weights into U3,
        - print stats (shape, min/max/mean, any_nan/any_inf) for weights/U1/U2/U3.
        """
        @torch.no_grad()
        def _stats(name, t):
            any_nan = torch.isnan(t).any().item()
            any_inf = torch.isinf(t).any().item()
            t_min = t.min().item() if t.numel() > 0 else float('nan')
            t_max = t.max().item() if t.numel() > 0 else float('nan')
            t_mean = t.mean().item() if t.numel() > 0 else float('nan')
            nonfinite = (~torch.isfinite(t)).sum().item()
            print(f"[CP-INIT][{name}] shape={tuple(t.shape)} "
                f"min={t_min:.3e} max={t_max:.3e} mean={t_mean:.3e} "
                f"any_nan={bool(any_nan)} any_inf={bool(any_inf)} nonfinite={nonfinite}")

        def _near_zero(t, eps=1e-12):
            # If all zeros or extremely small range → treat as near-zero
            return (not torch.isfinite(t).all()) or (t.abs().max() <= eps)

        xyz = gaussian_model._xyz
        scaling = gaussian_model._scaling
        rotation = gaussian_model._rotation
        features_dc = gaussian_model._features_dc.squeeze(-1)      # (G, 1)
        features_rest = gaussian_model._features_rest.squeeze(-1)  # (G, 31)
        opacity = gaussian_model._opacity

        # make CP-init-safe copies of feature blocks if they are near-zero 
        f_dc_stable = features_dc if not _near_zero(features_dc) else torch.rand_like(features_dc)
        f_rest_stable = features_rest if not _near_zero(features_rest) else torch.rand_like(features_rest)

        # flatten per-point blocks into (G, M) and add identity mode (1,G,M)
        params_for_cp = [xyz, scaling, rotation, f_dc_stable, f_rest_stable, opacity]
        W_GM = torch.cat(
            [x if x.ndim == 2 else x.view(x.shape[0], -1) for x in params_for_cp],
            dim=1
        ).contiguous()  # contiguous helps TensorLy sometimes
        W_identity = W_GM.unsqueeze(0)  # (1, G, M)

        # CP via TensorLy's CPPower
        cp_model = CPPower(rank=self.rank, n_repeat=self.n_repeat, n_iteration=self.n_iteration)
        weights, (U2, U3, U1) = cp_model.fit_transform(W_identity)  # modes: (I, G, M)

        # Print raw stats straight from CPPower
        _stats("weights (raw)", weights)
        _stats("U1 (raw)", U1)
        _stats("U2 (raw)", U2)
        _stats("U3 (raw)", U3)

        # large-rank NaN fallback
        if self.rank >= 100:
            weights = torch.nan_to_num(weights, nan=1e-20)
            U1 = torch.nan_to_num(U1, nan=1e-7)
            U2 = torch.nan_to_num(U2, nan=1.0)
            U3 = torch.nan_to_num(U3, nan=1e-7)

            # Print after sanitization
            _stats("weights (sanitized)", weights)
            _stats("U1 (sanitized)", U1)
            _stats("U2 (sanitized)", U2)
            _stats("U3 (sanitized)", U3)

        # fold CP weights into U3 to keep scales reasonable
        U3 = U3 * weights[None, :]

        # Print after folding weights into U3
        _stats("U3 (after fold weights)", U3)

        # dims: xyz(3), scaling(3), rotation(4), dc(1), rest(...), opacity(1)
        dc_dim = features_dc.shape[1] if features_dc.ndim > 1 else 1  # -> 1 in our case
        dc_end = 10 + dc_dim  # 3 + 3 + 4 = 10, then + dc_dim

        self.U1_xyz      = nn.Parameter(U1[0:3].detach())
        self.U1_scaling  = nn.Parameter(U1[3:6].detach())
        self.U1_rotation = nn.Parameter(U1[6:10].detach())
        self.U1_dc       = nn.Parameter(U1[10:dc_end].detach())
        self.U1_rest     = nn.Parameter(U1[dc_end:-1].detach())
        self.U1_opacity  = nn.Parameter(U1[-1:].detach())

        # store weights and factors
        self.weights = weights.detach()
        self.U2 = nn.Parameter(U2.detach())
        self.U3 = nn.Parameter(U3.detach())

        # Final assertions
        assert torch.isfinite(self.U2).all(), "U2 has non-finite values after CP init."
        assert torch.isfinite(self.U3).all(), "U3 has non-finite values after CP init."
        assert torch.isfinite(self.get_U1()).all(), "U1 has non-finite values after CP init."

        # Optional quick diagnostics
        try:
            device = W_identity.device
            W_original = W_identity.squeeze(0).to(device)
            W_reconstructed = self.get_W_for_identity(0).to(device)
            compare_reconstruction_per_block(W_original, W_reconstructed)
            plot_correlation_across_parameters(W_original, W_reconstructed)
            plot_pca_groupwise_xyz_auto(W_original, W_reconstructed, num_groups=10)
        except Exception:
            pass

        # freeze until delay
        self.freeze_cp_parameters()


    def freeze_cp_parameters(self):
        """Set requires_grad=False for U1/U2/U3 blocks."""
        for p in self.optimize_parameters():
            p.requires_grad = False

    def unfreeze_cp_parameters(self):
        """Set requires_grad=True for U1/U2/U3 blocks."""
        for p in self.optimize_parameters():
            p.requires_grad = True

    def rebuild_optimizer_like_before(self):
        """Rebuild optimizer/scheduler using the saved config."""
        if self._opt_cfg is None:
            return
        self.set_optimizer(self._opt_cfg)
        self._needs_opt_rebuild = False

    def get_U1(self):
        """Concatenate U1 sub-blocks into a (M, R) matrix."""
        return torch.cat(
            [self.U1_xyz, self.U1_scaling, self.U1_rotation, self.U1_dc, self.U1_rest, self.U1_opacity],
            dim=0
        )

    def set_optimizer(self, opt_cfg):
        """Create optimizer and optional LR scheduler."""
        self._opt_cfg = dict(opt_cfg) if opt_cfg is not None else {}

        position_lr_init = opt_cfg.get("position_lr_init", 1.6e-4)
        position_lr_final = opt_cfg.get("position_lr_final", 1.6e-6)
        iterations = opt_cfg.get("iterations", 50000)

        self.optim_groups = [
            # Decayed groups
            {"params": [self.U1_xyz], "lr": position_lr_init, "initial_lr": position_lr_init, "final_lr": position_lr_final},
            {"params": [self.U2],     "lr": position_lr_init, "initial_lr": position_lr_init, "final_lr": position_lr_final},
            {"params": [self.U3],     "lr": position_lr_init, "initial_lr": position_lr_init, "final_lr": position_lr_final},
            # Fixed-LR groups
            {"params": [self.U1_scaling],  "lr": opt_cfg.get("scaling_lr", 5e-3)},
            {"params": [self.U1_rotation], "lr": opt_cfg.get("rotation_lr", 1e-3)},
            {"params": [self.U1_dc],       "lr": opt_cfg.get("feature_lr", 2.5e-3)},
            {"params": [self.U1_rest],     "lr": opt_cfg.get("feature_lr", 2.5e-3)},
            {"params": [self.U1_opacity],  "lr": opt_cfg.get("opacity_lr", 5e-2)},
        ]

        self.optimizer = torch.optim.Adam(self.optim_groups)

        # Exponential decay for groups with final_lr
        if any("final_lr" in g for g in self.optim_groups):
            gamma = (position_lr_final / position_lr_init) ** (1.0 / iterations)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        else:
            self.scheduler = None

    def update_learning_rate(self):
        """Step the LR scheduler if present."""
        if self.scheduler is not None:
            self.scheduler.step()

    def expand_U2(self, n_identities: int):
        """Expand U2 to match the desired number of identities by duplicating row 0."""
        assert self.U2 is not None, "Call init_from_tensor first."
        base = self.U2.detach()[0].unsqueeze(0)
        self.U2 = nn.Parameter(base.repeat(n_identities, 1))
        print("expand_U2 → U2 shape:", self.U2.shape)

    def reconstruct_W(self):
        """Reconstruct the full tensor W with shape (I, G, M)."""
        U1 = self.get_U1()                              # (M, R)
        U2_KR_U1 = tl.tenalg.khatri_rao([self.U2, U1])  # (I* M, R) Khatri–Rao order matches CPPower
        W_flat = self.U3 @ U2_KR_U1.T                   # (G, I*M)
        I, G, M = self.U2.shape[0], self.U3.shape[0], U1.shape[0]
        return W_flat.view(G, I, M).permute(1, 0, 2)    # (I, G, M)

    def get_W_for_identity(self, idx: int):
        """Reconstruct W for a single identity i (G, M)."""
        U1 = self.get_U1()          # (M, R)
        u2 = self.U2[idx]           # (R,)
        # Scale U1 columns by u2, then multiply by U3
        return self.U3 @ (U1.mul(u2).T).contiguous()  # (G, M)

    @torch.no_grad()
    def add_identity(self, noise_scale: float = 0.05, rebuild_optimizer: bool = True) -> int:
        """
        Append a new U2 row using a neutral initialization:
        mean(U2) + noise_scale * std(U2) * N(0,1), then rescale to median row norm.
        """
        assert self.U2 is not None, "Call init_from_tensor() first."
        device = self.U2.device
        R = self.U2.shape[1]

        if self.U2.shape[0] > 0:
            UI = self.U2.detach()
            mu = UI.mean(dim=0, keepdim=True)
            sig = UI.std(dim=0, unbiased=False, keepdim=True).clamp_(min=1e-8)
            new_row = mu + noise_scale * sig * torch.randn(1, R, device=device)

            target_norm = UI.norm(dim=1).median()
            cur_norm = new_row.norm()
            new_row = new_row / (cur_norm + 1e-8) * float(target_norm)
        else:
            new_row = torch.randn(1, R, device=device) * 0.02

        self.U2 = nn.Parameter(torch.cat([self.U2, new_row], dim=0))
        self.n_identities = self.U2.shape[0]
        self._needs_opt_rebuild = True

        if rebuild_optimizer and (self.optimizer is not None):
            self.rebuild_optimizer_like_before()

        return self.U2.shape[0] - 1

    def update_identity_row(self, idx: int, new_row: torch.Tensor):
        """Replace U2[idx] with new_row (shape must be (1, R))."""
        assert new_row.shape == (1, self.rank)
        self.U2.data[idx:idx + 1] = new_row

    def get_cp_factors(self):
        """Return (U1, U2, U3, weights)."""
        return self.get_U1(), self.U2, self.U3, self.weights

    def optimize_parameters(self):
        """Return Parameters to freeze/unfreeze as a group."""
        return [
            self.U1_xyz, self.U1_scaling, self.U1_rotation,
            self.U1_dc, self.U1_rest, self.U1_opacity,
            self.U2, self.U3,
        ]

    def loss_mars(self):
        """No MARS/L0 regularizer in CP; return zero for logging compatibility."""
        device = self.U2.device if self.U2.is_cuda else 'cpu'
        return torch.zeros([], device=device)

    def enable_identity_finetune(self, idx: int, color_mlp: nn.Module,
                                 lr_id: float = 3e-3, lr_tex: float = 1e-3,
                                 include_color_in_ft_opt: bool = False):
        """
        Finetune only U2[idx] (and optionally the color MLP).
        Other parameters are frozen.
        """
        # Freeze all
        for p in self.parameters():
            p.requires_grad = False

        # Train only the selected U2 row (mask gradients)
        self.U2.requires_grad = True
        mask = torch.zeros_like(self.U2)
        mask[idx:idx + 1, :] = 1.0
        if hasattr(self, "_u2_mask_hook") and self._u2_mask_hook is not None:
            self._u2_mask_hook.remove()
        self._u2_mask_hook = self.U2.register_hook(lambda g: g * mask)

        # Optionally include color MLP in the FT optimizer
        for p in color_mlp.parameters():
            p.requires_grad = include_color_in_ft_opt

        params = [{"params": [self.U2], "lr": lr_id}]
        if include_color_in_ft_opt:
            params.append({"params": list(color_mlp.parameters()), "lr": lr_tex})
        self._ft_opt = torch.optim.Adam(params)

    def ft_step(self):
        """Step the finetune optimizer if present."""
        if hasattr(self, "_ft_opt") and (self._ft_opt is not None):
            self._ft_opt.step()
            self._ft_opt.zero_grad()

    def disable_identity_finetune(self):
        """Exit finetune mode; unmask and unfreeze everything."""
        if hasattr(self, "_u2_mask_hook") and self._u2_mask_hook is not None:
            self._u2_mask_hook.remove()
        self._u2_mask_hook = None
        self._ft_opt = None
        for p in self.parameters():
            p.requires_grad = True

    def step(self, iteration=None):
        """
        Optimizer step:
        - If in finetune mode, use the FT optimizer.
        - Otherwise use the main optimizer with optional warmup delay.
        """
        if hasattr(self, "_ft_opt") and (self._ft_opt is not None):
            self.ft_step()
            return

        if self.optimizer is None:
            return

        if iteration is not None:
            if iteration < self.delay:
                self.freeze_cp_parameters()
                if iteration == self.delay - 1:
                    print(f"[MIGS] U1/U2/U3 frozen until iteration {iteration}")
                return
            elif iteration == self.delay:
                self.unfreeze_cp_parameters()
                print(f"[MIGS] U1/U2/U3 unfrozen at iteration {iteration}")

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.update_learning_rate()