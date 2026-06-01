import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor

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

        self.tt_rank = tt_cfg.get("rank", [1, 10, 10, 10, 10, 1])  # 5 cores
        self.tt_shape = tuple(tt_cfg.get("tt_shape", [1, 100, 50,10,43]))
        self.verbose = tt_cfg.get("verbose", False)

        self.delay = cfg.model.gaussian.get("delay", 0)
        self.optimizer = None
        self.scheduler = None

        assert len(self.tt_shape) == len(self.tt_rank) - 1, \
            f"tt_shape ({len(self.tt_shape)}) must match len(tt_rank) - 1 ({len(self.tt_rank) - 1})"

        # === Allocate dummy TT cores to enable loading ===
        self.tt_cores = nn.ParameterList()
        for i in range(len(self.tt_shape)):
            r1 = self.tt_rank[i]
            n = self.tt_shape[i]
            r2 = self.tt_rank[i + 1]
            core = nn.Parameter(torch.randn(r1, n, r2) * 1e-2)
            self.tt_cores.append(core)

        # === Allocate core 4 slices with correct shapes ===
        r4 = self.tt_rank[4]
        r5 = self.tt_rank[5]

        self.core4_xyz      = nn.Parameter(torch.empty(r4, 3,  r5))  # 0:3
        self.core4_scaling  = nn.Parameter(torch.empty(r4, 3,  r5))  # 3:6
        self.core4_rotation = nn.Parameter(torch.empty(r4, 4,  r5))  # 6:10
        self.core4_features = nn.Parameter(torch.empty(r4, 32, r5))  # 10:42
        self.core4_opacity  = nn.Parameter(torch.empty(r4, 1,  r5))  # 42:43


    def init_from_tensor(self, gaussian_model):
        G = gaussian_model._xyz.shape[0]
        xyz = gaussian_model._xyz
        scaling = gaussian_model._scaling
        rotation = gaussian_model._rotation
        features_dc = gaussian_model._features_dc.squeeze(-1)
        features_rest = gaussian_model._features_rest.squeeze(-1)
        opacity = gaussian_model._opacity

        all_params = [xyz, scaling, rotation, features_dc, features_rest, opacity]
        W_GM = torch.cat([x if x.ndim == 2 else x.view(x.shape[0], -1) for x in all_params], dim=1)
        W_identity = W_GM.unsqueeze(0)  # shape: (1, G, M)

        expected_shape = tuple(int(x) for x in self.tt_shape)
        W_tt = W_identity.reshape(expected_shape)
        print(f"[DEBUG] W_identity.shape = {W_identity.shape}, total = {W_identity.numel()}")
        print(f"[DEBUG] W_tt.shape = {W_tt.shape}, total = {W_tt.numel()}")

        # TT-SVD decomposition
        tt_tensor = tensor_train(W_tt, rank=self.tt_rank, verbose=self.verbose)

        # Inject sliced core4 components
        core4 = tt_tensor.factors[4]  # shape: (r4, 43, r5)

        with torch.no_grad():
            self.core4_xyz.copy_(core4[:, 0:3, :])
            self.core4_scaling.copy_(core4[:, 3:6, :])
            self.core4_rotation.copy_(core4[:, 6:10, :])
            self.core4_features.copy_(core4[:, 10:42, :])
            self.core4_opacity.copy_(core4[:, 42:43, :])

        # Update cores 0, 1, 2, 3 — we keep 4 separate
        r1 = self.tt_rank[1]
        d1 = self.tt_shape[1]
        r2 = self.tt_rank[2]

        # === Core 0: expand with noise ===
        base_value = tt_tensor.factors[0][0, 0, 0].detach().view(1, 1, 1)
        core0_list = [base_value]
        for _ in range(r1 - 1):
            noisy = base_value + 0.01 * torch.randn_like(base_value)
            core0_list.append(noisy)
        core0_new = torch.cat(core0_list, dim=2)  # shape: (1, 1, r1)

        # === Core 1: duplicate base slice ===
        base_core1 = tt_tensor.factors[1]
        base_slice = base_core1[0]  # (d1, r2)
        core1_list = [base_slice.unsqueeze(0)]
        for _ in range(r1 - 1):
            noisy_slice = base_slice + 0.01 * torch.randn_like(base_slice)
            core1_list.append(noisy_slice.unsqueeze(0))
        core1_new = torch.cat(core1_list, dim=0)  # (r1, d1, r2)

        # Replace tt_cores (core 0–3)
        self.tt_cores = nn.ParameterList([
            nn.Parameter(core0_new),
            nn.Parameter(core1_new),
            nn.Parameter(tt_tensor.factors[2]),
            nn.Parameter(tt_tensor.factors[3])
        ])

        print("📦 TT Cores Shapes & Stats:")
        for i, core in enumerate(self.tt_cores):
            print(f"Core {i}: shape = {core.shape}")

    def recombine_core4(self):
        return torch.cat([
            self.core4_xyz,
            self.core4_scaling,
            self.core4_rotation,
            self.core4_features,
            self.core4_opacity
        ], dim=1)  # shape: (r4, 43, r5)



    def expand_first_core(self, n_identities):
        if self.tt_cores is None:
            raise RuntimeError("TT cores must be initialized before expansion.")

        first_core = self.tt_cores[0]  # shape: (1, current_n_id, r1)
        r0, current_n_id, r1 = first_core.shape
        assert r0 == 1, f"Expected first TT rank r0 = 1, got {r0}"

        if current_n_id >= n_identities:
            print(f"ℹ️ First core already has {current_n_id} identities (≥ {n_identities}), no expansion needed.")
            return

        n_new = n_identities - current_n_id
        print(f"➕ Expanding first core from {current_n_id} to {n_identities} identities...")

        new_identities = torch.randn(1, n_new, r1, device=first_core.device) * 0.01
        expanded_core = torch.cat([first_core, new_identities], dim=1)

        self.tt_cores[0] = nn.Parameter(expanded_core)


        print(f"✔️ First core expanded to shape: {self.tt_cores[0].shape}")

    def reconstruct(self):
        full_tt_cores = list(self.tt_cores[:4]) + [self.recombine_core4()]
        return tt_to_tensor(full_tt_cores)


    def get_W_for_identity(self, idx: int) -> torch.Tensor:
        """
        Reconstruct the Gaussian parameter matrix W[idx] for a specific identity
        from the Tensor Train (TT) decomposition, using the decomposed TT cores.

        Args:
            idx (int): Identity index to reconstruct.

        Returns:
            torch.Tensor: Reconstructed Gaussian parameters for identity `idx`,
                        of shape (num_gaussians, num_params)
        """
        # Recombine the last core (core 4) from its decomposed parts (xyz, scaling, etc.)
        core4 = self.recombine_core4()  # shape: (r4, 43, r5)

        # Rebuild the list of TT cores (including the recombined core4)
        full_tt_cores = list(self.tt_cores[:4]) + [core4]

        # === Begin TT reconstruction ===
        # Select the slice for identity `idx` from the first core
        # Core 0 shape: (1, num_identities, r1) → select (1, r1)
        x = full_tt_cores[0][:, idx, :]  # shape: (1, r1)

        # Contract sequentially with the rest of the TT cores
        for i, core in enumerate(full_tt_cores[1:]):
            if x.dim() == 2:
                # Add dummy dimension for compatibility if needed
                x = x.unsqueeze(1)  # shape: (1, 1, r_k)

            # Perform tensor contraction: (1, d, r_k) × (r_k, n_k, r_{k+1}) → (1, d, n_k, r_{k+1})
            x = torch.einsum('bdr,rnq->bdnq', x, core)

            if i < len(full_tt_cores[1:]) - 1:
                # If not the last core, flatten the (d, n_k) dimensions to continue contraction
                b, d, n, r = x.shape
                x = x.reshape(b, d * n, r)
            else:
                # Last core, stop here
                pass

        # Final output shape: (1, num_gaussians, num_params, 1)
        # Remove batch and trailing singleton dimensions
        x = x.squeeze(0).squeeze(-1)  # shape: (num_gaussians, num_params)

        return x


    def forward(self, *tt_indices):
        full_tt_cores = list(self.tt_cores[:4]) + [self.recombine_core4()]
        assert len(tt_indices) == len(full_tt_cores), "Index count mismatch"
        output = full_tt_cores[0][:, tt_indices[0], :]
        for i in range(1, len(full_tt_cores)):
            output = torch.einsum('br,rjkc->bjkc', output, full_tt_cores[i][:, tt_indices[i], :])
        return output.squeeze()


    def optimize_parameters(self):
        return list(self.tt_cores[:4]) + [
            self.core4_xyz,
            self.core4_scaling,
            self.core4_rotation,
            self.core4_features,
            self.core4_opacity,
        ]


    def freeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = False

    def unfreeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = True

    def set_optimizer(self, opt_cfg):
        tt_lrs = opt_cfg.get("tt_lrs", [1.6e-4] * 4)
        tt_final_lrs = opt_cfg.get("tt_final_lrs", [1.6e-6] * 4)
        tt_decay_iters = opt_cfg.get("tt_decay_iters", 50000)

        assert len(self.tt_cores) >= 4, "Expected at least 4 TT cores (core4 is split separately)"

        param_groups = []

        # Cores 0 to 3 — with decay
        for i in range(4):
            param_groups.append({
                "params": [self.tt_cores[i]],
                "lr": tt_lrs[i],
                "initial_lr": tt_lrs[i],
                "final_lr": tt_final_lrs[i]
            })

        # Core 4 slices — with or without decay
        core4_slices = [
            ("core4_xyz", self.core4_xyz, 1.6e-4, 1.6e-6),
            ("core4_scaling", self.core4_scaling, 5e-3, None),
            ("core4_rotation", self.core4_rotation, 1e-3, None),
            ("core4_features", self.core4_features, 2.5e-3, None),
            ("core4_opacity", self.core4_opacity, 5e-2, None),
        ]

        for name, param, lr, final_lr in core4_slices:
            group = {"params": [param], "lr": lr}
            if final_lr is not None:
                group["initial_lr"] = lr
                group["final_lr"] = final_lr
            param_groups.append(group)

        self.optimizer = torch.optim.Adam(param_groups)

        # Scheduler only for groups with decay
        gammas = []
        for pg in param_groups:
            if "final_lr" in pg:
                init_lr = pg["initial_lr"]
                final_lr = pg["final_lr"]
                gamma = (final_lr / init_lr) ** (1.0 / tt_decay_iters)
                gammas.append(gamma)
            else:
                gammas.append(None)

        if any(g is not None for g in gammas):
            decay_gamma = [g for g in gammas if g is not None][0]  # all decayed groups use same gamma
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=decay_gamma
            )
        else:
            self.scheduler = None


    def update_learning_rate(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, iteration=None):
        if self.optimizer is None:
            return

        if iteration is not None:
            if iteration < self.delay:
                self.freeze_tt_parameters()
                if iteration == self.delay - 1:
                    print(f"[TTUltra] TT cores frozen until iteration {iteration}")
                return
            elif iteration == self.delay:
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
