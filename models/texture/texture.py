import torch
import torch.nn as nn

from utils.sh_utils import eval_sh, eval_sh_bases, augm_rots
from utils.general_utils import build_rotation, make_subseed, torch_rng_context
from models.network_utils import VanillaCondMLP


class ColorPrecompute(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata

    def forward(self, gaussians, camera, iteration=None):
        raise NotImplementedError


class SH2RGB(ColorPrecompute):
    def __init__(self, cfg, metadata, root_seed=None):
        super().__init__(cfg, metadata)

    def forward(self, gaussians, camera, iteration=None):
        shs_view = gaussians.get_features.transpose(1, 2).view(
            -1, 3, (gaussians.max_sh_degree + 1) ** 2
        )
        dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1))
        if self.cfg.cano_view_dir:
            T_fwd = gaussians.fwd_transform
            R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
            dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
            view_noise_scale = self.cfg.get('view_noise', 0.)
            if self.training and view_noise_scale > 0.:
                # Optional: deterministic view-noise (per iter/frame)
                seed = make_subseed(
                    int(getattr(self.cfg, "seed", self.metadata.get("seed", 123))),
                    f"texture/sh_view_noise/{int(iteration) if iteration is not None else -1}/{int(getattr(camera,'frame_id',-1))}"
                )
                with torch_rng_context(seed):
                    vn = torch.tensor(
                        augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                        dtype=torch.float32, device=dir_pp.device
                    ).transpose(0, 1)
                dir_pp = torch.matmul(dir_pp, vn)

        dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
        sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return colors_precomp


class ColorMLP(ColorPrecompute):
    def __init__(self, cfg, metadata, root_seed=None):
        super().__init__(cfg, metadata)
        self.base_seed = int(root_seed) if root_seed is not None else int(
            getattr(cfg, "seed", (metadata.get("seed", 123) if isinstance(metadata, dict) else 123))
        )

        d_in = cfg.feature_dim
        self.use_xyz     = cfg.get('use_xyz', False)
        self.use_cov     = cfg.get('use_cov', False)
        self.use_normal  = cfg.get('use_normal', False)
        self.sh_degree   = cfg.get('sh_degree', 0)
        self.cano_view_dir = cfg.get('cano_view_dir', False)
        self.non_rigid_dim = cfg.get('non_rigid_dim', 0)

        if self.use_xyz:    d_in += 3
        if self.use_cov:    d_in += 6  # upper triangle
        if self.use_normal: d_in += 3
        if self.sh_degree > 0:
            d_in += (self.sh_degree + 1) ** 2 - 1
            self.sh_embed = lambda dir: eval_sh_bases(self.sh_degree, dir)[..., 1:]
        if self.non_rigid_dim > 0:
            d_in += self.non_rigid_dim

        d_out = 3

        # Deterministic weights (same across 4D/5D/6D)
        with torch_rng_context(make_subseed(self.base_seed, "texture/mlp")):
            self.mlp = VanillaCondMLP(d_in, 0, d_out, cfg.mlp)

        self.color_activation = nn.Sigmoid()

    def compose_input(self, gaussians, camera, iteration=None):
        features = gaussians.get_features.squeeze(-1)  # (N, F)
        n_points = features.shape[0]

        if self.use_xyz:
            aabb = self.metadata["aabb"]
            xyz_norm = aabb.normalize(gaussians.get_xyz, sym=True)
            features = torch.cat([features, xyz_norm], dim=1)

        if self.use_cov:
            cov = gaussians.get_covariance()
            features = torch.cat([features, cov], dim=1)

        if self.use_normal:
            scale = gaussians._scaling
            rot = build_rotation(gaussians._rotation)
            normal = torch.gather(
                rot, dim=2,
                index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)
            ).squeeze(-1)
            features = torch.cat([features, normal], dim=1)

        if self.sh_degree > 0:
            dir_pp = (gaussians.get_xyz - camera.camera_center.repeat(n_points, 1))
            if self.cano_view_dir:
                T_fwd = gaussians.fwd_transform
                R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
                dir_pp = torch.matmul(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
                view_noise_scale = self.cfg.get('view_noise', 0.)
                if self.training and view_noise_scale > 0.:
                    # Deterministic per-iter/per-frame view-noise
                    seed = make_subseed(
                        self.base_seed,
                        f"texture/view_noise/{int(iteration) if iteration is not None else -1}/{int(getattr(camera,'frame_id',-1))}"
                    )
                    with torch_rng_context(seed):
                        vn = torch.tensor(
                            augm_rots(view_noise_scale, view_noise_scale, view_noise_scale),
                            dtype=torch.float32, device=dir_pp.device
                        ).transpose(0, 1)
                    dir_pp = torch.matmul(dir_pp, vn)

            dir_pp_normalized = dir_pp / (dir_pp.norm(dim=1, keepdim=True) + 1e-12)
            dir_embed = self.sh_embed(dir_pp_normalized)
            features = torch.cat([features, dir_embed], dim=1)

        if self.non_rigid_dim > 0:
            assert hasattr(gaussians, "non_rigid_feature")
            features = torch.cat([features, gaussians.non_rigid_feature], dim=1)

        return features

    def forward(self, gaussians, camera, iteration=None):
        inp = self.compose_input(gaussians, camera, iteration)
        output = self.mlp(inp)
        color = self.color_activation(output)
        # (facultatif) debug:
        # print("Predicted color stats:", color.min().item(), color.max().item(), color.mean().item())
        return color


def get_texture(cfg, metadata, root_seed=None):
    name = cfg.name
    model_dict = {
        "sh2rgb": SH2RGB,
        "mlp": ColorMLP,
    }
    return model_dict[name](cfg, metadata, root_seed=root_seed)
