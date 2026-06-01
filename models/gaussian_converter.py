import torch
import torch.nn as nn
import numpy as np
import os
from .deformer import get_deformer
from .pose_correction import get_pose_correction
from .texture import get_texture
#from utils.snapshot_hooks import maybe_dump_gaussians
from utils.general_utils import make_subseed, torch_rng_context
from typing import Optional

class GaussianConverter(nn.Module):
    def __init__(self, cfg, metadata, root_seed: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata
        self.root_seed = int(root_seed) if root_seed is not None else int(getattr(cfg, "seed", 123))

        # Pose correction (deterministic init)
        with torch_rng_context(make_subseed(self.root_seed, "pose_correction/init")):
            self.pose_correction = get_pose_correction(cfg.model.pose_correction, metadata)

        # Deformer (rigid + non-rigid inside) — pass the seed down
        with torch_rng_context(make_subseed(self.root_seed, "deformer/init")):
            self.deformer = get_deformer(cfg.model.deformer, metadata, root_seed=self.root_seed)

        # Texture MLP (deterministic init)
        with torch_rng_context(make_subseed(self.root_seed, "texture/init")):
            self.texture = get_texture(cfg.model.texture, metadata, root_seed=self.root_seed)

        self.optimizer, self.scheduler = None, None
        self.set_optimizer()

    def set_optimizer(self):
        
        opt_params = [
            {'params': self.deformer.rigid.parameters(), 'lr': self.cfg.opt.get('rigid_lr', 0.)},
            # {'params': self.deformer.non_rigid.parameters(), 'lr': self.cfg.opt.get('non_rigid_lr', 0.)},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' not in n],
             'lr': self.cfg.opt.get('non_rigid_lr', 0.)},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' in n],
             'lr': self.cfg.opt.get('nr_latent_lr', 0.), 'weight_decay': self.cfg.opt.get('latent_weight_decay', 0.05)},
            {'params': self.pose_correction.parameters(), 'lr': self.cfg.opt.get('pose_correction_lr', 0.)},
            {'params': [p for n, p in self.texture.named_parameters() if 'latent' not in n],
             'lr': self.cfg.opt.get('texture_lr', 0.)},
            # {'params': [p for n, p in self.texture.named_parameters() if 'latent' in n],
            #  'lr': self.cfg.opt.get('tex_latent_lr', 0.), 'weight_decay': self.cfg.opt.get('latent_weight_decay', 0.05)},
        ]
        self.optimizer = torch.optim.Adam(params=opt_params, lr=0.001, eps=1e-15)

        gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def _export_gate(self, iteration: int):
        """
        Decide if we should export at this iteration and return the target folder or None.
        All export policy is centralized here.
        """
        exp = getattr(self.cfg, "export", None)
        if not exp or not getattr(exp, "enable", False):
            return None

        # Only export on configured iterations
        iters = set(getattr(exp, "iters", []))
        if iteration not in iters:
            return None

        base = exp.dir  # e.g. "${exp_dir}/exports"
        per_iter = bool(getattr(exp, "per_iter_dir", True))
        outdir = os.path.join(base, f"{iteration:06d}") if per_iter else base
        os.makedirs(outdir, exist_ok=True)
        return outdir

    def forward(self, gaussians, camera, iteration, compute_loss=True):
        """
        Orchestrates pose correction -> deformer -> color precompute.
        Optionally exports PLY snapshots depending on config.export.
        """
        loss_reg = {}

        # 1) Pose correction
        camera, loss_reg_pose = self.pose_correction(camera, iteration)

        # 2) Decide export folder once, then pass it down
        outdir = self._export_gate(iteration)

        # 3) Deformer (non-rigid + rigid) + optional snapshots
        deformed_gaussians, loss_reg_deformer = self.deformer(
            gaussians, camera, iteration, compute_loss, outdir=outdir
        )
        loss_reg.update(loss_reg_pose)
        loss_reg.update(loss_reg_deformer)

        # 4) Texture precompute (colors)
        color_precompute = self.texture(deformed_gaussians, camera, iteration=iteration)
        setattr(deformed_gaussians, "colors_precomp", color_precompute)

        # 5) Optional final snapshot with colors (after_texture)
        exp = getattr(self.cfg, "export", None)
        if outdir is not None and exp and getattr(exp, "include_color", True):
            # Color array must be (N,3) in [0,1]
            rgb = color_precompute.detach().float().clamp(0, 1).cpu().numpy()
            deformed_gaussians.save_ply_playcanvas(f"{outdir}/after_texture.ply", rgb=rgb)

        return deformed_gaussians, loss_reg, color_precompute

    # def optimize(self):
    #     grad_clip = self.cfg.opt.get('grad_clip', 0.)
    #     if grad_clip > 0:
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
    #     self.scheduler.step()


    def optimize(self, iteration=None):
        texture_delay = self.cfg.model.texture.get("delay", 0)

        # 1. Freeze texture MLP before delay
        if iteration is not None and iteration < texture_delay:
            for name, param in self.texture.named_parameters():
                param.requires_grad = False
            if iteration == texture_delay - 1:
                print(f"[Converter] Texture MLP frozen (iteration {iteration})")
        else:
            for name, param in self.texture.named_parameters():
                param.requires_grad = True
            if iteration == texture_delay:
                print(f"[Converter] Texture MLP unfrozen (iteration {iteration})")

        # 2. Optim step for all params that require grad
        grad_clip = self.cfg.opt.get('grad_clip', 0.)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

