# models/deformer/deformer.py
import os
import torch.nn as nn
from utils.general_utils import make_subseed, torch_rng_context

from models.deformer.rigid import get_rigid_deform
from models.deformer.non_rigid import get_non_rigid_deform

class Deformer(nn.Module):
    def __init__(self, cfg, metadata, root_seed=None):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata
        self.root_seed = int(root_seed) if root_seed is not None else 123

        # Non-rigid created under its own RNG namespace
        with torch_rng_context(make_subseed(self.root_seed, "deformer/nonrigid/init")):
            # pass the same root seed down so subparts can create their own namespaces
            self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata, root_seed=self.root_seed)

        # Rigid created under its own RNG namespace
        with torch_rng_context(make_subseed(self.root_seed, "deformer/rigid/init")):
            self.rigid = get_rigid_deform(cfg.rigid, metadata, root_seed=self.root_seed)

    def forward(self, gaussians, camera, iteration, compute_loss=True, outdir=None):
        """
        Apply non-rigid, then rigid deformations.
        If `outdir` is provided, write intermediate PLY snapshots into that folder.
        """
        loss_reg = {}

        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            gaussians.save_ply_playcanvas(f"{outdir}/before_nonrigid.ply")

        # Non-rigid
        deformed_gaussians, loss_non_rigid = self.non_rigid(
            gaussians, iteration, camera, compute_loss
        )
        if outdir is not None:
            deformed_gaussians.save_ply_playcanvas(f"{outdir}/after_nonrigid.ply")

        # Rigid
        deformed_gaussians = self.rigid(deformed_gaussians, iteration, camera)
        if outdir is not None:
            deformed_gaussians.save_ply_playcanvas(f"{outdir}/after_rigid.ply")

        loss_reg.update(loss_non_rigid)
        return deformed_gaussians, loss_reg


def get_deformer(cfg, metadata, root_seed=None):
    return Deformer(cfg, metadata, root_seed=root_seed)
