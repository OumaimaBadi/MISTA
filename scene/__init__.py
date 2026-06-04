# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import GaussianConverter
from scene.gaussian_model import GaussianModel
from dataset import load_dataset
from models.cp_migs_module import CPMIGSModule
from models.mista import MISTA
from utils.snapshot_hooks import maybe_dump_gaussians
from utils.general_utils import make_subseed, torch_rng_context
from models.importance_analysis.frobenius import compute_frobenius_LR
from models.AutRank.tt_mars_adapter_perblock import TensorizedTTAdapterPerBlock
from models.AutRank.tt_mars_adapter import TensorizedTTAdapter
from models.AutRank.mars import MARS

from models.importance_analysis.ablation import compute_delta_W_all, compute_delta_all_components
from models.importance_analysis.reporter import generate_reports

MIGS_CLASS_MAP = {
    "cp": CPMIGSModule,
    "tt5d": MISTA,
}
TT_MIGS_TYPES = (
    "tt4d", "tt5d", "tt6d", "tt5d_perblock",
    "tt5d_xyz", "tt5d_rotation", "tt5d_scaling", "tt5d_dc", "tt5d_rest", "tt5d_opacity","tt5d_color_split", 
)


def _unwrap_tt_module_from_scene(scene):
    """
    Retourne le vrai module TT qui contient les cores.
    Compatible:
      - scene.migs_module direct
      - MARS(wrapper).tensorized_model.tt
    """
    m = scene.migs_module
    m = getattr(m, "tensorized_model", m)  # unwrap MARS
    m = getattr(m, "tt", m)               # unwrap adapter (tt)
    return m


def _normalize_mars_masks(masks):
    """
    Normalise la sortie de get_all_masks() pour fournir un dict:
      {"r1": (n,), "r2": (n,), "r3": (n,), "r4": (n,)}
    Cas gérés:
      1) masks déjà au bon format.
      2) masks per-block: {"xyz": {"r1":..}, "scaling": {...}, ...}
         -> on moyenne sur les blocks (sur les ranks communs).
    """
    if masks is None:
        return None

    # Cas 1: déjà ok
    if all(k in masks for k in ("r1", "r2", "r3", "r4")):
        return masks

    # Cas 2: per-block
    # Exemple attendu: masks["xyz"]["r1"], etc.
    ranks = ("r1", "r2", "r3", "r4")
    acc = {r: [] for r in ranks}

    for block_name, block_masks in masks.items():
        if not isinstance(block_masks, dict):
            continue
        for r in ranks:
            if r in block_masks:
                acc[r].append(block_masks[r])

    out = {}
    for r in ranks:
        if len(acc[r]) == 0:
            continue
        # moyenne torch (sur dim 0) -> tensor (n,)
        stack = torch.stack(acc[r], dim=0)
        out[r] = stack.mean(dim=0)

    if all(k in out for k in ranks):
        return out

    # sinon: format non reconnu
    return None


def _collect_samples_by_id(dataset, n_per_id=3, max_scan=5000, expected_ids=None):
    samples_by_id = {}
    if dataset is None:
        return samples_by_id

    N = len(dataset)
    scan = min(N, int(max_scan))

    for i in range(scan):
        item = dataset[i]
        pid = int(getattr(item, "person_id", 0))
        samples_by_id.setdefault(pid, [])
        if len(samples_by_id[pid]) < n_per_id:
            samples_by_id[pid].append(item)

        # break seulement si on a rempli toutes les identités attendues
        if expected_ids is not None:
            if all(pid in samples_by_id and len(samples_by_id[pid]) >= n_per_id for pid in expected_ids):
                break

    return {pid: lst for pid, lst in samples_by_id.items() if len(lst) > 0}


class Scene:
    gaussians: GaussianModel

    def __init__(self, cfg, gaussians: GaussianModel, save_dir: str):
        """Initialize datasets, Gaussian model, MIGS module (wrapped with MARS if TT), and converter."""
        self.cfg = cfg
        self.save_dir = save_dir
        self.gaussians = gaussians
        self.appearance_identity = None
        mars_cfg = cfg.migs.get("mars", {})
        mars_kwargs = {k: v for k, v in mars_cfg.items() if k in MARS.__init__.__code__.co_varnames}

        self._gaussian_vis_done = set()


        self.root_seed = int(getattr(cfg, "seed", 123))


        if cfg.mode == "predict":
            self.train_dataset = load_dataset(cfg, split="train")
            self.test_dataset  = load_dataset(cfg, split="predict")
            self.metadata      = self.train_dataset.metadata

        elif cfg.mode == "test":
            self.train_dataset = load_dataset(cfg, split="train")
            self.test_dataset = load_dataset(cfg, split="test")
            print(f"[Test mode] Train samples: {len(self.train_dataset)}, Test samples: {len(self.test_dataset)}")
            self.metadata = self.train_dataset.metadata

        elif cfg.mode == "train":
            self.train_dataset = load_dataset(cfg, split="train")
            self.test_dataset = load_dataset(cfg, split="val")
            print(f"[Train mode] Train samples: {len(self.train_dataset)}, Val samples: {len(self.test_dataset)}")
            self.metadata = self.train_dataset.metadata

        else:
            raise ValueError("cfg.mode must be 'train', 'test', or 'predict'.")

        self.appearance_identity = getattr(cfg, "appearance_identity", None)

        def _maybe_pick_identity(ds, idx):
            if not hasattr(ds, "datasets"):
                return ds
            if len(ds.datasets) == 1:
                return ds
            if idx is None:
                return ds
            if idx < 0 or idx >= len(ds.datasets):
                raise IndexError(f"appearance_identity={idx} out of range (0..{len(ds.datasets)-1})")
            return ds.datasets[idx]

        self.cameras_extent = self.metadata.get('cameras_extent', 1.0)
        if self.appearance_identity is not None:
            if self.train_dataset is not None:
                self.train_dataset = _maybe_pick_identity(self.train_dataset, self.appearance_identity)
                new_metadata = self.train_dataset.metadata
                if 'cameras_extent' not in new_metadata:
                    new_metadata['cameras_extent'] = self.cameras_extent
                self.metadata = new_metadata
            if self.test_dataset is not None and cfg.mode != "predict": 
                self.test_dataset = _maybe_pick_identity(self.test_dataset, self.appearance_identity)
                new_metadata = self.test_dataset.metadata
                if 'cameras_extent' not in new_metadata:
                    new_metadata['cameras_extent'] = self.cameras_extent
                self.metadata = new_metadata


        ref_container = self.train_dataset if self.train_dataset is not None else self.test_dataset
        self.ref_dataset = ref_container.datasets[0] if hasattr(ref_container, "datasets") else ref_container
        self.gaussians.create_from_pcd(
            self.ref_dataset.readPointCloud(),
            spatial_lr_scale=self.cameras_extent
        )

        self.migs_type = cfg.migs.type
        migs_class = MIGS_CLASS_MAP.get(self.migs_type)
        if migs_class is None:
            raise ValueError(f"Unsupported MIGS type: {self.migs_type}")

        with torch_rng_context(make_subseed(self.root_seed, "migs/init")):
            base_migs = migs_class(cfg)
            if not getattr(cfg.migs, 'skip_init_from_tensor', False):
                base_migs.init_from_tensor(self.gaussians)
            base_migs.to("cuda")


        use_mars = bool(getattr(cfg.migs, "use_mars", True))  # True par défaut pour rétrocompatibilité

        if use_mars:
            if self.migs_type in TT_MIGS_TYPES and self.migs_type != "tt5d_perblock":
                # Shared masks version
                tt_adapter = TensorizedTTAdapter(base_migs)
                self.migs_module = MARS(tt_adapter, **mars_kwargs)
                print(f"[MARS ENABLED] TT-based MIGS ({self.migs_type}) wrapped with MARS.")
            else:
                self.migs_module = base_migs
                print(f"[MIGS] Using base MIGS (no MARS wrapper).")
        else:
            # Bypass MARS entirely
            self.migs_module = base_migs
            print(f"[MARS DISABLED] Using base MIGS only ({self.migs_type}).")

        self.migs_module.to("cuda")

        skip_identity_expansion = (
            hasattr(cfg.migs, 'skip_init_from_tensor') and 
            getattr(cfg.migs, 'skip_init_from_tensor', False)
        )

        if skip_identity_expansion:
            print("[MIGS] Skipping identity expansion (will be loaded from checkpoint)")
        elif cfg.migs.get("add_identity", False):
            with torch_rng_context(make_subseed(self.root_seed, "migs/add_identity")):
                new_id = self.migs_module.add_identity()
            print(f"[MIGS] Added identity index {new_id}")
        else:
            n_id = 1 if self.appearance_identity is not None else (
                len(self.train_dataset.datasets) if hasattr(self.train_dataset, "datasets") else 1
            )
            print(f"[MIGS] n_id = {n_id}")
            with torch_rng_context(make_subseed(self.root_seed, "migs/expand_first_core")):
                if self.migs_type in TT_MIGS_TYPES:
                    print(f"[DEBUG] Expanding first core via MIGS interface")
                    self.migs_module.expand_first_core(n_id)
                else:
                    # CP/Tucker use expand_U2 instead of expand_first_core
                    self.migs_module.expand_U2(n_id)


        with torch_rng_context(make_subseed(self.root_seed, "converter/init")):
            self.converter = GaussianConverter(cfg, self.metadata, root_seed=self.root_seed).cuda()

        # Optimizers (separate for TT + MARS φ)
        if self.migs_type in TT_MIGS_TYPES:
            # FIX: Handle both with/without MARS
            if use_mars:
                # WITH MARS: MARS → TensorizedTTAdapter → MISTA
                migs_core = getattr(self.migs_module, "tensorized_model", self.migs_module)
                tt_module = migs_core.tt
            else:
                # WITHOUT MARS: Direct MISTA
                tt_module = self.migs_module
            
            # 1) TT optimizer
            tt_module.set_optimizer(cfg.migs)

            # 2) φ optimizer (only if MARS is used)
            if use_mars:
                if hasattr(self.migs_module, "set_phi_optimizer"):
                    self.migs_module.set_phi_optimizer(cfg.migs)
                else:
                    self.migs_module.set_optimizer(cfg.migs)
        else:
            # Non-TT models (CP/Tucker) keep the legacy path
            self.migs_module.set_optimizer(cfg.migs)

        self.is_finetune = False
        self.ft_identity_idx = None


    def update_gaussians_from_migs(self, identity_id: int):
        """Decode factorized parameters for a given identity and update the Gaussian model."""
        W = self.migs_module.get_W_for_identity(identity_id)  # (G, M)
        print(f"[DEBUG] W mean={W.mean():.6f} std={W.std():.6f} for identity {identity_id}")
        current = 0

        def split_dims(size: int):
            nonlocal current
            chunk = W[:, current:current + size]
            current += size
            return chunk

        # Slice per-parameter blocks
        xyz = split_dims(3)
        scaling = split_dims(3)
        rotation = split_dims(4)
        dc = split_dims(3 if self.gaussians.use_sh else 1)
        rest = split_dims(W.shape[1] - current - 1)
        opacity = split_dims(1)

        # Map color/features to SH or feature space
        if self.gaussians.use_sh:
            sh_deg = self.gaussians.max_sh_degree
            sh_total = (sh_deg + 1) ** 2
            features_dc = dc.view(-1, 3, 1)
            features_rest = rest.view(-1, 3, sh_total - 1)
        else:
            features_dc = dc.unsqueeze(-1)      # (G, 1, 1)
            features_rest = rest.unsqueeze(-1)  # (G, F-1, 1)

        # Assign decoded tensors
        self.gaussians._xyz          = xyz.contiguous()
        self.gaussians._scaling      = scaling.contiguous()
        self.gaussians._rotation     = rotation.contiguous()
        self.gaussians._features_dc  = features_dc.contiguous()
        self.gaussians._features_rest= features_rest.contiguous()
        self.gaussians._opacity      = opacity.contiguous()

        # Debug stats
        print(f"[DEBUG] XYZ     min={xyz.min().item():.4f} max={xyz.max().item():.4f} mean={xyz.mean().item():.4f} std={xyz.std().item():.4f}")
        print(f"[DEBUG] Scaling min={scaling.min().item():.4f} max={scaling.max().item():.4f} mean={scaling.mean().item():.4f} std={scaling.std().item():.4f}")
        print(f"[DEBUG] Rot     min={rotation.min().item():.4f} max={rotation.max().item():.4f} mean={rotation.mean().item():.4f} std={rotation.std().item():.4f}")
        print(f"[DEBUG] DC      min={features_dc.min().item():.4f} max={features_dc.max().item():.4f} mean={features_dc.mean().item():.4f} std={features_dc.std().item():.4f}")
        print(f"[DEBUG] Rest    min={features_rest.min().item():.4f} max={features_rest.max().item():.4f} mean={features_rest.mean().item():.4f} std={features_rest.std().item():.4f}")
        print(f"[DEBUG] Opacity min={opacity.min().item():.4f} max={opacity.max().item():.4f} mean={opacity.mean().item():.4f} std={opacity.std().item():.4f}")
        print("[DEBUG] W identity used =", identity_id)



    def maybe_save_gaussian_param_hist(self, iteration: int):
        """
        Local visualization (PDF only) of Gaussian parameter distributions
        at a given iteration.

        Parameters visualized: xyz, scaling, rotation, opacity.
        No WandB logging, only .pdf files stored under exp_dir/gaussian_param_vis.
        """
        migs_cfg = getattr(self.cfg, "migs", None)
        if migs_cfg is None:
            return

        vis_cfg = migs_cfg.get("gaussian_vis", {})

        # List of iterations to visualize, e.g. migs.gaussian_vis.iterations: [1000, 5000, ...]
        iters = vis_cfg.get("iterations", [])
        if iters and (iteration not in iters):
            return

        if iteration in self._gaussian_vis_done:
            return
        self._gaussian_vis_done.add(iteration)

        root_dir = os.path.join(self.save_dir, "gaussian_param_vis")
        os.makedirs(root_dir, exist_ok=True)

        # One subdirectory per iteration
        out_dir = os.path.join(root_dir, f"iter_{iteration:06d}")
        os.makedirs(out_dir, exist_ok=True)

        g = self.gaussians

        with torch.no_grad():
            xyz      = g._xyz.detach().cpu().numpy()              # (G, 3)
            scaling  = g._scaling.detach().cpu().numpy()          # (G, 3)
            rotation = g._rotation.detach().cpu().numpy()         # (G, 4)
            opacity  = g._opacity.detach().cpu().view(-1).numpy() # (G,)

        # 1) XYZ: x, y, z
        plt.figure()
        labels = ["x", "y", "z"]
        for i, lab in enumerate(labels):
            plt.hist(xyz[:, i], bins=64, density=False, alpha=0.4, label=lab)
        plt.legend()
        plt.title(f"XYZ components – iter {iteration}")
        plt.xlabel("value")
        plt.ylabel("density")
        plt.tight_layout()
        pdf_path = os.path.join(out_dir, f"xyz_iter_{iteration:06d}.pdf")
        plt.savefig(pdf_path)
        plt.close()
        print(f"[GAUSS VIS] Saved {pdf_path}")

        # 2) Scaling: sx, sy, sz
        plt.figure()
        labels = ["sx", "sy", "sz"]
        for i, lab in enumerate(labels):
            plt.hist(scaling[:, i], bins=64, density=False, alpha=0.4, label=lab)
        plt.legend()
        plt.title(f"Scaling components – iter {iteration}")
        plt.xlabel("value")
        plt.ylabel("density")
        plt.tight_layout()
        pdf_path = os.path.join(out_dir, f"scaling_iter_{iteration:06d}.pdf")
        plt.savefig(pdf_path)
        plt.close()
        print(f"[GAUSS VIS] Saved {pdf_path}")

        # 3) Rotation: 4 quaternion components
        plt.figure()
        labels = ["r0", "r1", "r2", "r3"]
        for i, lab in enumerate(labels):
            plt.hist(rotation[:, i], bins=64, density=False, alpha=0.4, label=lab)
        plt.legend()
        plt.title(f"Rotation components – iter {iteration}")
        plt.xlabel("value")
        plt.ylabel("density")
        plt.tight_layout()
        pdf_path = os.path.join(out_dir, f"rotation_iter_{iteration:06d}.pdf")
        plt.savefig(pdf_path)
        plt.close()
        print(f"[GAUSS VIS] Saved {pdf_path}")

        # 4) Opacity (logits)
        plt.figure()
        plt.hist(opacity, bins=64, density=False, alpha=0.7)
        plt.title(f"Opacity logits – iter {iteration}")
        plt.xlabel("logit")
        plt.ylabel("density")
        plt.tight_layout()
        pdf_path = os.path.join(out_dir, f"opacity_logit_iter_{iteration:06d}.pdf")
        plt.savefig(pdf_path)
        plt.close()
        print(f"[GAUSS VIS] Saved {pdf_path}")

        # 5) Opacity after sigmoid (0–1)
        opacity_sig = 1.0 / (1.0 + np.exp(-opacity))
        plt.figure()
        plt.hist(opacity_sig, bins=64, density=False, alpha=0.7)
        plt.title(f"Opacity (σ(logit)) – iter {iteration}")
        plt.xlabel("opacity")
        plt.ylabel("density")
        plt.tight_layout()
        pdf_path = os.path.join(out_dir, f"opacity_sig_iter_{iteration:06d}.pdf")
        plt.savefig(pdf_path)
        plt.close()
        print(f"[GAUSS VIS] Saved {pdf_path}")



    def train(self):
        """Put converter in train mode."""
        self.converter.train()
        self.migs_module.train() 
    def eval(self):
        """Put converter in eval mode."""
        self.converter.eval()
        self.migs_module.eval()

    def optimize(self, iteration: int):
        if self.migs_type in TT_MIGS_TYPES:
            # Handle both with/without MARS
            use_mars = bool(getattr(self.cfg.migs, "use_mars", True))
            
            if use_mars:
                # WITH MARS: MARS → TensorizedTTAdapter → MISTA
                migs_core = getattr(self.migs_module, "tensorized_model", self.migs_module)
                tt_module = migs_core.tt
            else:
                # WITHOUT MARS: Direct MISTA
                tt_module = self.migs_module
            
            # 1) TT step
            tt_module.step(iteration)

            # 2) φ step (only if MARS is used)
            if use_mars:
                if hasattr(self.migs_module, "step_phi"):
                    self.migs_module.step_phi(iteration)
                else:
                    self.migs_module.step(iteration)
        else:
            # CP/Tucker: unchanged
            self.migs_module.step(iteration)

        # 3) Converter: unchanged
        self.converter.optimize(iteration)


    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True):
        """
        Select identity (finetune target, fixed appearance, or per-sample) and
        convert Gaussians for the current camera.
        """
        if getattr(self, "is_finetune", False) and (self.ft_identity_idx is not None):
            identity_id = int(self.ft_identity_idx)
        elif self.appearance_identity is not None:
            identity_id = int(self.appearance_identity)
        else:
            identity_id = int(viewpoint_camera.person_id)

        self.update_gaussians_from_migs(identity_id)
        self.maybe_save_gaussian_param_hist(iteration)
        #self.export_core4_visual_analysis(iteration)


        print(
            f"[Person (cfg:{self.appearance_identity} -> local:{identity_id})] "
            f"Opacity min={self.gaussians.get_opacity.min():.4f} "
            f"max={self.gaussians.get_opacity.max():.4f} "
            f"mean={self.gaussians.get_opacity.mean():.4f}"
        )
        print("[DEBUG CONVERT] final identity_id =", identity_id)
        print("[DEBUG CONVERT] camera.person_id =", viewpoint_camera.person_id)
        print("[DEBUG CONVERT] image_name =", viewpoint_camera.image_name)
        return self.converter(self.gaussians, viewpoint_camera, iteration, compute_loss)


    def export_core4_visual_analysis(self, iteration: int):
        """
        Export Core4 basis pour analyse visuelle (PlayCanvas) + metadata (CSV/JSON).
        
        Structure outputs:
        core4_visual_analysis/
            ├── iter_000000/
            │   ├── basis.ply
            │   ├── metadata.csv
            │   ├── metadata.json
            │   └── LEGEND.png
            ├── iter_010000/
            │   ├── basis.ply
            │   ├── metadata.csv
            │   ├── metadata.json
            │   └── LEGEND.png
            └── ...
        """
        key_iters = [0,1, 10000,10001, 20000, 30000, 40000, 50000]
        
        if iteration not in key_iters:
            return
        
        if not hasattr(self, "_core4_visual_done"):
            self._core4_visual_done = set()
        if iteration in self._core4_visual_done:
            return
        self._core4_visual_done.add(iteration)
        
        # Check if TT-based model
        if self.migs_type not in TT_MIGS_TYPES:
            return
        
        # Access TT module (unwrap MARS if present)
        use_mars = bool(getattr(self.cfg.migs, "use_mars", True))
        if use_mars:
            migs_core = getattr(self.migs_module, "tensorized_model", self.migs_module)
            tt_module = getattr(migs_core, "tt", self.migs_module)
        else:
            tt_module = self.migs_module
        
        # Create iteration directory
        base_dir = os.path.join(self.save_dir, "core4_visual_analysis")
        iter_dir = os.path.join(base_dir, f"iter_{iteration:06d}")
        os.makedirs(iter_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"[CORE4 VISUAL] Iteration {iteration}")
        print(f"{'='*60}")
        
        with torch.no_grad():

            xyz = tt_module.core4_xyz.detach().cpu().numpy().squeeze(-1)          # (64, 3)
            scaling = tt_module.core4_scaling.detach().cpu().numpy().squeeze(-1)  # (64, 3)
            rotation = tt_module.core4_rotation.detach().cpu().numpy().squeeze(-1)# (64, 4)
            dc = tt_module.core4_dc.detach().cpu().numpy().squeeze(-1)            # (64, 1)
            rest = tt_module.core4_rest.detach().cpu().numpy().squeeze(-1)        # (64, 31) ← AJOUTÉ !
            opacity = tt_module.core4_opacity.detach().cpu().numpy().squeeze(-1)  # (64, 1)
            
            r4 = xyz.shape[0]
            
            all_params = np.concatenate([xyz, scaling, rotation, dc, rest, opacity], axis=1)  # (64, 43) ← COMPLET !
            importance = np.linalg.norm(all_params, axis=1)
            
            mask_r4 = None
            warmup = int(self.cfg.migs.mars.warmup_iterations) if use_mars else 0
            if use_mars and iteration >= warmup:
                try:
                    masks = self.migs_module.get_all_masks()
                    if "r4" in masks:
                        mask_r4 = masks["r4"].detach().cpu().numpy()
                except:
                    pass
            
            rgb = np.zeros((r4, 3), dtype=np.float32)
            
            if mask_r4 is not None:
                for i in range(r4):
                    m = mask_r4[i]
                    if m > 0.9:
                        rgb[i] = [0.0, 1.0, 0.0]  # Green (active)
                    elif m < 0.1:
                        rgb[i] = [1.0, 0.0, 0.0]  # Red (pruned)
                    else:
                        rgb[i] = [1.0, 1.0, 0.0]  # Yellow (soft)
            else:
                rgb[:] = [0.3, 0.5, 1.0]  # Blue (MARS inactive)
            
            ply_path = os.path.join(iter_dir, "basis.ply")
            self._save_core4_ply(ply_path, xyz, scaling, rotation, opacity, rgb)
            print(f"PLY: {ply_path}")
            

            csv_path = os.path.join(iter_dir, "metadata.csv")
            self._save_core4_csv(csv_path, xyz, scaling, rotation, dc, rest, opacity, importance, mask_r4, r4)  # ← dc, rest ajoutés
            print(f"CSV: {csv_path}")
            
            json_path = os.path.join(iter_dir, "metadata.json")
            self._save_core4_json(json_path, xyz, scaling, rotation, dc, rest, opacity, importance, mask_r4, r4, iteration, use_mars)  # ← dc, rest ajoutés
            print(f"JSON: {json_path}")
            
            legend_path = os.path.join(iter_dir, "LEGEND.png")
            self._save_core4_legend(legend_path, iteration)
            print(f"Legend: {legend_path}")
            
            print(f"\n  Summary:")
            print(f"    Basis count: {r4}")
            print(f"    Importance: min={importance.min():.4f}, max={importance.max():.4f}, mean={importance.mean():.4f}")
            
            if mask_r4 is not None:
                n_active = (mask_r4 > 0.9).sum()
                n_pruned = (mask_r4 < 0.1).sum()
                n_soft = ((mask_r4 >= 0.1) & (mask_r4 <= 0.9)).sum()
                print(f"    MARS status:")
                print(f"      🟢 Active (>0.9):  {n_active}/{r4} ({100*n_active/r4:.1f}%)")
                print(f"      🟡 Soft (0.1-0.9): {n_soft}/{r4} ({100*n_soft/r4:.1f}%)")
                print(f"      🔴 Pruned (<0.1):  {n_pruned}/{r4} ({100*n_pruned/r4:.1f}%)")
            else:
                print(f"    MARS: Not active yet (warmup)")
        
        print(f"{'='*60}\n")


    def _save_core4_ply(self, path, xyz, scaling, rotation, opacity, rgb):
        """
        Save Core4 basis en PLY avec ORDRE STRICT PlayCanvas.
        
        ORDRE EXACT (comme gaussian_model.py):
        x, y, z
        scale_0, scale_1, scale_2
        rot_0, rot_1, rot_2, rot_3
        opacity
        f_dc_0, f_dc_1, f_dc_2
        f_rest_0, f_rest_1, ..., f_rest_44
        """
        from plyfile import PlyData, PlyElement
        
        N = xyz.shape[0]
        f_rest45 = np.zeros((N, 45), dtype=np.float32)
        
        if opacity.ndim == 1:
            opacity = opacity[:, None]
        
        # ORDRE STRICT (comme gaussian_model.py save_ply_playcanvas)
        prop_names = [
            'x','y','z',
            'scale_0','scale_1','scale_2',
            'rot_0','rot_1','rot_2','rot_3',
            'opacity',
            'f_dc_0','f_dc_1','f_dc_2'
        ] + [f'f_rest_{i}' for i in range(45)]
        
        dtype_full = [(n, 'f4') for n in prop_names]
        
        # CONCATENATE DANS LE MÊME ORDRE
        attrib = np.concatenate([
            xyz,        # (N, 3)  x,y,z
            scaling,    # (N, 3)  scale_0,scale_1,scale_2
            rotation,   # (N, 4)  rot_0,rot_1,rot_2,rot_3
            opacity,    # (N, 1)  opacity
            rgb,        # (N, 3)  f_dc_0,f_dc_1,f_dc_2
            f_rest45    # (N, 45) f_rest_0...f_rest_44
        ], axis=1)
        
        elements = np.empty(N, dtype=dtype_full)
        elements[:] = list(map(tuple, attrib))
        el = PlyElement.describe(elements, 'vertex')
        
        PlyData([el]).write(path)


    def _save_core4_csv(self, path, xyz, scaling, rotation, dc, rest, opacity, importance, mask_r4, r4):
        """
        Save Core4 metadata as CSV.
        
        NOUVEAU : Inclut dc et rest_0...rest_30 (31 colonnes SH)
        """
        import pandas as pd
        
        data = {
            'basis_id': np.arange(r4),
            'importance': importance,
            'mars_mask': mask_r4 if mask_r4 is not None else np.full(r4, -1.0),
            
            # XYZ (3 cols)
            'xyz_x': xyz[:, 0],
            'xyz_y': xyz[:, 1],
            'xyz_z': xyz[:, 2],
            
            # Scaling (3 cols)
            'scale_x': scaling[:, 0],
            'scale_y': scaling[:, 1],
            'scale_z': scaling[:, 2],
            
            # Rotation (4 cols)
            'rot_0': rotation[:, 0],
            'rot_1': rotation[:, 1],
            'rot_2': rotation[:, 2],
            'rot_3': rotation[:, 3],
            
            # DC (1 col) ← AJOUTÉ !
            'dc': dc.squeeze(),
            
            # Opacity (1 col)
            'opacity': opacity.squeeze(),
        }
        
        # REST (31 cols) ← AJOUTÉ !
        for i in range(31):
            data[f'rest_{i}'] = rest[:, i]
        
        # Status string
        status = []
        for i in range(r4):
            if mask_r4 is not None:
                m = mask_r4[i]
                if m > 0.9:
                    status.append("active")
                elif m < 0.1:
                    status.append("pruned")
                else:
                    status.append("soft")
            else:
                status.append("unknown")
        data['status'] = status
        
        df = pd.DataFrame(data)
        df.to_csv(path, index=False, float_format='%.6f')


    def _save_core4_json(self, path, xyz, scaling, rotation, dc, rest, opacity, importance, mask_r4, r4, iteration, use_mars):
        """
        Save Core4 metadata as JSON.
        
        NOUVEAU : Inclut dc et rest (31 valeurs SH)
        """
        import json
        
        metadata = {
            "iteration": int(iteration),
            "r4": int(r4),
            "mars_active": bool(mask_r4 is not None),
            "summary": {},
            "basis": []
        }
        
        # Summary
        if mask_r4 is not None:
            n_active = int((mask_r4 > 0.9).sum())
            n_pruned = int((mask_r4 < 0.1).sum())
            n_soft = int(((mask_r4 >= 0.1) & (mask_r4 <= 0.9)).sum())
            
            metadata["summary"] = {
                "active": n_active,
                "soft": n_soft,
                "pruned": n_pruned,
                "importance_min": float(importance.min()),
                "importance_max": float(importance.max()),
                "importance_mean": float(importance.mean())
            }
        else:
            metadata["summary"] = {
                "mars_status": "warmup",
                "importance_min": float(importance.min()),
                "importance_max": float(importance.max()),
                "importance_mean": float(importance.mean())
            }
        
        # Per-basis data
        for i in range(r4):
            basis_info = {
                "id": int(i),
                "importance": float(importance[i]),
                "mask": float(mask_r4[i]) if mask_r4 is not None else -1.0,
                "xyz": [float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])],
                "scaling": [float(scaling[i, 0]), float(scaling[i, 1]), float(scaling[i, 2])],
                "rotation": [float(rotation[i, 0]), float(rotation[i, 1]), float(rotation[i, 2]), float(rotation[i, 3])],
                "dc": float(dc[i]),                           # ← AJOUTÉ !
                "rest": [float(rest[i, j]) for j in range(31)],  # ← AJOUTÉ !
                "opacity": float(opacity[i])
            }
            
            if mask_r4 is not None:
                m = mask_r4[i]
                if m > 0.9:
                    basis_info["status"] = "active"
                elif m < 0.1:
                    basis_info["status"] = "pruned"
                else:
                    basis_info["status"] = "soft"
            else:
                basis_info["status"] = "unknown"
            
            metadata["basis"].append(basis_info)
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)


    def _save_core4_legend(self, path, iteration):
        """Save color legend as PNG."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        legend_items = [
            ("Active (mask > 0.9)", [0.0, 1.0, 0.0]),
            ("Soft (mask 0.1–0.9)", [1.0, 1.0, 0.0]),
            ("Pruned (mask < 0.1)", [1.0, 0.0, 0.0]),
            ("MARS inactive", [0.3, 0.5, 1.0]),
        ]

        
        y = 0
        for label, color in legend_items:
            ax.add_patch(plt.Rectangle((0, y), 1, 0.2, color=color))
            ax.text(1.1, y + 0.1, label, va='center', fontsize=12)
            y += 0.25
        
        ax.set_xlim(0, 3)
        ax.set_ylim(0, y)
        ax.axis('off')
        ax.set_title(
            f'Color Legend (iter {iteration})\n' + 
            'All values (xyz, scaling, rotation, dc, rest, opacity) = PURE Core4 (43 params total)',
            fontsize=14, pad=20
        )
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        

    def get_skinning_loss(self):
        """Return skinning regularization loss from the converter."""
        loss_reg = self.converter.deformer.rigid.regularization()
        return loss_reg.get('loss_skinning', torch.tensor(0.).cuda())

    def save(self, iteration: int):
        """Save current point cloud as a PLY file."""
        point_cloud_path = os.path.join(self.save_dir, f"point_cloud/iteration_{iteration}")
        os.makedirs(point_cloud_path, exist_ok=True)
        self.gaussians.save_ply_playcanvas(os.path.join(point_cloud_path, "point_cloud.ply"))

 
    def save_checkpoint(self, iteration: int):
        """Save a training checkpoint."""
        use_mars = bool(getattr(self.cfg.migs, "use_mars", True))

        checkpoint = {
            "gaussians": self.gaussians.capture(),
            "converter_state": self.converter.state_dict(),
            "converter_opt_state": self.converter.optimizer.state_dict(),
            "converter_scheduler_state": self.converter.scheduler.state_dict(),
            "iteration": iteration,
            "migs_module_state_dict": self.migs_module.state_dict(),
            "migs_type": self.migs_type,
            "use_mars": use_mars, 
        }

        # MARS optimizer/scheduler (only if active)
        if use_mars:
            checkpoint["migs_optimizer"] = (
                self.migs_module.optimizer.state_dict()
                if getattr(self.migs_module, "optimizer", None) is not None else None
            )
            checkpoint["migs_scheduler"] = (
                self.migs_module.scheduler.state_dict()
                if getattr(self.migs_module, "scheduler", None) is not None else None
            )


            checkpoint["mars_state"] = {
                "temperature": getattr(self.migs_module, "temperature", None),
                "current_iteration": getattr(self.migs_module, "current_iteration", 0),
                "warmup_iterations": getattr(self.migs_module, "warmup_iterations", None),
                "mask_warmup_iterations": getattr(self.migs_module, "mask_warmup_iterations", None),
                "temp_gamma": getattr(self.migs_module, "temp_gamma", None),
                "temp_end": getattr(self.migs_module, "temp_end", None),
                "eval_logits_threshold": getattr(self.migs_module, "eval_logits_threshold", None),
            }
            print(f"[Checkpoint] MARS state saved (temp={checkpoint['mars_state']['temperature']:.4f})")
        else:
            checkpoint["migs_optimizer"] = None
            checkpoint["migs_scheduler"] = None
            checkpoint["mars_state"] = None
            print(f"[Checkpoint] MARS disabled — no φ state saved")

        # TT optimizer/scheduler
        if self.migs_type in TT_MIGS_TYPES:
            # Unwrap to get TT module
            if use_mars:
                migs_core = getattr(self.migs_module, "tensorized_model", self.migs_module)
                tt_module = getattr(migs_core, "tt", self.migs_module)
            else:
                tt_module = self.migs_module
            
            # FIX: Use .state_dict(), not the optimizer object!
            tt_opt = getattr(tt_module, "optimizer", None)
            tt_sched = getattr(tt_module, "scheduler", None)
            checkpoint["tt_optimizer"] = tt_opt.state_dict() if tt_opt is not None else None
            checkpoint["tt_scheduler"] = tt_sched.state_dict() if tt_sched is not None else None

        save_path = os.path.join(self.save_dir, f"ckpt{iteration}.pth")
        torch.save(checkpoint, save_path)
        print(f"[Checkpoint] Saved → {save_path}")


    def load_checkpoint(self, path: str):
        """Load a training checkpoint and rebuild modules/optimizers."""
        checkpoint = torch.load(path, map_location="cuda")

        use_mars_ckpt = checkpoint.get("use_mars", None)
        use_mars_cfg = bool(getattr(self.cfg.migs, "use_mars", True))
        
        if use_mars_ckpt is None:
            # Old checkpoint without flag
            print("[Warning] Old checkpoint (no use_mars flag). Using config value.")
            use_mars = use_mars_cfg
        elif use_mars_ckpt != use_mars_cfg:
            # Mismatch → ERROR
            raise RuntimeError(
                f"   CHECKPOINT INCOMPATIBLE!\n"
                f"   Checkpoint: use_mars={use_mars_ckpt}\n"
                f"   Config:     use_mars={use_mars_cfg}\n"
                f"   → Please use matching config or retrain."
            )
        else:
            use_mars = use_mars_cfg  # Validated, safe to use

        # Load converter (unchanged)
        self.converter.load_state_dict(checkpoint["converter_state"])
        self.converter.optimizer.load_state_dict(checkpoint["converter_opt_state"])
        self.converter.scheduler.load_state_dict(checkpoint["converter_scheduler_state"])


        appearance_id = getattr(self.cfg, "appearance_identity", None)
        self.migs_type = checkpoint.get("migs_type", self.migs_type)
        state_dict = checkpoint["migs_module_state_dict"]

 
        if self.migs_type in ("cp", "tucker"):
            # CP/Tucker (no MARS support)
            if self.migs_type == "cp":
                self.migs_module = CPMIGSModule(self.cfg)
            
            self.migs_module.init_from_tensor(self.gaussians)
            
            if "U2" in state_dict:
                n_id_ckpt = state_dict["U2"].shape[0]
                print(f"[CP/Tucker] Keeping all checkpoint identities: U2 has {n_id_ckpt} identities")
                self.migs_module.expand_U2(n_id_ckpt)

                if appearance_id is not None:
                    if int(appearance_id) < 0 or int(appearance_id) >= n_id_ckpt:
                        raise IndexError(
                            f"appearance_identity={appearance_id} invalid. "
                            f"Checkpoint contains identities 0..{n_id_ckpt-1}"
                        )
            
            self.migs_module.load_state_dict(state_dict, strict=False)
            self.migs_module.to("cuda")

        elif self.migs_type in TT_MIGS_TYPES:
            # Step 1: Create BASE TT module
            class_map = {
                "tt5d": MISTA,
            }
            base_tt = class_map[self.migs_type](self.cfg)
            if not getattr(self.cfg.migs, 'skip_init_from_tensor', False):
                base_tt.init_from_tensor(self.gaussians)
            
            # Handle identity expansion
            if self.migs_type == "tt5d_perblock":
                core0_key = "tt_blocks.xyz.0"
            else:
                core0_key = "tt_tensor_gpu.0"
            
            # Alternative key names for old checkpoints
            if core0_key not in state_dict:
                core0_key = next((k for k in state_dict if k.startswith(("tt_tensor_gpu.0", "tt_cores.0", "tt_blocks"))), None)
            
            if core0_key is None:
                raise KeyError("Missing TT first core key in checkpoint.")
            
            if appearance_id is not None:
                # Single identity mode
                if self.migs_type == "tt5d_perblock":
                    # Update all blocks
                    for block_name in ["xyz", "scaling", "rotation", "dc", "rest", "opacity"]:
                        key = f"tt_blocks.{block_name}.0"
                        if key in state_dict:
                            full_core0 = state_dict[key]
                            state_dict[key] = full_core0[:, appearance_id:appearance_id + 1, :]
                else:
                    # Single global core0
                    full_core0 = state_dict[core0_key]
                    state_dict[core0_key] = full_core0[:, appearance_id:appearance_id + 1, :]
            else:
                # Multi-identity mode
                n_id_ckpt = state_dict[core0_key].shape[1]
                base_tt.expand_first_core(n_id_ckpt)
            
            # Step 2: Wrap with MARS if needed
            if use_mars:
                # Import MARS classes
                
                # Get MARS config
                mars_cfg = self.cfg.migs.get("mars", {})
                mars_kwargs = {k: v for k, v in mars_cfg.items() 
                            if k in MARS.__init__.__code__.co_varnames}
                
                adapter = TensorizedTTAdapter(base_tt)
                self.migs_module = MARS(adapter, **mars_kwargs)
                
                print(f"[Checkpoint] Rebuilt MARS wrapper for {self.migs_type}")
            else:
                # No MARS, use TT directly
                self.migs_module = base_tt
                print(f"[Checkpoint] Using plain TT module ({self.migs_type})")
            
            # Step 3: Load state dict
            missing, unexpected = self.migs_module.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[Warning] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"[Warning] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            
            self.migs_module.to("cuda")
            
            # Step 4: Restore MARS internal state
            if use_mars and checkpoint.get("mars_state") is not None:
                mars_state = checkpoint["mars_state"]
                # Set attributes on MARS module
                for k, v in mars_state.items():
                    if v is not None:
                        setattr(self.migs_module, k, v)
                print(f"[Checkpoint] Restored MARS state (temp={mars_state.get('temperature'):.4f})")
        
        else:
            raise ValueError(f"Unsupported MIGS type: {self.migs_type}")


        if self.migs_type in TT_MIGS_TYPES:
            # Get TT module (unwrap if MARS)
            if use_mars:
                tt_module = self.migs_module.tensorized_model.tt
            else:
                tt_module = self.migs_module
            
            # Restore TT optimizer
            tt_module.set_optimizer(self.cfg.migs)
            if checkpoint.get("tt_optimizer") and getattr(tt_module, "optimizer", None):
                tt_module.optimizer.load_state_dict(checkpoint["tt_optimizer"])
            if checkpoint.get("tt_scheduler") and getattr(tt_module, "scheduler", None):
                tt_module.scheduler.load_state_dict(checkpoint["tt_scheduler"])
            
            # Restore MARS φ optimizer
            if use_mars:
                if hasattr(self.migs_module, "set_phi_optimizer"):
                    self.migs_module.set_phi_optimizer(self.cfg.migs)
                else:
                    self.migs_module.set_optimizer(self.cfg.migs)
                
                if checkpoint.get("migs_optimizer") and getattr(self.migs_module, "optimizer", None):
                    self.migs_module.optimizer.load_state_dict(checkpoint["migs_optimizer"])
                if checkpoint.get("migs_scheduler") and getattr(self.migs_module, "scheduler", None):
                    self.migs_module.scheduler.load_state_dict(checkpoint["migs_scheduler"])
                
                print("[Checkpoint] MARS φ optimizer restored")
        
        else:
            # CP/Tucker
            self.migs_module.set_optimizer(self.cfg.migs)
            if checkpoint.get("migs_optimizer"):
                self.migs_module.optimizer.load_state_dict(checkpoint["migs_optimizer"])

        print(f"[Checkpoint] Loaded iteration {checkpoint['iteration']} (MARS={'ON' if use_mars else 'OFF'})")
