# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import fix_random, Evaluator, PSEvaluator
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss
import torchvision
import hydra
from omegaconf import OmegaConf
import wandb
import lpips
import torch.nn as nn
import time
from contextlib import contextmanager
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.AutRank.mars import MARS as _MARS
from models.AutRank.mars_perblock import MARSPerBlock as _MARSPerBlock
from scene import TT_MIGS_TYPES

lpips_val_fn = None
def _mars_reg_coef_at(iteration: int, warmup: int, transition: int, base_coef: float) -> float:
    """
    Piecewise schedule for the sparsity regularizer:
      - [0, warmup)          : 0.0
      - [warmup, warmup+tran): linearly ramps 0.0 -> base_coef
      - [warmup+tran, +inf)  : base_coef
    """
    if iteration < warmup:
        return 0.0
    end = warmup + max(0, transition)
    if iteration >= end or transition <= 0:
        return float(base_coef)
    alpha = float(iteration - warmup) / float(transition)
    return float(base_coef) * max(0.0, min(1.0, alpha))


def mars_set_attr(model, name, value):
    for layer in model.modules():
        if isinstance(layer, (_MARS, _MARSPerBlock)):
            setattr(layer, name, value)

def mars_compute_cum_reg(model):
    """Compute cumulative MARS regularization."""
    # Direct access (clearer than looping through modules)
    if isinstance(model, (_MARS, _MARSPerBlock)):
        return model.compute_reg()
    return torch.tensor(0.0, device='cuda')

class DeterministicEpochSampler:
    """
    Yields a deterministic per-epoch permutation of [0..n_items-1].
    Epoch e uses RNG(seed + e) to shuffle, so all runs with the same (seed, n_items)
    see identical orders per epoch, across machines/GPUs.
    """
    def __init__(self, n_items: int, seed: int):
        self.n = int(n_items)
        assert self.n > 0, "Dataset is empty."
        self.seed = int(seed)
        self.epoch = 0
        self.i_in_epoch = 0
        self.order = self._perm_for_epoch(self.epoch)

    def _perm_for_epoch(self, epoch: int):
        rng = random.Random(self.seed + epoch) 
        order = list(range(self.n))
        rng.shuffle(order)
        return order

    def next_index(self):
        idx = self.order[self.i_in_epoch]
        self.i_in_epoch += 1
        if self.i_in_epoch == self.n:
            self.epoch += 1
            self.i_in_epoch = 0
            self.order = self._perm_for_epoch(self.epoch)
        return idx

def preview_epoch_order(epoch_id: int, order: list, k: int = 10):
    k = min(k, len(order))
    head = order[:k]
    tail = order[-k:]
    print(f"[Sampler] Epoch {epoch_id} — first {k}: {head} | last {k}: {tail}")


def set_seed_all(seed: int = 123):
    """
    Make results as reproducible as possible across Python, NumPy, PyTorch (CPU+CUDA),
    and OpenCV. Call this ONCE, as early as possible in your program (before any
    CUDA initialization or model creation).
    """

    # 1) Python hashing & interpreter-level randomness 
    # Python uses hash randomization (affects dict/set iteration order across processes).
    # Setting PYTHONHASHSEED makes that deterministic, which helps when program logic depends—directly or indirectly—on iteration orders.
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) cuDNN kernel selection (PyTorch) 
    # cudnn.benchmark searches for "best" kernels dynamically based on input sizes.
    # That search is non-deterministic; disable it to avoid run-to-run variations.
    torch.backends.cudnn.benchmark = False

    # When possible, force cuDNN to use only deterministic kernels.
    # If a deterministic implementation does not exist for a given op,
    # PyTorch may raise an error later when deterministic algorithms are enforced.
    torch.backends.cudnn.deterministic = True

    #  3) cuBLAS deterministic reductions 
    # Many matmul/reduction paths go through cuBLAS. By default, some are
    # numerically but not bitwise deterministic (e.g., parallel reduction order).
    # This environment variable forces deterministic behavior for those paths.

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Alternative values sometimes seen: ":16:8" (smaller workspace).
    # Larger workspace can be slightly faster but still deterministic.

    # 4) Seed all PRNGs you control 
    # Python's standard RNG
    random.seed(seed)

    # NumPy RNG (used in lots of data pipelines and preprocessing)
    np.random.seed(seed)

    # PyTorch RNG on CPU
    torch.manual_seed(seed)

    # PyTorch RNG on ALL visible CUDA devices
    torch.cuda.manual_seed_all(seed)

    # Disable TF32 to keep matmuls/convs strictly reproducible on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


    # 5) OpenCV RNG (optional, if you use cv2.rand* or augmentations) 
    try:
        import cv2
        # Sets the seed used by OpenCV's random number generator.
        cv2.setRNGSeed(seed)
    except Exception:
        # If OpenCV isn't installed, just ignore.
        pass

    torch.use_deterministic_algorithms(True, warn_only=True)



@contextmanager
def cuda_timer(name: str, store: dict):
    """
    Measure GPU time (ms) for the enclosed block using torch.cuda.Event.
    Stores the elapsed time in store[name] after synchronizing CUDA.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        torch.cuda.synchronize()
        store[name] = float(start.elapsed_time(end))  # milliseconds


# fix_random(42)

def check_nan_loss(name, value, iteration):
    if torch.isnan(value).any() or torch.isinf(value).any():
        print(f"[NAN WARNING] Loss {name} contains NaN or Inf.")
        wandb.log({f'nan_detected/{name}': 1}, step=iteration)

def log_grad_norm(module, name, iteration):
    total_norm = 0.0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    wandb.log({f'grad_norms/{name}': total_norm}, step=iteration)

def log_param_grad(param, name, iteration):
    if param.grad is not None:
        norm = param.grad.data.norm(2).item()
        wandb.log({f'grad_norms/{name}': norm}, step=iteration)
    else:
        wandb.log({f'grad_norms/{name}': 0.0}, step=iteration)

def _grad_nz_frac(p: torch.Tensor) -> float:
    g = p.grad
    if g is None:
        return 0.0
    # fraction of non-zero gradient entries
    return float((g != 0).sum().item()) / max(1, g.numel())


def log_mars_phi_gradients(migs_module, iteration):
    """
    Log gradients of MARS φ logits (mask parameters).
    Each φ logit controls one TT rank's mask.
    """
    # Check if it's a MARS-wrapped module
    if not isinstance(migs_module, (_MARS, _MARSPerBlock)):
        return  # Not MARS, skip
    
    # Check if phi_logits_list exists
    if not hasattr(migs_module, 'phi_logits_list'):
        return
    
    print(f"[MARSGradLogger] Logging φ logits gradients ({len(migs_module.phi_logits_list)} ranks)")
    
    total_sq = 0.0
    for rank_idx, logits in enumerate(migs_module.phi_logits_list):
        if logits.grad is not None:
            grad_norm = logits.grad.data.norm(2).item()
            total_sq += grad_norm ** 2
            
            # Log individual rank gradient
            wandb.log({f"grad_norms/mars/phi_rank{rank_idx}": grad_norm}, step=iteration)
            
            # Log non-zero fraction (useful for sparsity)
            nz_frac = _grad_nz_frac(logits)
            wandb.log({f"grad_mask/phi_rank{rank_idx}_nonzero_frac": nz_frac}, step=iteration)
            
            # Log mean/std of gradients
            wandb.log({
                f"grad_stats/phi_rank{rank_idx}_mean": logits.grad.mean().item(),
                f"grad_stats/phi_rank{rank_idx}_std": logits.grad.std().item(),
            }, step=iteration)
        else:
            wandb.log({f"grad_norms/mars/phi_rank{rank_idx}": 0.0}, step=iteration)
    
    # Log total φ gradient norm
    total_norm = total_sq ** 0.5
    wandb.log({"grad_norms/mars/phi_TOTAL": total_norm}, step=iteration)

def save_mars_rank_histograms(migs_module, exp_dir, iteration, use_masks=True, subfolder="training"):
    """
    Save per-rank histograms of:
      - phi logits  (raw logits)
      - mask values s (the same s used for pruning, via get_mask)

    Directory structure:
      exp_dir/mars_rank_stats/
        rank_r1/iter_000500/logits_hist.pdf
        rank_r1/iter_000500/masks_hist.pdf
        rank_r1/iter_001000/...
        rank_r2/iter_000500/...
        ...

    migs_module : scene.migs_module (MARS or MARSPerBlock wrapper)
    exp_dir     : config.exp_dir
    iteration   : current iteration
    use_masks   : if True, use get_mask(logits) (Binary Concrete s); 
                  if False, use sigmoid(logits) as “soft probs”.
    """
    # We expect migs_module to be a MARS wrapper with phi_logits_list
    if not hasattr(migs_module, "phi_logits_list"):
        return

    base_dir = Path(exp_dir) / "mars_rank_stats" / subfolder  # ← AJOUT subfolder
    base_dir.mkdir(parents=True, exist_ok=True)

    for rank_idx, logits in enumerate(migs_module.phi_logits_list):
        with torch.no_grad():
            logits_cpu = logits.detach().cpu().numpy()

            #  mask values 
            if use_masks:
                if migs_module.training:
                    s = migs_module.get_mask(logits)  # SOFT
                else:
                    s = (logits > migs_module.eval_logits_threshold).float()  # HARD
            else:
                s = torch.sigmoid(logits)

            mask_cpu = s.detach().cpu().numpy()

        rank_name = f"rank_r{rank_idx+1}"
        iter_name = f"iter_{iteration:06d}"
        out_dir = base_dir / rank_name / iter_name
        out_dir.mkdir(parents=True, exist_ok=True)


        # ---- histogram of logits ----
        plt.figure()
        plt.hist(logits_cpu, bins=50)
        plt.title(f"{rank_name} – phi logits – iter {iteration}")
        plt.xlabel("phi logits")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "logits_hist.pdf")
        plt.close()

        # ---- histogram of masks ----
        plt.figure()
        try:
            plt.hist(mask_cpu, bins=50, range=(0.0, 1.0))
        except Exception:
            plt.hist(mask_cpu, bins=50)
        plt.title(f"{rank_name} – mask values – iter {iteration}")
        plt.xlabel("mask value")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / "masks_hist.pdf")
        plt.close()

        idxs = np.arange(logits_cpu.size)  # 0, 1, 2, ..., N-1

        plt.figure()
        plt.stem(idxs, logits_cpu, use_line_collection=True)
        plt.title(f"{rank_name} – phi logits by index – iter {iteration}")
        plt.xlabel("component index")      # x = index de la composante (0..N-1)
        plt.ylabel("phi logit")           # y = valeur exacte du logit φ_i
        plt.tight_layout()
        plt.savefig(out_dir / "logits_by_index.pdf")
        plt.close()

        plt.figure()
        plt.stem(idxs, mask_cpu, use_line_collection=True)
        plt.title(f"{rank_name} – mask values by index – iter {iteration}")
        plt.xlabel("component index")
        plt.ylabel("mask value (s)")
        plt.tight_layout()
        plt.savefig(out_dir / "masks_by_index.pdf")
        plt.close()
        print(f"[MARS] Saved histograms for {rank_name} at iter {iteration} in {out_dir}")



def _log_tt_shapes_once(migs_module, step=0):
    migs_core = getattr(migs_module, "tensorized_model", migs_module)
    if hasattr(migs_core, "tt_blocks") and isinstance(migs_core.tt_blocks, nn.ModuleDict) and len(migs_core.tt_blocks) > 0:
        block_names = [n for n, _ in getattr(migs_core, "block_specs", [])] or list(migs_core.tt_blocks.keys())
        core_tags = ["core0", "core1", "core2", "core3", "core4"]
        logs = {}
        for bname in block_names:
            cores = migs_core.tt_blocks[bname]
            for i, c in enumerate(cores):
                logs[f"tt_shapes/{core_tags[i]}_{bname}"] = str(tuple(c.shape))
        if logs:
            wandb.log(logs, step=step)


SINGLE_BLOCKS = ["xyz", "scaling", "rotation", "dc", "rest", "opacity"]
PARAM_ATTR = {b: f"{b}_param" for b in SINGLE_BLOCKS}

def detect_single_block_tt(m):
    # looks like a TT module
    if not (hasattr(m, "tt_tensor_gpu") and isinstance(m.tt_tensor_gpu, nn.ParameterList) and len(m.tt_tensor_gpu) > 0):
        return None

    # exclude TTPerBlock
    if hasattr(m, "tt_blocks"):
        return None

    present = [b for b in SINGLE_BLOCKS if hasattr(m, PARAM_ATTR[b])]
    missing = [b for b in SINGLE_BLOCKS if b not in present]

    # typical: exactly one missing param = that block is TT-ized
    if len(missing) == 1:
        return missing[0]
    return None


def log_migs_gradients(migs_module, iteration):
    """
    Log gradients for MIGS modules across all possible tensorization types:
    - TTPerBlock (used in TTUltraMIGSModule5DPerBlock, possibly under MARS)
    - Global TT (legacy 5D/6D)
    - Tucker/CP variants
    """
    # Step 1: Get past MARS wrapper if present
    migs_core = getattr(migs_module, "tensorized_model", migs_module)
    
    # Step 2: Get past TensorizedTTAdapter if present
    if hasattr(migs_core, "tt"):
        migs_core = migs_core.tt 
    
    # TTPerBlock 
    if hasattr(migs_core, "tt_blocks") and isinstance(getattr(migs_core, "tt_blocks"), nn.ModuleDict) and len(migs_core.tt_blocks) > 0:
        print(f"[GradLogger] Detected TTPerBlock with {len(migs_core.tt_blocks)} blocks.")
        core_tags = ["core0", "core1", "core2", "core3", "core4"]
        per_core_sq_totals = [0.0] * 5
        grand_sq = 0.0

        if hasattr(migs_core, "block_specs") and migs_core.block_specs:
            block_names = [name for name, _ in migs_core.block_specs]
        else:
            block_names = list(migs_core.tt_blocks.keys())

        for bname in block_names:
            cores = migs_core.tt_blocks[bname]
            block_sq = 0.0
            for i, core in enumerate(cores):
                tag = f"{core_tags[i]}_{bname}"
                if core.grad is not None:
                    gnorm = core.grad.data.norm(2).item()
                    per_core_sq_totals[i] += gnorm * gnorm
                    block_sq += gnorm * gnorm
                    grand_sq += gnorm * gnorm
                else:
                    gnorm = 0.0
                wandb.log({f"grad_norms/tt/{tag}": gnorm}, step=iteration)

            wandb.log({f"grad_mask/core0_nonzero_frac_{bname}": _grad_nz_frac(cores[0])}, step=iteration)
            wandb.log({f"grad_norms/tt/block_total_{bname}": block_sq ** 0.5}, step=iteration)

        for i, s in enumerate(per_core_sq_totals):
            wandb.log({f"grad_norms/tt/{core_tags[i]}_TOTAL": s ** 0.5}, step=iteration)
        wandb.log({"grad_norms/tt/ALL_TOTAL": grand_sq ** 0.5}, step=iteration)
        return


    block = detect_single_block_tt(migs_core)
    if block is not None:
        print(f"[GradLogger] Detected single-block TT (TT only on {block}).")
        prefix = f"tt_{block}"

        # 1) TT cores
        for i, core in enumerate(migs_core.tt_tensor_gpu):
            log_param_grad(core, f"{prefix}/core{i}", iteration)

        # 2) Per-identity params that exist (the other blocks)
        for b in SINGLE_BLOCKS:
            attr = PARAM_ATTR[b]
            if hasattr(migs_core, attr):
                log_param_grad(getattr(migs_core, attr), f"{prefix}/{attr}", iteration)

        # 3) nz grad frac on core0
        c0 = migs_core.tt_tensor_gpu[0]
        wandb.log({f"grad_mask/{prefix}_core0_nonzero_frac": _grad_nz_frac(c0)}, step=iteration)
        return


    # Global TT (5D/6D)
    if hasattr(migs_core, "tt_tensor_gpu") and isinstance(migs_core.tt_tensor_gpu, nn.ParameterList) and len(migs_core.tt_tensor_gpu) > 0:
        print("[GradLogger] Detected global TT (5D/6D).")
        for i, core in enumerate(migs_core.tt_tensor_gpu):
            if i == 4 and any(hasattr(migs_core, name) for name in [
                "core4_xyz", "core4_scaling", "core4_rotation",
                "core4_dc", "core4_rest", "core4_opacity"
            ]):
                continue
            log_param_grad(core, f"tt/core{i}", iteration)

        for name in ["core4_xyz", "core4_scaling", "core4_rotation", "core4_dc", "core4_rest", "core4_opacity"]:
            if hasattr(migs_core, name):
                log_param_grad(getattr(migs_core, name), f"tt/{name}", iteration)

        c0 = migs_core.tt_tensor_gpu[0]
        wandb.log({"grad_mask/core0_nonzero_frac": _grad_nz_frac(c0)}, step=iteration)
        return

    # Case C: Tucker / CP
    if hasattr(migs_core, "U1_xyz"):
        print("[GradLogger] Detected Tucker/CP decomposition.")
        log_param_grad(migs_core.U1_xyz, "tucker/U1_xyz", iteration)
        log_param_grad(migs_core.U1_scaling, "tucker/U1_scaling", iteration)
        log_param_grad(migs_core.U1_rotation, "tucker/U1_rotation", iteration)
        log_param_grad(migs_core.U1_dc, "tucker/U1_dc", iteration)
        log_param_grad(migs_core.U1_rest, "tucker/U1_rest", iteration)
        log_param_grad(migs_core.U1_opacity, "tucker/U1_opacity", iteration)
        if hasattr(migs_core, "core"):
            log_param_grad(migs_core.core, "tucker/core", iteration)
        if hasattr(migs_core, "U2"):
            log_param_grad(migs_core.U2, "tucker/U2", iteration)
        if hasattr(migs_core, "U3"):
            log_param_grad(migs_core.U3, "tucker/U3", iteration)
        return

    # Fallback (only if all above fail)
    print("[GradLogger] WARNING: Unknown MIGS module type, using fallback logging.")
    print(f"[DEBUG] migs_core type: {type(migs_core)}")
    print(f"[DEBUG] migs_core attributes: {dir(migs_core)}")
    for i, p in enumerate(migs_core.parameters()):
        log_param_grad(p, f"migs_param_{i}", iteration)


def C(iteration, value):
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = OmegaConf.to_container(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        value_list = [0] + value
        i = 0
        current_step = iteration
        while i < len(value_list):
            if current_step >= value_list[i]:
                i += 2
            else:
                break
        value = value_list[i - 1]
    return value

def training(config):
    model = config.model
    dataset = config.dataset
    opt = config.opt
    profile_clean = bool(opt.get('profile_clean', False))
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from
    migs_cfg = getattr(config, "migs", None)
    use_mars = bool(getattr(migs_cfg, "use_mars", True)) if migs_cfg is not None else True
    mars_cfg = getattr(migs_cfg, "mars", None) if use_mars else None

    warmup_iters      = int(getattr(mars_cfg, "warmup_iterations", 0))      if mars_cfg else 0
    mask_warmup_iters = int(getattr(mars_cfg, "mask_warmup_iterations", 0)) if mars_cfg else 0




    # LPIPS setup
    lpips_type = config.opt.get('lpips_type', 'vgg')
    loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda()
    global lpips_val_fn
    lpips_val_fn = loss_fn_vgg

    evaluator = PSEvaluator() if dataset.name == 'people_snapshot' else Evaluator()

    first_iter = 0
    gaussians = GaussianModel(model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    if use_mars and mars_cfg is not None:
        mars_set_attr(scene.migs_module, "lambda_sparsity", float(getattr(mars_cfg, "lambda_sparsity", 1.0)))
        mars_set_attr(scene.migs_module, "lambda_binary",  float(getattr(mars_cfg, "lambda_binary", 0.1)))

    if not checkpoint:
        if 0 in checkpoint_iterations:
            scene.save_checkpoint(0)
    scene.train()

    # Load checkpoint / finetune setup
    if checkpoint:
        scene.load_checkpoint(checkpoint)
        if checkpoint and str(getattr(config, "train_mode", "scratch")) == "finetune":
            # 1) Add a new U2 row or select an existing one
            if config.finetune.identity == "new":
                noise = config.finetune.get("noise_scale", 0.05)  # defined in config
                idx = scene.migs_module.add_identity(noise_scale=noise)
            else:
                idx = int(config.finetune.identity)

            # 2) Finetune only the selected identity (and optionally the color MLP)
            scene.migs_module.enable_identity_finetune(
                idx=idx,
                color_mlp=scene.converter.texture,
                lr_id=config.finetune.lr_id,
                lr_tex=config.finetune.lr_tex,
                include_color_in_ft_opt=config.finetune.include_color_in_ft_opt
            )

            # 3) Freeze converter modules except texture MLP if requested
            for p in scene.converter.deformer.parameters():
                p.requires_grad = False
            for p in scene.converter.pose_correction.parameters():
                p.requires_grad = False
            for p in scene.converter.texture.parameters():
                p.requires_grad = bool(config.finetune.include_color_in_ft_opt)

            # Set LR=0 for optimizer groups containing texture params when color MLP is excluded
            if not config.finetune.include_color_in_ft_opt:
                tex_param_ids = {id(p) for p in scene.converter.texture.parameters()}
                for g in scene.converter.optimizer.param_groups:
                    if any(id(p) in tex_param_ids for p in g['params']):
                        g['lr'] = 0.0

            # 4) Use only the target identity’s data
            
            if hasattr(scene.train_dataset, "datasets"):
                data_idx = getattr(config.finetune, "data_identity_idx", None)
                if data_idx is None:
                    data_idx = 0 if len(scene.train_dataset.datasets) == 1 else 0
                scene.train_dataset = scene.train_dataset.datasets[data_idx]
                scene.test_dataset  = scene.test_dataset.datasets[data_idx]

            scene.is_finetune = True
            scene.ft_identity_idx = idx

    # Log TT core shapes once at step 0 (after any changes from checkpoint/finetune)
    _log_tt_shapes_once(scene.migs_module, step=0)


    # Set initial warmup flag once (pre-loop)
    initial_is_warmup = (warmup_iters > 0)
    mars_set_attr(scene.migs_module, "warmup", initial_is_warmup)
    wandb.log({"migs/mars_warmup": int(initial_is_warmup)}, step=0)

    # Background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #  Deterministic per-epoch sampler (create once, after train_dataset is final)
    seed = int(getattr(config, "seed", 123))
    sampler = DeterministicEpochSampler(
        n_items=len(scene.train_dataset),
        seed=seed
    )

    # If resuming (or if first_iter != 1), advance the sampler so its position matches the iteration count.
       # your code sets first_iter to 1reg for fresh runs
    start_iter    = max(1, first_iter)  # how your loop will start
    n             = len(scene.train_dataset)
    already_drawn = start_iter - 1      # how many samples were consumed before this loop starts

    sampler.epoch      = already_drawn // n   # which epoch you’re in
    sampler.i_in_epoch = already_drawn % n    # which index within that epoch
    sampler.order      = sampler._perm_for_epoch(sampler.epoch)
    # Preview the current epoch’s order at (re)start
    preview_epoch_order(sampler.epoch, sampler.order, k=10)

    #data_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        # MARS: warmup flag that mirrors the internal schedule (optional, for logging only) 
        is_warmup = (use_mars and iteration < warmup_iters)
        mars_set_attr(scene.migs_module, "warmup", bool(is_warmup))
        wandb.log({"migs/mars_warmup": int(is_warmup)}, step=iteration)
        # MARS: sync current iteration + decay temperature (BEFORE forward/render) 
        mars_set_attr(scene.migs_module, "current_iteration", iteration)

        if use_mars and iteration in [1, warmup_iters, warmup_iters + 1, warmup_iters + mask_warmup_iters]:
            mars_layers = [m for m in scene.migs_module.modules() if isinstance(m, (_MARS, _MARSPerBlock))]
            print(f"[CHECK] iter={iteration} n_mars_layers={len(mars_layers)} types={[type(m).__name__ for m in mars_layers]}")

            a = mars_compute_cum_reg(scene.migs_module)
            b = scene.migs_module.compute_reg()  # marche seulement si scene.migs_module est bien un wrapper MARS
            diff = (a - b).abs().detach().float().cpu().item()
            print(f"[CHECK] cum_reg={float(a.detach().cpu()):.6f} direct_reg={float(b.detach().cpu()):.6f} diff={diff:.6e}")


        if use_mars:
            if isinstance(scene.migs_module, (_MARS, _MARSPerBlock)):
                if iteration >= warmup_iters:
                    old_temp = scene.migs_module.temperature
                    scene.migs_module.decay_temperature()
                    new_temp = scene.migs_module.temperature
                    
                    if iteration % 100 == 0:  # Log tous les 100 iters
                        print(f"[MARS TEMP] iter={iteration} temp: {old_temp:.6f} → {new_temp:.6f}")
                    
                    wandb.log({"mars/temperature": float(new_temp)}, step=iteration)
        timings = {}
        iter_start = torch.cuda.Event(enable_timing=True); iter_end = torch.cuda.Event(enable_timing=True)
        iter_start.record()

        # Sample a random training item
        with cuda_timer('time/data_fetch_ms', timings):
            # if not data_stack:
            #     data_stack = list(range(len(scene.train_dataset)))
            # data_idx = data_stack.pop(randint(0, len(data_stack)-1))
            data_idx = sampler.next_index()
            data = scene.train_dataset[data_idx]

            # Print once at the start of each new epoch
            if sampler.i_in_epoch == 1:  # after next_index(), the first item in a fresh epoch sets i_in_epoch to 1
                preview_epoch_order(sampler.epoch, sampler.order, k=10)


        # Optional sanity check on person_id
        if hasattr(scene.train_dataset, "cumulative_sizes"):
            expected_person_id = next(i for i, cs in enumerate(scene.train_dataset.cumulative_sizes) if data_idx < cs) - 1
            try:
                if data.person_id != expected_person_id:
                    print(f"[WARN] Mismatch in person_id: got {data.person_id}, expected {expected_person_id}")
            except Exception:
                pass  # no person_id in single-identity mode
        else:
            expected_person_id = 0  # single-identity default

        # Enable renderer debug from a given iteration
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Forward render
        lambda_mask = C(iteration, config.opt.lambda_mask)
        use_mask = lambda_mask > 0.
        with cuda_timer('time/forward_render_ms', timings):
            render_pkg = render(data, iteration, scene, pipe, background, compute_loss=True, return_opacity=use_mask)

        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        )
        with torch.no_grad():
            vis_cnt = int(visibility_filter.sum().item())
            r_vis = radii[visibility_filter]
            if not torch.is_floating_point(r_vis):
                r_vis = r_vis.float()
            if vis_cnt > 0:
                avg_r = float(r_vis.mean().item())
                sum_r2 = float((r_vis * r_vis).sum().item())
            else:
                avg_r = 0.0
                sum_r2 = 0.0

            H, W = image.shape[-2], image.shape[-1]
            timings.update({
                'render/visible_count': vis_cnt,
                'render/avg_radius_px': avg_r,
                'render/sum_r2_px': sum_r2,
                'render/image_pixels': int(H*W),
                'render/views_per_iter': 1,
            })

        # Load ground-truth image to GPU
        with cuda_timer('time/h2d_image_ms', timings):
            gt_image = data.original_image.cuda(non_blocking=True)

        # Optional debug image dumps
        if (not profile_clean) and (iteration <= 1000):
            os.makedirs(f"debug_images/iter_{iteration:04d}", exist_ok=True)
            img = image.detach().cpu().clamp(0, 1)
            torchvision.utils.save_image(img, f"debug_images/iter_{iteration:04d}/render.png")

            gt_img = gt_image.detach().cpu().clamp(0, 1)
            torchvision.utils.save_image(gt_img, f"debug_images/iter_{iteration:04d}/gt.png")

            if "opacity_render" in render_pkg:
                opacity = render_pkg["opacity_render"].detach().cpu().clamp(0, 1)
                torchvision.utils.save_image(opacity, f"debug_images/iter_{iteration:04d}/opacity.png")

            print(f"[SAVED] Iter {iteration} - render, gt, and opacity saved.")

        if not profile_clean:
            print(f"[DEBUG] Image range: min={image.min().item():.4f}, max={image.max().item():.4f}, mean={image.mean().item():.4f}")
        opacity = render_pkg["opacity_render"] if use_mask else None

        # Loss
        with cuda_timer('time/loss_ms', timings):
            # Schedules
            lambda_l1         = C(iteration, config.opt.lambda_l1)
            lambda_dssim      = C(iteration, config.opt.lambda_dssim)
            lambda_perceptual = C(iteration, config.opt.get('lambda_perceptual', 0.))
            lambda_skinning   = C(iteration, config.opt.lambda_skinning)
            lambda_aiap_xyz   = C(iteration, config.opt.get('lambda_aiap_xyz', 0.))
            lambda_aiap_cov   = C(iteration, config.opt.get('lambda_aiap_cov', 0.))

            # Initialize total loss
            loss = torch.zeros([], device="cuda")

            # L1 loss
            if lambda_l1 > 0.:
                with cuda_timer('time/loss_l1_ms', timings):
                    loss_l1 = l1_loss(image, gt_image)
                loss = loss + lambda_l1 * loss_l1
            else:
                loss_l1 = torch.tensor(0., device="cuda")
            check_nan_loss("l1_loss", loss_l1, iteration)

            # DSSIM
            if lambda_dssim > 0.:
                with cuda_timer('time/loss_ssim_ms', timings):
                    loss_dssim = 1.0 - ssim(image, gt_image)
                loss = loss + lambda_dssim * loss_dssim
            else:
                loss_dssim = torch.tensor(0., device="cuda")
            check_nan_loss("loss_dssim", loss_dssim, iteration)

            # Perceptual (LPIPS)
            if lambda_perceptual > 0:
                mask_np = data.original_mask.cpu().numpy()
                mask = np.where(mask_np)
                y1, y2 = mask[1].min(), mask[1].max() + 1
                x1, x2 = mask[2].min(), mask[2].max() + 1

                fg_image    = image[:, y1:y2, x1:x2]
                gt_fg_image = gt_image[:, y1:y2, x1:x2]
                with cuda_timer('time/loss_lpips_ms', timings):
                    loss_perceptual = loss_fn_vgg(fg_image, gt_fg_image, normalize=True).mean()
                loss = loss + lambda_perceptual * loss_perceptual
            else:
                loss_perceptual = torch.tensor(0., device="cuda")
            check_nan_loss("loss_perceptual", loss_perceptual, iteration)

            # Mask loss
            if use_mask:
                with cuda_timer('time/h2d_mask_ms', timings):
                    gt_mask = data.original_mask.cuda(non_blocking=True)
                with cuda_timer('time/loss_mask_ms', timings):
                    op = torch.clamp(opacity, 1e-3, 1.-1e-3)
                    if config.opt.mask_loss_type == 'bce':
                        loss_mask = F.binary_cross_entropy(op, gt_mask)
                    elif config.opt.mask_loss_type == 'l1':
                        loss_mask = F.l1_loss(op, gt_mask)
                    else:
                        raise ValueError("Unknown mask_loss_type")
                loss = loss + lambda_mask * loss_mask
            else:
                loss_mask = torch.tensor(0., device="cuda")
            check_nan_loss("loss_mask", loss_mask, iteration)

            # Skinning regularization
            if lambda_skinning > 0:
                with cuda_timer('time/loss_skinning_ms', timings):
                    loss_skinning = scene.get_skinning_loss()
                loss = loss + lambda_skinning * loss_skinning
            else:
                loss_skinning = torch.tensor(0., device="cuda")
            check_nan_loss("loss_skinning", loss_skinning, iteration)

            # AIAP losses
            if (lambda_aiap_xyz > 0.) or (lambda_aiap_cov > 0.):
                with cuda_timer('time/loss_aiap_ms', timings):
                    loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(scene.gaussians, render_pkg["deformed_gaussian"])
                loss = loss + lambda_aiap_xyz * loss_aiap_xyz + lambda_aiap_cov * loss_aiap_cov
            else:
                loss_aiap_xyz = torch.tensor(0., device="cuda")
                loss_aiap_cov = torch.tensor(0., device="cuda")
            check_nan_loss("loss_aiap_xyz", loss_aiap_xyz, iteration)
            check_nan_loss("loss_aiap_cov", loss_aiap_cov, iteration)

            # Additional regularizers
            loss_reg = render_pkg["loss_reg"]
            for name, value in loss_reg.items():
                lbd = C(iteration, opt.get(f"lambda_{name}", 0.))
                loss = loss + lbd * value

            mars_reg = torch.tensor(0.0, device="cuda")
            if use_mars:
                # Update lambda avec rampe
                if hasattr(scene.migs_module, 'update_lambda_sparsity'):
                    scene.migs_module.update_lambda_sparsity(total_iterations=opt.iterations)
                
                # Compute regularization
                mars_reg = mars_compute_cum_reg(scene.migs_module)
                
                # Add directly (no reg_eff multiplier!)
                loss = loss + mars_reg
                
                # Log
                wandb.log({
                    "mars/lambda_sparsity_current": float(getattr(scene.migs_module, 'lambda_sparsity', 0.0))
                }, step=iteration)




        # Backward
        with cuda_timer('time/backward_ms', timings):
            loss.backward()

        # Gradient logs
        log_grad_norm(scene.converter.deformer.non_rigid, "non_rigid", iteration)
        log_grad_norm(scene.converter.deformer.rigid, "rigid", iteration)
        log_grad_norm(scene.converter.texture, "texture", iteration)
        log_migs_gradients(scene.migs_module, iteration)
        log_mars_phi_gradients(scene.migs_module, iteration)

        # Optimizer step
        with cuda_timer('time/optimizer_ms', timings):
            scene.optimize(iteration)


            mars_log_steps = [
                1,
                500,
                warmup_iters,  # 10k (fin warmup)
                20000,         # 20k
                30000,         # 30k
                40000,         # 40k
                50000,         # 50k (fin training)
                60000,         # 60k
                70000,         # 70k
                80000,         # 80k
            ]

            # Filter invalid / too-large steps
            mars_log_steps = [
                s for s in mars_log_steps
                if s is not None and s > 0 and s <= opt.iterations
            ]

            if use_mars and (iteration in mars_log_steps) and (iteration > warmup_iters):
                print(f"[MARS monitor] Saving local rank histograms at iteration {iteration}")
                save_mars_rank_histograms(
                    scene.migs_module,
                    config.exp_dir,
                    iteration,
                    use_masks=True,
                    subfolder="training"  # True = use get_mask(logits) -> the same s used in adapter
                )
                was_training = scene.migs_module.training
                scene.migs_module.eval()
                
                save_mars_rank_histograms(
                    scene.migs_module,
                    config.exp_dir,
                    iteration,
                    use_masks=True,
                    subfolder="validation"  # ← HARD (threshold)
                )
                
                # Restaurer train mode
                if was_training:
                    scene.migs_module.train()


                total_active = 0
                total_elements = 0
                temps = []
                for layer in scene.migs_module.modules():
                    if isinstance(layer, (_MARS, _MARSPerBlock)) and hasattr(layer, "phi_logits_list"):
                        temps.append(float(getattr(layer, "temperature", 0.0)))
                        for logits in layer.phi_logits_list:
                            probs = torch.sigmoid(logits.detach())
                            total_active += (probs > 0.5).sum().item()
                            total_elements += probs.numel()

                if total_elements > 0:
                    frac_active = total_active / float(total_elements)
                    wandb.log({"mars/active_fraction": frac_active}, step=iteration)

                if temps:
                    wandb.log(
                        {"mars/state_summary/avg_temperature": float(np.mean(temps))},
                        step=iteration
                    )


        # Timing
        iter_end.record()
        torch.cuda.synchronize()
        timings['time/iteration_train_ms'] = float(iter_start.elapsed_time(iter_end))

        # Validation
        with cuda_timer('time/validation_ms', timings):
            with torch.no_grad():
                validation(iteration, testing_iterations, testing_interval, scene, evaluator, (pipe, background))

        # Total timing
        iter_end.record()
        torch.cuda.synchronize()
        timings['time/iteration_total_ms'] = float(iter_start.elapsed_time(iter_end))
        
        # Log weighted contributions of each loss term 
        wandb.log({
            "loss_contrib/l1": lambda_l1 * loss_l1.item(),
            "loss_contrib/ssim": lambda_dssim * loss_dssim.item(),
            "loss_contrib/perceptual": lambda_perceptual * loss_perceptual.item(),
            "loss_contrib/mask": lambda_mask * loss_mask.item(),
            "loss_contrib/skinning": lambda_skinning * loss_skinning.item(),
            "loss_contrib/aiap_xyz": lambda_aiap_xyz * loss_aiap_xyz.item(),
            "loss_contrib/aiap_cov": lambda_aiap_cov * loss_aiap_cov.item(),
            "loss_contrib/mars": mars_reg.item() if use_mars else 0.0,
        }, step=iteration)
        wandb.log({
            "loss_contrib/total": (
                lambda_l1 * loss_l1.item()
                + lambda_dssim * loss_dssim.item()
                + lambda_perceptual * loss_perceptual.item()
                + lambda_mask * loss_mask.item()
                + lambda_skinning * loss_skinning.item()
                + lambda_aiap_xyz * loss_aiap_xyz.item()
                + lambda_aiap_cov * loss_aiap_cov.item()
                + (mars_reg.item() if use_mars else 0.0)
            )
        }, step=iteration)


        # W&B logging
        with torch.no_grad():
            log_loss = {
                'loss/l1_loss': float(loss_l1.item()),
                'loss/ssim_loss': float(loss_dssim.item()),
                'loss/perceptual_loss': float(loss_perceptual.item()),
                'loss/mask_loss': float(loss_mask.item()),
                'loss/loss_skinning': float(loss_skinning.item()),
                'loss/xyz_aiap_loss': float(loss_aiap_xyz.item()),
                'loss/cov_aiap_loss': float(loss_aiap_cov.item()),
                'loss/total_loss': float(loss.item()),
            }
            log_loss.update({'loss/loss_' + k: float(v) for k, v in loss_reg.items()})

            wandb.log({ "iteration": iteration, **log_loss, **timings }, step=iteration)

            ema_loss_for_log = 0.4 * float(loss.item()) + 0.6 * ema_loss_for_log
            if iteration % 10 == 0 and not profile_clean:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

        # Save artifacts
        if (iteration in saving_iterations):
            if not profile_clean:
                print(f"\n[ITER {iteration}] Saving Gaussians")
            scene.save(iteration)

        if iteration in checkpoint_iterations:
            scene.save_checkpoint(iteration)



def validation(iteration, testing_iterations, testing_interval, scene: Scene, evaluator, renderArgs):
    global lpips_val_fn
    if testing_interval > 0:
        if not iteration % testing_interval == 0:
            return
    else:
        if iteration not in testing_iterations:
            return

    scene.eval()
    torch.cuda.empty_cache()
    
    # Import masked metric functions
    from utils.loss_utils import (
        masked_l1_loss,
        masked_psnr,
        masked_ssim,
        compute_union_mask,
        compute_artifact_metrics
    )
    
    validation_configs = (
        {'name': 'test', 'cameras': list(range(len(scene.test_dataset)))},
        {'name': 'train', 'cameras': [idx for idx in range(0, len(scene.train_dataset), len(scene.train_dataset) // 10)]}
    )

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:

            # FULL IMAGE METRICS (sans masque)

            l1_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            

            # MASKED METRICS (avec union)
            l1_masked_test = 0.0
            psnr_masked_test = 0.0
            ssim_masked_test = 0.0
            lpips_masked_test = 0.0
            

            # ARTIFACT METRICS 
            overflow_ratio_test = 0.0
            missing_ratio_test = 0.0

            # Table avec colonnes supplémentaires pour visualiser les masques
            table = wandb.Table(columns=[
                "Frame ID", "Person ID", 
                "GT Mask", "Rendered Mask", "Union Mask", 
                "Opacity", "Rendered", "Ground Truth"
            ])
            
            for idx, data_idx in enumerate(config['cameras']):
                data = getattr(scene, config['name'] + '_dataset')[data_idx]
                render_pkg = render(data, iteration, scene, *renderArgs, compute_loss=False, return_opacity=True)

                # Tensors on GPU for metrics
                img = torch.clamp(render_pkg["render"], 0.0, 1.0)  # [3, H, W]
                gt  = torch.clamp(data.original_image.to(img.device), 0.0, 1.0)  # [3, H, W]
                op  = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)  # [1, H, W]
                

                # GET MASKS FOR UNION (NOUVEAU)
                gt_mask = data.original_mask.to(img.device)  # [1, H, W]
                rendered_mask = op  # [1, H, W]
                
                # Compute UNION mask
                mask_union = compute_union_mask(gt_mask, rendered_mask)  # [1, H, W]
                
                # Compute artifact metrics
                overflow_ratio, missing_ratio = compute_artifact_metrics(gt_mask, rendered_mask)
                overflow_ratio_test += overflow_ratio
                missing_ratio_test += missing_ratio


                # CPU COPIES FOR WANDB
                image_cpu = img.detach().cpu()
                gt_cpu = gt.detach().cpu()
                opacity_cpu = op.detach().cpu()
                
                # Convert masks to RGB for visualization (NOUVEAU)
                gt_mask_rgb = gt_mask.detach().cpu().repeat(3, 1, 1)  # [3, H, W]
                rend_mask_rgb = rendered_mask.detach().cpu().repeat(3, 1, 1)
                union_mask_rgb = mask_union.detach().cpu().repeat(3, 1, 1)

                pid = data.data.get('person_id', -1) if hasattr(data, 'data') else -1
                
                # Add data to table with new mask columns
                table.add_data(
                    data.image_name,
                    pid,
                    wandb.Image(gt_mask_rgb),      
                    wandb.Image(rend_mask_rgb),    
                    wandb.Image(union_mask_rgb),   
                    wandb.Image(opacity_cpu),
                    wandb.Image(image_cpu),
                    wandb.Image(gt_cpu)
                )


                # FULL IMAGE METRICS
                l1_test += l1_loss(img, gt).mean().double()
                metrics_test = evaluator(img, gt)
                psnr_test += metrics_test["psnr"]
                ssim_test += metrics_test["ssim"]
                lpips_test += metrics_test["lpips"]


                # MASKED METRICS (NOUVEAU avec union)
                l1_masked_test += masked_l1_loss(img, gt, mask_union).double()
                psnr_masked_test += masked_psnr(img, gt, mask_union)
                ssim_masked_test += masked_ssim(img, gt, mask_union)
                
                # LPIPS masked (crop à bounding box)
                mask_np = mask_union.cpu().numpy()[0]  # [H, W]
                if mask_np.sum() > 100:  # Au moins 100 pixels
                    coords = np.where(mask_np > 0.5)
                    y1, y2 = coords[0].min(), coords[0].max() + 1
                    x1, x2 = coords[1].min(), coords[1].max() + 1
                    
                    img_crop = img[:, y1:y2, x1:x2]
                    gt_crop = gt[:, y1:y2, x1:x2]
                    
                    # LPIPS
                    lpips_val = lpips_val_fn(
                        img_crop.unsqueeze(0),
                        gt_crop.unsqueeze(0),
                        normalize=True
                    ).item()
                    lpips_masked_test += lpips_val
                else:
                    lpips_masked_test += 1.0  # pénalité si zone trop petite


            # AVERAGE ALL METRICS
            n_samples = len(config['cameras'])
            
 
            psnr_test /= n_samples
            ssim_test /= n_samples
            lpips_test /= n_samples
            l1_test /= n_samples
            

            psnr_masked_test /= n_samples
            ssim_masked_test /= n_samples
            lpips_masked_test /= n_samples
            l1_masked_test /= n_samples
            
            # Artifacts
            overflow_ratio_test /= n_samples
            missing_ratio_test /= n_samples

            def _to_float(x):
                if isinstance(x, torch.Tensor):
                    return float(x.detach().cpu().item())
                return float(x)

            l1_test           = _to_float(l1_test)
            psnr_test         = _to_float(psnr_test)
            ssim_test         = _to_float(ssim_test)
            lpips_test        = _to_float(lpips_test)

            l1_masked_test    = _to_float(l1_masked_test)
            psnr_masked_test  = _to_float(psnr_masked_test)
            ssim_masked_test  = _to_float(ssim_masked_test)
            lpips_masked_test = _to_float(lpips_masked_test)

            overflow_ratio_test = _to_float(overflow_ratio_test)
            missing_ratio_test  = _to_float(missing_ratio_test)



            print(f"\n[ITER {iteration}] Evaluating {config['name']}:")
            print(f"  Full (sans masque):")
            print(f"    L1={l1_test:.4f} PSNR={psnr_test:.2f} SSIM={ssim_test:.4f} LPIPS={lpips_test:.4f}")
            print(f"  Masked (avec UNION):")
            print(f"    L1={l1_masked_test:.4f} PSNR={psnr_masked_test:.2f} SSIM={ssim_masked_test:.4f} LPIPS={lpips_masked_test:.4f}")
            print(f"  Artifacts:")
            print(f"    Overflow={overflow_ratio_test:.2%} Missing={missing_ratio_test:.2%}")


            wandb.log({

                config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                config['name'] + '/loss_viewpoint - psnr': psnr_test,
                config['name'] + '/loss_viewpoint - ssim': ssim_test,
                config['name'] + '/loss_viewpoint - lpips': lpips_test,
                

                config['name'] + '/masked - l1_loss': l1_masked_test,
                config['name'] + '/masked - psnr': psnr_masked_test,
                config['name'] + '/masked - ssim': ssim_masked_test,
                config['name'] + '/masked - lpips': lpips_masked_test,
                

                config['name'] + '/artifacts - overflow_ratio': overflow_ratio_test,
                config['name'] + '/artifacts - missing_ratio': missing_ratio_test,
                

                f"{config['name']}_images/iter_{iteration}": table
            }, step=iteration)

    wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())}, step=iteration)
    wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]}, step=iteration)
    torch.cuda.empty_cache()
    scene.train()



print("Script top-level is running")
@hydra.main(version_base=None, config_path="configs", config_name="config_5d.yaml")
def main(config):
    seed = int(getattr(config, "seed", 123))
    set_seed_all(seed)
    print("Main started")
    print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config
    is_ft = str(getattr(config, "train_mode", "scratch")) == "finetune"
    if is_ft and not config.start_checkpoint:
        raise ValueError("train_mode=finetune requires a valid start_checkpoint.")
    ft_iters = int(config.finetune.get("iterations", config.opt.iterations)) if is_ft else config.opt.iterations

    if is_ft:
        # Align iteration counts for finetuning
        config.opt.iterations = ft_iters
        if not hasattr(config, "migs"):
            config.migs = {}
        config.migs["iterations"] = ft_iters

    if not config.checkpoint_iterations:
        config.checkpoint_iterations = [ft_iters]
        if getattr(config, "test_interval", 0) > ft_iters:
            config.test_interval = max(1, ft_iters // 10)

    os.makedirs(config.exp_dir, exist_ok=True)

    if not config.checkpoint_iterations:
        config.checkpoint_iterations = [config.opt.iterations]
    elif config.opt.iterations not in config.checkpoint_iterations:
        config.checkpoint_iterations.append(config.opt.iterations)

    if not config.save_iterations:
        config.save_iterations = [config.opt.iterations]
    elif config.opt.iterations not in config.save_iterations:
        config.save_iterations.append(config.opt.iterations)

    # W&B init
    wandb_name = config.exp_name
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project=config.get('wandb_project', 'migs_default'),
        entity='badioumaima11-insa-rennes',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )
    wandb.define_metric("iteration")
    wandb.define_metric("time/*", step_metric="iteration")
    wandb.define_metric("loss/*", step_metric="iteration")

    print("Optimizing " + config.exp_dir)

    # Run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    training(config)

    print("\n Training complete.")

if __name__ == "__main__":
    main()