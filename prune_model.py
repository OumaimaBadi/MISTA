"""
Physical pruning of a TT + MARS MIGS model after training.

Outputs:
  - *_pruned.pth          : keeps the original migs_module_state_dict structure (MARS-wrapped keys)
  - *_pruned_nomars.pth   : exports a pure TT checkpoint (recommended for inference)

Usage:
  python prune_model.py --ckpt /path/to/ckpt80000.pth --threshold 0.5
"""

import os
import argparse
import copy
import torch
from omegaconf import OmegaConf

from scene import Scene
from scene.gaussian_model import GaussianModel




def load_run_cfg(run_dir: str):
    hydra_cfg = os.path.join(run_dir, ".hydra", "config.yaml")
    if os.path.exists(hydra_cfg):
        cfg = OmegaConf.load(hydra_cfg)
        print(f" Loaded Hydra config: {hydra_cfg}")
        return cfg

    cfg_yaml = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg_yaml):
        cfg = OmegaConf.load(cfg_yaml)
        print(f" Loaded config: {cfg_yaml}")
        return cfg

    raise FileNotFoundError(f"Could not find config in {run_dir} (.hydra/config.yaml or config.yaml)")


def find_key_ending(sd: dict, suffix: str):
    """Return a key in sd that endswith(suffix). Prefer the shortest match."""
    cands = [k for k in sd.keys() if k.endswith(suffix)]
    if not cands:
        return None
    return sorted(cands, key=len)[0]


def get_prefix(sd: dict, suffix: str):
    k = find_key_ending(sd, suffix)
    if k is None:
        return None
    return k[:-len(suffix)]


def export_nomars_state_dict(mars_sd: dict) -> dict:
    """
    Keep only pure-TT params/buffers.
    If keys are MARS-wrapped like 'tensorized_model.tt.tt_tensor_gpu.0',
    strip 'tensorized_model.tt.' so they become 'tt_tensor_gpu.0'.
    Drop MARS-only params (phi logits, etc.).
    """
    out = {}
    strip_prefix = "tensorized_model.tt."

    for k, v in mars_sd.items():
        kk = k
        if kk.startswith(strip_prefix):
            kk = kk[len(strip_prefix):]

        # keep only TT cores + core4 slices + perm buffers
        if (kk.startswith("tt_tensor_gpu.")
            or kk.startswith("core4_")
            or kk in ("perm", "inv_perm")):
            out[kk] = v

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    thr = float(args.threshold)
    device = args.device

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    run_dir = os.path.dirname(ckpt_path)
    out_pruned = ckpt_path.replace(".pth", "_pruned.pth")
    out_nomars = ckpt_path.replace(".pth", "_pruned_nomars.pth")

    print("\n" + "=" * 80)
    print("🔪 PHYSICAL PRUNING - POST-TRAINING (TT + MARS)")
    print("=" * 80)
    print(f"  Input:     {ckpt_path}")
    print(f"  Threshold: {thr}")
    print(f"  Output:    {out_pruned}")
    print(f"  Output2:   {out_nomars}  (recommended for inference)")
    print(f"  Device:    {device}")
    print("=" * 80 + "\n")

    print("[1/5] Loading checkpoint + config...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = load_run_cfg(run_dir)

    OmegaConf.set_struct(cfg, False)
    cfg.mode = "test"
    cfg.dataset.preload = False

    # sanity for MultiPersonZJUMoCap
    if getattr(cfg.dataset, "name", "") == "MultiPersonZJUMoCap":
        names = getattr(cfg.dataset, "names", None)
        print(f"  cfg.dataset.names = {names}")
        if not names:
            raise ValueError("cfg.dataset.names is missing/empty; MultiPersonZJUMoCap needs it.")

    # must be a MARS run to prune from masks
    use_mars_cfg = bool(getattr(cfg.migs, "use_mars", True))
    if not use_mars_cfg:
        raise RuntimeError("cfg.migs.use_mars=false in this run; no MARS masks to prune from.")

    print("[2/5] Building Scene (no scene.load_checkpoint)...")
    gaussians = GaussianModel(cfg.model.gaussian)
    scene = Scene(cfg, gaussians, run_dir)
    scene.eval()

    # ---- CRITICAL: load MIGS/MARS state_dict directly, NOT scene.load_checkpoint()
    print("[3/5] Loading MIGS/MARS state_dict into scene.migs_module...")
    migs_sd = checkpoint.get("migs_module_state_dict", None)
    if migs_sd is None:
        raise KeyError("Checkpoint has no 'migs_module_state_dict'.")

    missing, unexpected = scene.migs_module.load_state_dict(migs_sd, strict=False)
    print(f"  Loaded. missing={len(missing)} unexpected={len(unexpected)}")

    # optional: restore any saved MARS runtime state if you stored it
    # (safe even if absent)
    mars_state = checkpoint.get("mars_state", None)
    if mars_state and hasattr(scene.migs_module, "load_mars_state"):
        scene.migs_module.load_mars_state(mars_state)

    print(f"\n[4/5] Analyzing masks (threshold={thr})...")
    masks = scene.migs_module.get_all_masks()

    stats = {}
    for rname in ["r1", "r2", "r3", "r4"]:
        m = masks[rname].detach().float()
        k = (m > thr)
        stats[rname] = (int(k.sum().item()), int(k.numel()))

    total_active = sum(v[0] for v in stats.values())
    total_all = sum(v[1] for v in stats.values())

    print("\n  BEFORE pruning:")
    for rname in ["r1", "r2", "r3", "r4"]:
        a, t = stats[rname]
        print(f"    {rname}: {a}/{t} active ({100*a/t:.1f}%)")
    print(f"\n  Total: {total_active}/{total_all} active ({100*total_active/total_all:.1f}%)")
    print(f"         {total_all - total_active} ranks will be removed")

    print("\n  Physical pruning cores via export_pruned()...")
    migs_core = getattr(scene.migs_module, "tensorized_model", scene.migs_module)  # TensorizedTTAdapter
    if not hasattr(migs_core, "export_pruned"):
        raise AttributeError("Your tensorized model has no export_pruned().")

    pruned_cores = migs_core.export_pruned(scene.migs_module)

    print("  Cores pruned physically")
    print("\n  AFTER pruning:")
    for i, c in enumerate(pruned_cores):
        print(f"    core[{i}]: {tuple(c.shape)}")

    pruned_ranks = [
        pruned_cores[0].shape[-1],
        pruned_cores[1].shape[-1],
        pruned_cores[2].shape[-1],
        pruned_cores[3].shape[-1],
    ]
    print(f"\n  Pruned ranks r1..r4: {pruned_ranks}")

    print("\n[5/5] Updating checkpoint(s) and saving...")

    ckptA = copy.deepcopy(checkpoint)
    sdA = ckptA["migs_module_state_dict"]

    # detect prefixes automatically
    pref_tt = get_prefix(sdA, "tt_tensor_gpu.0")
    if pref_tt is None:
        raise KeyError("Could not find any key ending with 'tt_tensor_gpu.0' in migs_module_state_dict.")

    pref_c4 = get_prefix(sdA, "core4_xyz") or pref_tt  # usually same prefix

    sdA[pref_tt + "tt_tensor_gpu.0"] = pruned_cores[0]
    sdA[pref_tt + "tt_tensor_gpu.1"] = pruned_cores[1]
    sdA[pref_tt + "tt_tensor_gpu.2"] = pruned_cores[2]
    sdA[pref_tt + "tt_tensor_gpu.3"] = pruned_cores[3]

    core4 = pruned_cores[4]  # (r4, 43, 1)
    sdA[pref_c4 + "core4_xyz"]      = core4[:, 0:3,   :]
    sdA[pref_c4 + "core4_scaling"]  = core4[:, 3:6,   :]
    sdA[pref_c4 + "core4_rotation"] = core4[:, 6:10,  :]
    sdA[pref_c4 + "core4_dc"]       = core4[:, 10:11, :]
    sdA[pref_c4 + "core4_rest"]     = core4[:, 11:42, :]
    sdA[pref_c4 + "core4_opacity"]  = core4[:, 42:43, :]

    ckptA["is_pruned"] = True
    ckptA["prune_threshold"] = thr
    ckptA["pruned_ranks"] = pruned_ranks

    torch.save(ckptA, out_pruned)
    print(f"  Saved → {out_pruned}")

    ckptB = copy.deepcopy(checkpoint)
    ckptB["use_mars"] = False
    ckptB["is_pruned"] = True
    ckptB["prune_threshold"] = thr
    ckptB["pruned_ranks"] = pruned_ranks

    sd_nomars = export_nomars_state_dict(checkpoint["migs_module_state_dict"])

    # overwrite with pruned cores (pure TT keys)
    sd_nomars["tt_tensor_gpu.0"] = pruned_cores[0]
    sd_nomars["tt_tensor_gpu.1"] = pruned_cores[1]
    sd_nomars["tt_tensor_gpu.2"] = pruned_cores[2]
    sd_nomars["tt_tensor_gpu.3"] = pruned_cores[3]

    sd_nomars["core4_xyz"]      = core4[:, 0:3,   :]
    sd_nomars["core4_scaling"]  = core4[:, 3:6,   :]
    sd_nomars["core4_rotation"] = core4[:, 6:10,  :]
    sd_nomars["core4_dc"]       = core4[:, 10:11, :]
    sd_nomars["core4_rest"]     = core4[:, 11:42, :]
    sd_nomars["core4_opacity"]  = core4[:, 42:43, :]

    ckptB["migs_module_state_dict"] = sd_nomars

    # drop optimizer/scheduler states (not needed for inference)
    for k in ["migs_optimizer", "migs_scheduler", "mars_optimizer", "mars_scheduler"]:
        if k in ckptB:
            ckptB.pop(k)

    torch.save(ckptB, out_nomars)
    print(f"  Saved → {out_nomars}")

    print("\n" + "=" * 80)
    print(" PRUNING COMPLETE!")
    print("=" * 80)
    print(f"  Active kept total:    {total_active}/{total_all} ({100*total_active/total_all:.1f}%)")
    print(f"  Pruned ranks r1..r4:  {pruned_ranks}")
    print("\nTo test (recommended):")
    print(f"  python test.py mode=test migs.use_mars=false load_ckpt={out_nomars}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
