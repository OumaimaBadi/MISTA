# utils/snapshot_hooks.py
import os
from utils.snapshot_utils import dump_gaussians_npz, should_dump

def maybe_dump_gaussians(tag, gauss, iteration, cfg):
    exp = getattr(cfg, "export", None)
    if not exp or not getattr(exp, "enable", False):
        return
    snapshot_set = set(getattr(exp, "iters", []))
    if not should_dump(int(iteration), snapshot_set):
        return

    out_dir = getattr(exp, "dir", "./snapshots")
    per_iter = bool(getattr(exp, "per_iter_dir", True))

    it_str = f"iter_{int(iteration):06d}"
    if per_iter:
        dump_dir = os.path.join(out_dir, it_str)
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"{tag}.npz")
    else:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{tag}_{it_str}.npz")

    dump_gaussians_npz(
        path=path,
        gauss=gauss,
        tag=tag,
        include_color=bool(getattr(exp, "include_color", False)),
        quat_src_order=str(getattr(exp, "quat_src_order", "wxyz")),
    )
