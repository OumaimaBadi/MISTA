import types
import torch
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext, contextmanager 
import torch.nn.functional as F



def unwrap_tt_module(scene):
    """
    Retourne le vrai module TT qui contient les cores.
    Compatible: scene.migs_module direct, ou MARS(wrapper).tensorized_model.tt
    """
    m = scene.migs_module
    m = getattr(m, "tensorized_model", m)  # unwrap MARS/MARSPerBlock
    m = getattr(m, "tt", m)               # unwrap adapter (tt)
    return m


@contextmanager
def force_raw_decode_in_scene(scene, tt_module):
    """
    Force TEMPORAIREMENT le rendu/loss à utiliser W RAW (sans MARS),
    même si scene.migs_module est un wrapper MARS.

    Idée: render() appelle scene.convert_gaussians -> update_gaussians_from_migs()
          qui appelle self.migs_module.get_W_for_identity(...)
          donc on monkey-patch juste get_W_for_identity.
    """
    migs = scene.migs_module
    if not hasattr(migs, "get_W_for_identity"):
        # Si jamais, on ne casse rien
        yield
        return

    orig = migs.get_W_for_identity

    def _raw_get_W(self, identity_id, *args, **kwargs):
        # On délègue au TT brut
        return tt_module.get_W_for_identity(int(identity_id), *args, **kwargs)

    try:
        migs.get_W_for_identity = types.MethodType(_raw_get_W, migs)
        yield
    finally:
        migs.get_W_for_identity = orig



def _has_split_core4(tt_module):
    core4_names = [
        "core4_xyz", "core4_scaling", "core4_rotation",
        "core4_dc", "core4_rest", "core4_opacity"
    ]
    return all(hasattr(tt_module, n) for n in core4_names)


def _core4_parts(tt_module):
    parts = []
    for n in [
        "core4_xyz", "core4_scaling", "core4_rotation",
        "core4_dc", "core4_rest", "core4_opacity"
    ]:
        if hasattr(tt_module, n):
            parts.append((n, getattr(tt_module, n)))
    return parts




@contextmanager
def mask_component(tt_module, rank_idx, component_idx):
    assert 0 <= rank_idx <= 3

    with torch.no_grad():
        left = tt_module.tt_tensor_gpu[rank_idx]
        saved_left = left[:, :, component_idx].detach().clone()

        is_last_rank = (rank_idx == 3)

        # ✅ PRIORITÉ: si r4 et core4 split → on masque les slices (le vrai core4 utilisé)
        if is_last_rank and _has_split_core4(tt_module):
            parts = _core4_parts(tt_module)
            saved_parts = [(name, part[component_idx].detach().clone()) for name, part in parts]

            left[:, :, component_idx].zero_()
            for _, part in parts:
                part[component_idx].zero_()

            try:
                yield
            finally:
                left[:, :, component_idx].copy_(saved_left)
                for name, row in saved_parts:
                    getattr(tt_module, name)[component_idx].copy_(row)
            return

        # cas normal: rank_idx=0..2
        if not is_last_rank:
            right = tt_module.tt_tensor_gpu[rank_idx + 1]
            saved_right = right[component_idx, :, :].detach().clone()

            left[:, :, component_idx].zero_()
            right[component_idx, :, :].zero_()

            try:
                yield
            finally:
                left[:, :, component_idx].copy_(saved_left)
                right[component_idx, :, :].copy_(saved_right)
            return

        # rank_idx==3 mais pas de split -> on accepte seulement si tt_tensor_gpu[4] est réellement utilisé
        if len(tt_module.tt_tensor_gpu) >= 5:
            right = tt_module.tt_tensor_gpu[4]
            saved_right = right[component_idx, :, :].detach().clone()

            left[:, :, component_idx].zero_()
            right[component_idx, :, :].zero_()

            try:
                yield
            finally:
                left[:, :, component_idx].copy_(saved_left)
                right[component_idx, :, :].copy_(saved_right)
            return

        raise RuntimeError("rank_idx=3: pas de core4 usable (ni split, ni tt_tensor_gpu[4]).")




def compute_delta_W_single(
    tt_module,
    rank_idx,
    component_idx,
    identity_id=0,
    normalize=True,
    eps=1e-12,
    get_W_fn=None,
):
    """
    ΔW(j,id):
      - si normalize=True: ||Wm - Wo|| / (||Wo|| + eps)
      - sinon:            ||Wm - Wo||

    get_W_fn: optionnel, signature get_W_fn(identity_id)->W.
              Par défaut: tt_module.get_W_for_identity (RAW).
    """
    if get_W_fn is None:
        get_W_fn = lambda pid: tt_module.get_W_for_identity(int(pid))

    with torch.no_grad():
        W_orig = get_W_fn(identity_id)

        with mask_component(tt_module, rank_idx, component_idx):
            W_masked = get_W_fn(identity_id)

        num = torch.norm(W_masked - W_orig)

        if not normalize:
            return float(num.item())

        denom = torch.norm(W_orig) + eps
        return float((num / denom).item())


def compute_delta_W_all(
    tt_module,
    identity_ids,
    rank_names=("r1","r2","r3","r4"),
    normalize=True,
    get_W_fn=None,
):
    """
    results[rank_name] = {
      "deltaW_mean": (rank_size,),
      "deltaW_max":  (rank_size,),
    }
    """
    results = {}
    for rank_idx, rank_name in enumerate(rank_names):
        rank_size = int(tt_module.tt_tensor_gpu[rank_idx].shape[2])
        dW_mean = np.zeros(rank_size, dtype=np.float32)
        dW_max  = np.zeros(rank_size, dtype=np.float32)

        for j in tqdm(range(rank_size), desc=f"ΔW {rank_name}"):
            vals = []
            for pid in identity_ids:
                vals.append(
                    compute_delta_W_single(
                        tt_module, rank_idx, j,
                        identity_id=pid,
                        normalize=normalize,
                        get_W_fn=get_W_fn
                    )
                )
            vals = np.asarray(vals, dtype=np.float32)
            dW_mean[j] = float(vals.mean())
            dW_max[j]  = float(vals.max())

        results[rank_name] = {"deltaW_mean": dW_mean, "deltaW_max": dW_max}
    return results



def _C(iteration, value):
    if value is None:
        return 0.0

    # OmegaConf ListConfig -> list
    try:
        from omegaconf import ListConfig
        if isinstance(value, ListConfig):
            value = list(value)
    except Exception:
        pass

    # constant
    if isinstance(value, (int, float)):
        return float(value)

    # schedule list/tuple
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0


        if len(value) % 2 != 0:
            return float(value[-1])

        out = float(value[1])
        for k in range(0, len(value), 2):
            step = int(value[k])
            val  = float(value[k + 1])
            if iteration >= step:
                out = val
            else:
                break
        return out

    # string numeric
    if isinstance(value, str):
        return float(value)

    return float(value)




def compute_losses_on_samples(
    scene,
    samples,
    iteration,
    lpips_fn=None,
    return_opacity=True,
    decode_mode="raw",   # "raw" or "mars"
):
    """
    Calcule les losses (eval), moyennées sur samples.

    decode_mode:
      - "raw"  : BYPASS MARS pendant le render (W reconstruit depuis TT brut)
      - "mars" : comportement normal (si MARS activé dans scene)
    """
    scene.eval()
    cfg = scene.cfg

    bg_color = [1, 1, 1] if cfg.dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    lam_l1       = _C(iteration, cfg.opt.lambda_l1)
    lam_dssim    = _C(iteration, cfg.opt.lambda_dssim)
    lam_lpips    = _C(iteration, cfg.opt.get("lambda_perceptual", 0.0))
    lam_mask     = _C(iteration, cfg.opt.lambda_mask)
    lam_skinning = _C(iteration, cfg.opt.lambda_skinning)
    lam_aiap_xyz = _C(iteration, cfg.opt.get("lambda_aiap_xyz", 0.0))
    lam_aiap_cov = _C(iteration, cfg.opt.get("lambda_aiap_cov", 0.0))

    mask_loss_type = getattr(cfg.opt, "mask_loss_type", "bce")

    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim, full_aiap_loss

    acc = {
        "l1": 0.0, "dssim": 0.0, "lpips": 0.0, "mask": 0.0,
        "img_total": 0.0,
        "skinning": 0.0, "aiap_xyz": 0.0, "aiap_cov": 0.0,
        "reg_terms_total": 0.0,
        "reg_total": 0.0,
        "quality_total": 0.0,
    }
    reg_terms_acc = {}

    tt_module = unwrap_tt_module(scene)

    n = max(1, len(samples))
    with torch.no_grad():
        ctx = force_raw_decode_in_scene(scene, tt_module) if decode_mode=="raw" else nullcontext()
        with ctx:
            for data in samples:
                gt = data.original_image.cuda(non_blocking=True)

                render_pkg = render(
                    data,
                    iteration,
                    scene,
                    cfg.pipeline,
                    background,
                    compute_loss=True,
                    return_opacity=return_opacity
                )

                img = render_pkg["render"]

                L1 = l1_loss(img, gt) if lam_l1 > 0 else torch.zeros([], device="cuda")
                DSSIM = (1.0 - ssim(img, gt)) if lam_dssim > 0 else torch.zeros([], device="cuda")

                LP = torch.zeros([], device="cuda")
                if lam_lpips > 0 and (lpips_fn is not None):
                    mask_np = data.original_mask.cpu().numpy()
                    coords = np.where(mask_np)
                    if len(coords[0]) > 0:
                        y1, y2 = coords[1].min(), coords[1].max() + 1
                        x1, x2 = coords[2].min(), coords[2].max() + 1
                        fg_img = img[:, y1:y2, x1:x2]
                        fg_gt  = gt[:,  y1:y2, x1:x2]
                        LP = lpips_fn(fg_img, fg_gt, normalize=True).mean()
                    else:
                        LP = torch.ones([], device="cuda")

                LM = torch.zeros([], device="cuda")
                if lam_mask > 0 and ("opacity_render" in render_pkg):
                    op = torch.clamp(render_pkg["opacity_render"], 1e-3, 1.0 - 1e-3)
                    gt_mask = data.original_mask.cuda(non_blocking=True)
                    if mask_loss_type == "bce":
                        LM = F.binary_cross_entropy(op, gt_mask)
                    elif mask_loss_type == "l1":
                        LM = F.l1_loss(op, gt_mask)
                    else:
                        raise ValueError(f"Unknown mask_loss_type={mask_loss_type}")

                L_img_total = lam_l1 * L1 + lam_dssim * DSSIM + lam_lpips * LP + lam_mask * LM

                LS = torch.zeros([], device="cuda")
                if lam_skinning > 0:
                    LS = scene.get_skinning_loss()

                Lxyz = torch.zeros([], device="cuda")
                Lcov = torch.zeros([], device="cuda")
                if (lam_aiap_xyz > 0) or (lam_aiap_cov > 0):
                    Lxyz, Lcov = full_aiap_loss(scene.gaussians, render_pkg["deformed_gaussian"])

                Lreg_terms_total = torch.zeros([], device="cuda")
                loss_reg = render_pkg.get("loss_reg", {})
                for name, val in loss_reg.items():
                    lam = _C(iteration, cfg.opt.get(f"lambda_{name}", 0.0))
                    Lreg_terms_total = Lreg_terms_total + lam * val
                    reg_terms_acc[name] = reg_terms_acc.get(name, 0.0) + float((lam * val).item())

                L_reg_total = lam_skinning * LS + lam_aiap_xyz * Lxyz + lam_aiap_cov * Lcov + Lreg_terms_total
                L_quality = L_img_total + L_reg_total

                acc["l1"] += float(L1.item())
                acc["dssim"] += float(DSSIM.item())
                acc["lpips"] += float(LP.item())
                acc["mask"] += float(LM.item())
                acc["img_total"] += float(L_img_total.item())

                acc["skinning"] += float(LS.item())
                acc["aiap_xyz"] += float(Lxyz.item())
                acc["aiap_cov"] += float(Lcov.item())
                acc["reg_terms_total"] += float(Lreg_terms_total.item())
                acc["reg_total"] += float(L_reg_total.item())
                acc["quality_total"] += float(L_quality.item())

    for k in acc:
        acc[k] /= float(n)
    for name in reg_terms_acc:
        reg_terms_acc[name] /= float(n)

    acc["reg_terms"] = reg_terms_acc
    return acc



def compute_delta_for_component(
    scene,
    samples_by_id,
    rank_idx,
    component_idx,
    iteration,
    lpips_fn=None,
    decode_mode="raw",   # "raw" or "mars"
    normalize_deltaW=True,
):
    """
    Deltas PAR IDENTITÉ pour un composant (rank_idx, component_idx).

    decode_mode contrôle *le monde* des losses (render):
      - "raw": no MARS influence
      - "mars": with MARS
    ΔW reste RAW par défaut (tt_module), ce qui est cohérent avec decode_mode="raw".
    """
    tt_module = unwrap_tt_module(scene)

    out = {}
    with torch.no_grad():
        base = {}
        for pid, samples in samples_by_id.items():
            base[pid] = compute_losses_on_samples(
                scene, samples, iteration,
                lpips_fn=lpips_fn,
                return_opacity=True,
                decode_mode=decode_mode
            )

        with mask_component(tt_module, rank_idx, component_idx):
            masked = {}
            for pid, samples in samples_by_id.items():
                masked[pid] = compute_losses_on_samples(
                    scene, samples, iteration,
                    lpips_fn=lpips_fn,
                    return_opacity=True,
                    decode_mode=decode_mode
                )

        for pid in samples_by_id.keys():
            b = base[pid]
            m = masked[pid]

            dW = compute_delta_W_single(
                tt_module, rank_idx, component_idx,
                identity_id=pid,
                normalize=normalize_deltaW
            )

            out[int(pid)] = {
                "deltaW": dW,
                "delta_l1": abs(m["l1"] - b["l1"]),
                "delta_dssim": abs(m["dssim"] - b["dssim"]),
                "delta_lpips": abs(m["lpips"] - b["lpips"]),
                "delta_mask": abs(m["mask"] - b["mask"]),
                "deltaL_img": abs(m["img_total"] - b["img_total"]),
                "delta_skinning": abs(m["skinning"] - b["skinning"]),
                "delta_aiap_xyz": abs(m["aiap_xyz"] - b["aiap_xyz"]),
                "delta_aiap_cov": abs(m["aiap_cov"] - b["aiap_cov"]),
                "delta_reg_terms_total": abs(m["reg_terms_total"] - b["reg_terms_total"]),
                "deltaL_reg": abs(m["reg_total"] - b["reg_total"]),
                "deltaL_quality": abs(m["quality_total"] - b["quality_total"]),
            }

    return out


def compute_delta_all_components(
    scene,
    samples_by_id,
    iteration,
    lpips_fn=None,
    rank_names=("r1","r2","r3","r4"),
    decode_mode="raw",   
    normalize_deltaW=True,
):
    """
    results[rank]["per_component"][j]["per_id"][pid] = {...}
    results[rank]["summary"] = arrays mean/max

    decode_mode:
      - "raw": ΔLoss sans influence MARS (bypass MARS pendant render)
      - "mars": ΔLoss avec MARS
    """
    tt_module = unwrap_tt_module(scene)
    results = {}

    for rank_idx, rank_name in enumerate(rank_names):
        rank_size = int(tt_module.tt_tensor_gpu[rank_idx].shape[2])

        per_comp = []
        summary = {
            "deltaW_mean": np.zeros(rank_size, dtype=np.float32),
            "deltaW_max":  np.zeros(rank_size, dtype=np.float32),
            "deltaLquality_mean": np.zeros(rank_size, dtype=np.float32),
            "deltaLquality_max":  np.zeros(rank_size, dtype=np.float32),
            "deltaLimg_mean": np.zeros(rank_size, dtype=np.float32),
            "deltaLimg_max":  np.zeros(rank_size, dtype=np.float32),
            "deltaLreg_mean": np.zeros(rank_size, dtype=np.float32),
            "deltaLreg_max":  np.zeros(rank_size, dtype=np.float32),
        }

        for j in tqdm(range(rank_size), desc=f"Δ all {rank_name} ({decode_mode})"):
            per_id = compute_delta_for_component(
                scene, samples_by_id, rank_idx, j, iteration,
                lpips_fn=lpips_fn,
                decode_mode=decode_mode,
                normalize_deltaW=normalize_deltaW
            )
            per_comp.append({"component": j, "per_id": per_id})

            dW = np.array([per_id[pid]["deltaW"] for pid in per_id.keys()], dtype=np.float32)
            dQ = np.array([per_id[pid]["deltaL_quality"] for pid in per_id.keys()], dtype=np.float32)
            dI = np.array([per_id[pid]["deltaL_img"] for pid in per_id.keys()], dtype=np.float32)
            dR = np.array([per_id[pid]["deltaL_reg"] for pid in per_id.keys()], dtype=np.float32)

            summary["deltaW_mean"][j] = float(dW.mean()); summary["deltaW_max"][j] = float(dW.max())
            summary["deltaLquality_mean"][j] = float(dQ.mean()); summary["deltaLquality_max"][j] = float(dQ.max())
            summary["deltaLimg_mean"][j] = float(dI.mean()); summary["deltaLimg_max"][j] = float(dI.max())
            summary["deltaLreg_mean"][j] = float(dR.mean()); summary["deltaLreg_max"][j] = float(dR.max())

        results[rank_name] = {"per_component": per_comp, "summary": summary}

    return results
