# utils/snapshot_utils.py
import os
import numpy as np
import torch

def tensor_to_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def _ensure_2d_features(fdc, frest, use_sh: bool, sh_deg: int):
    """
    Remet en (G,C) pour l’export .npz, quel que soit le format interne.
    - sans SH : fdc (G,1[,1])  -> (G,1) ; frest (G,F-1[,1]) -> (G,F-1)
    - avec SH : fdc (G,3[,1])  -> (G,3) ; frest (G,3,(sh^2-1)[,1]) -> (G, 3*(sh^2-1))
    """
    if fdc is not None:
        fdc = np.asarray(fdc)
        if fdc.ndim >= 3:
            # ex: (G,1,1) ou (G,3,1)
            fdc = fdc.reshape(fdc.shape[0], -1)
        elif fdc.ndim == 2:
            pass
        elif fdc.ndim == 1:
            fdc = fdc[:, None]
        else:
            raise ValueError(f"Unexpected features_dc ndim={fdc.ndim}")

    if frest is not None:
        frest = np.asarray(frest)
        if not use_sh:
            # attendu (G, F-1[,1]) -> (G, F-1)
            if frest.ndim >= 3:
                frest = frest.reshape(frest.shape[0], -1)
            elif frest.ndim == 2:
                pass
            elif frest.ndim == 1:
                frest = frest[:, None]
            else:
                raise ValueError(f"Unexpected features_rest ndim={frest.ndim} (no SH)")
        else:
            # attendu (G, 3, (sh^2-1)[,1]) -> (G, 3*(sh^2-1))
            if frest.ndim == 4 and frest.shape[-1] == 1:
                frest = frest[..., 0]            # (G, 3, (sh^2-1))
            if frest.ndim == 3:
                G, C, K = frest.shape
                if C != 3:
                    raise ValueError(f"features_rest avec SH doit avoir C=3, got {C}")
                frest = frest.reshape(G, C * K)
            elif frest.ndim == 2:
                # déjà aplati (G, 3*(sh^2-1))
                pass
            else:
                raise ValueError(f"Unexpected features_rest ndim={frest.ndim} (with SH)")
    return fdc, frest

def _rotation_to_wxyz(rot, src_order: str):
    """
    Retourne un tableau (G,4) en WXYZ.
    src_order in {'wxyz','xyzw'}.
    """
    r = np.asarray(rot)
    if r.ndim != 2 or r.shape[1] != 4:
        raise ValueError(f"rotation doit être (G,4), got {r.shape}")
    if src_order == "wxyz":
        return r
    elif src_order == "xyzw":
        # (x,y,z,w) -> (w,x,y,z)
        x, y, z, w = r[:,0], r[:,1], r[:,2], r[:,3]
        return np.stack([w, x, y, z], axis=1)
    else:
        raise ValueError(f"src_order inconnu: {src_order}")

def dump_gaussians_npz(path,
                       gauss,
                       *,
                       tag=None,
                       include_color=False,
                       camera=None,
                       quat_src_order="wxyz"):
    """
    Sauve un snapshot .npz compatible avec ton viewer:
      xyz (G,3), scaling (G,3), rotation (G,4, ordre=WXYZ), opacity (G,1),
      features_dc (2D), features_rest (2D),
      use_sh(bool), sh_deg(int).

    Args:
        quat_src_order: 'wxyz' si tes quaternions internes sont déjà WXYZ,
                        'xyzw' si tu stockes (x,y,z,w) et qu’on doit convertir.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Tensors -> numpy
    xyz      = tensor_to_np(getattr(gauss, "_xyz", None))           # (G,3)
    scaling  = tensor_to_np(getattr(gauss, "_scaling", None))       # (G,3) (log-scale 3DGS)
    rotation = tensor_to_np(getattr(gauss, "_rotation", None))      # (G,4) (src order)
    opacity  = tensor_to_np(getattr(gauss, "_opacity", None))       # (G,1)
    fdc      = tensor_to_np(getattr(gauss, "_features_dc", None))   # (G,1,1) / (G,3,1) / etc.
    frest    = tensor_to_np(getattr(gauss, "_features_rest", None)) # (G,F-1,1) ou (G,3,(sh^2-1)[,1])

    use_sh = bool(getattr(gauss, "use_sh", False))
    sh_deg = int(getattr(gauss, "max_sh_degree", 0) if use_sh else 0)

    if xyz is None or scaling is None or rotation is None:
        raise ValueError("xyz, scaling, rotation sont requis")

    # Rotation → WXYZ
    rotation = _rotation_to_wxyz(rotation, quat_src_order)

    # Opacity : squeeze en (G,1) si (G,1,...) par mégarde
    if opacity is not None:
        opacity = np.asarray(opacity)
        if opacity.ndim > 2 and opacity.shape[-1] == 1:
            opacity = opacity.reshape(opacity.shape[0], 1)
        elif opacity.ndim == 2:
            pass
        elif opacity.ndim == 1:
            opacity = opacity[:, None]
        else:
            raise ValueError(f"Unexpected opacity ndim={opacity.ndim}")

    # Features en 2D
    fdc, frest = _ensure_2d_features(fdc, frest, use_sh, sh_deg)

    save_dict = dict(
        xyz=xyz.astype(np.float32),
        scaling=scaling.astype(np.float32),
        rotation=rotation.astype(np.float32),       # WXYZ
        opacity=None if opacity is None else opacity.astype(np.float32),
        features_dc=None if fdc   is None else fdc.astype(np.float32),
        features_rest=None if frest is None else frest.astype(np.float32),
        use_sh=np.array([use_sh], dtype=np.bool_),
        sh_deg=np.array([sh_deg], dtype=np.int64),
    )

    # Optionnel : couleur (pré-computée par texture)
    if include_color and hasattr(gauss, "colors_precomp") and gauss.colors_precomp is not None:
        save_dict["colors_precomp"] = tensor_to_np(gauss.colors_precomp).astype(np.float32)  # (G,3)

    # Tag (métadonnée)
    if tag is not None:
        save_dict["tag"] = np.array([str(tag)], dtype=object)

    np.savez_compressed(path, **{k:v for k,v in save_dict.items() if v is not None})
    return path

def should_dump(iteration: int, snapshot_iters: set) -> bool:
    return iteration in snapshot_iters
