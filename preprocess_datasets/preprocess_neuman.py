"""
preprocess_neuman.py
====================
Converts a NeuMan sequence into a clean ZJU-like format with:
- one logical camera
- per-frame extrinsics
- one image/mask folder containing all frames
- SMPL files in models/

Output structure
----------------
out_dir/
├── cam_params.json
├── 1/
│   ├── 000000.jpg
│   ├── 000000.png
│   ├── 000001.jpg
│   ├── 000001.png
│   └── ...
└── models/
    ├── 000000.npz
    ├── 000001.npz
    └── ...

cam_params.json keys
--------------------
{
  "all_cam_names": ["1"],
  "1": {
    "K": ...,
    "D": ...,
    "S": ...,
    "frames": {
      "000000": {"R": ..., "T": ...},
      "000001": {"R": ..., "T": ...},
      ...
    }
  }
}

Note
----
This format requires a small loader change so that R/T are read from
cam_params["1"]["frames"][frame_key].

models/*.npz keys
-----------------
  minimal_shape   (6890, 3)  SMPL A-pose verts in normalized SMPL-scale world
  betas           (10,)
  Jtr_posed       (24, 3)    posed joints in normalized world units
  bone_transforms (24, 4, 4) LBS transforms in normalized world units (no global trans)
  trans           (3,)       global translation normalized by alignment scale s
  root_orient     (3,)       axis-angle root orientation in aligned world frame
  pose_body       (63,)
  pose_hand       (6,)

Alignment convention (alignments.npy)
--------------------------------------
  A = alignments["NNNNN.png"]   shape (4, 3)
  A[:3]  →  sR  (3x3 scaled rotation)
  A[3]   →  t   (3,) translation in COLMAP world coordinates

  Row-vector convention:
      v_world_colmap = v_smpl @ A[:3] + A[3]

  We decompose A[:3] into:
      A[:3] = s * R_align

Normalized convention used in this script
-----------------------------------------
To make all NeuMan sequences share a consistent canonical body scale:
- we DO NOT bake s into vertices, joints, or bone_transforms
- we DO bake R_align into root_orient
- we store trans as t / s
- we store camera translations as colmap_t / s

This keeps all sequences in a comparable SMPL-scale coordinate system,
which is much safer for shared training across multiple sequences.

Usage
-----
  python preprocess_neuman.py \
      --neuman_seq  /data/neuman/seattle \
      --out_dir     /data/neuman_preprocessed/seattle \
      --bm_path     body_models/smpl/neutral/model.pkl \
      --faces_npz   body_models/misc/faces.npz \
      [--export_ply]
"""

import os
import json
import shutil
import argparse

import numpy as np
import joblib
import torch
import trimesh

from scipy.spatial.transform import Rotation
from human_body_prior.body_model.body_model import BodyModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# COLMAP parsers
# ---------------------------------------------------------------------------

def parse_cameras_txt(path):
    """
    Returns dict camera_id -> {W, H, fx, fy, cx, cy}
    Handles PINHOLE and SIMPLE_PINHOLE.
    """
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            W, H = int(parts[2]), int(parts[3])

            if model == 'PINHOLE':
                fx, fy, cx, cy = (
                    float(parts[4]), float(parts[5]),
                    float(parts[6]), float(parts[7])
                )
            elif model == 'SIMPLE_PINHOLE':
                f_val = float(parts[4])
                fx = fy = f_val
                cx, cy = float(parts[5]), float(parts[6])
            else:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")

            cameras[cam_id] = dict(W=W, H=H, fx=fx, fy=fy, cx=cx, cy=cy)

    return cameras


def parse_images_txt(path):
    """
    Returns list of frame dicts sorted by IMAGE_ID.

    Each dict:
        image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name

    COLMAP convention:
        x_cam = R @ x_world + t
    where R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    """
    frames = []
    with open(path) as f:
        lines = [l.strip() for l in f
                 if l.strip() and not l.strip().startswith('#')]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id = int(parts[0])
        qw, qx, qy, qz = (
            float(parts[1]), float(parts[2]),
            float(parts[3]), float(parts[4])
        )
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id = int(parts[8])
        name = parts[9]

        frames.append(dict(
            image_id=image_id,
            qw=qw, qx=qx, qy=qy, qz=qz,
            tx=tx, ty=ty, tz=tz,
            cam_id=cam_id,
            name=name,
        ))
        i += 2  # skip POINTS2D line

    frames.sort(key=lambda x: x['image_id'])
    return frames


# ---------------------------------------------------------------------------
# Alignment decomposition
# ---------------------------------------------------------------------------

def decompose_alignment(A):
    """
    Decompose alignment matrix A (4, 3) into scale, rotation, translation.

    Convention:
        v_world = v_smpl @ A[:3] + A[3]

    Parameters
    ----------
    A : np.ndarray (4, 3)

    Returns
    -------
    s       : float
    R_align : (3, 3)
    t       : (3,)
    """
    sR = A[:3].astype(np.float64)
    t = A[3].astype(np.float32)

    U, S, Vt = np.linalg.svd(sR)
    s = float(np.mean(S))
    R_align = (U @ Vt).astype(np.float32)

    if np.linalg.det(R_align) < 0:
        U[:, -1] *= -1
        R_align = (U @ Vt).astype(np.float32)

    return s, R_align, t


# ---------------------------------------------------------------------------
# SMPL per-frame processing
# ---------------------------------------------------------------------------

def load_body_model(bm_path, faces_npz, device):
    bm = BodyModel(bm_path=bm_path, num_betas=10, batch_size=1).to(device)
    faces = np.load(faces_npz)['faces']
    return bm, faces


def smpl_for_frame(
    bm,
    device,
    pose72,
    betas10,
    s,
    R_align,
    t,
    faces=None,
    export_ply=False,
    ply_path=None,
):
    """
    Run SMPL body model and return arrays ready to be saved in the ZJU npz.

    Normalized convention:
    - no scale baking into vertices / joints / bone transforms
    - only rotation alignment is baked into root_orient
    - global translation is stored as t / s
    """
    pose72 = np.asarray(pose72, dtype=np.float32).reshape(72)
    betas10 = np.asarray(betas10, dtype=np.float32).reshape(10)

    root_orient_smpl = pose72[:3].copy()
    pose_body = pose72[3:66].copy()
    pose_hand = pose72[66:].copy()

    # Compose world root orientation: R_world = R_align @ R_root
    R_root = Rotation.from_rotvec(root_orient_smpl).as_matrix()
    R_world = R_align @ R_root
    root_orient_world = Rotation.from_matrix(R_world).as_rotvec().astype(np.float32)

    def _t(arr):
        return torch.from_numpy(arr.reshape(1, -1)).float().to(device)

    betas_t = _t(betas10)
    ro_t = _t(root_orient_world)
    pb_t = _t(pose_body)
    ph_t = _t(pose_hand)

    # Minimal shape in canonical SMPL scale
    body_min = bm(betas=betas_t)
    minimal_shape = body_min.v[0].detach().cpu().numpy()

    # Posed body, still in canonical SMPL scale
    body = bm(
        root_orient=ro_t,
        pose_body=pb_t,
        pose_hand=ph_t,
        betas=betas_t,
    )

    vertices = body.v[0].detach().cpu().numpy()
    Jtr_posed = body.Jtr[0].detach().cpu().numpy()

    if not hasattr(body, "bone_transforms"):
        raise AttributeError(
            "BodyModel output does not contain 'bone_transforms'. "
            "Check your human_body_prior version."
        )
    bone_transforms = body.bone_transforms[0].detach().cpu().numpy().copy()

    # Normalized global translation
    trans = (t / s).astype(np.float32)

    if export_ply and faces is not None and ply_path is not None:
        verts_vis = vertices + trans[None, :]
        mesh = trimesh.Trimesh(vertices=verts_vis, faces=faces)
        mesh.export(ply_path)

    return dict(
        minimal_shape=minimal_shape.astype(np.float32),
        betas=betas10.astype(np.float32),
        Jtr_posed=Jtr_posed.astype(np.float32),
        bone_transforms=bone_transforms.astype(np.float32),
        trans=trans.astype(np.float32),
        root_orient=root_orient_world.astype(np.float32),
        pose_body=pose_body.astype(np.float32),
        pose_hand=pose_hand.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Preprocess one NeuMan sequence into a clean ZJU-like format."
    )
    ap.add_argument(
        "--neuman_seq",
        required=True,
        help="Root of ONE NeuMan sequence (e.g. .../neuman/seattle)",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for the preprocessed sequence",
    )
    ap.add_argument(
        "--bm_path",
        default="body_models/smpl/neutral/model.pkl",
        help="Path to SMPL neutral body model pkl",
    )
    ap.add_argument(
        "--faces_npz",
        default="body_models/misc/faces.npz",
        help="Path to faces.npz",
    )
    ap.add_argument(
        "--export_ply",
        action="store_true",
        help="Export one PLY mesh per frame for visual inspection",
    )
    args = ap.parse_args()

    seq_dir = args.neuman_seq
    out_dir = ensure_dir(args.out_dir)

    # ------------------------------------------------------------------ #
    # 1. LOAD COLMAP CAMERAS
    # ------------------------------------------------------------------ #
    sparse_dir = os.path.join(seq_dir, "sparse")
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    images_txt = os.path.join(sparse_dir, "images.txt")

    if not os.path.exists(cameras_txt):
        raise FileNotFoundError(f"cameras.txt not found in {sparse_dir}")
    if not os.path.exists(images_txt):
        raise FileNotFoundError(f"images.txt not found in {sparse_dir}")

    colmap_cams = parse_cameras_txt(cameras_txt)
    colmap_frames = parse_images_txt(images_txt)
    N = len(colmap_frames)
    print(f"[INFO] {N} frames in images.txt")

    assert len(colmap_cams) == 1, \
        f"Expected 1 COLMAP camera, found {len(colmap_cams)}"

    cam_info = list(colmap_cams.values())[0]
    W, H = cam_info['W'], cam_info['H']
    fx, fy, cx, cy = cam_info['fx'], cam_info['fy'], cam_info['cx'], cam_info['cy']

    K = [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]]
    D = [[0.0], [0.0], [0.0], [0.0], [0.0]]
    S = [[W], [H]]

    # ------------------------------------------------------------------ #
    # 2. LOAD ALIGNMENTS EARLY
    # ------------------------------------------------------------------ #
    align_path = os.path.join(seq_dir, "alignments.npy")
    if not os.path.exists(align_path):
        raise FileNotFoundError(f"Not found: {align_path}")

    alignments = np.load(align_path, allow_pickle=True).item()
    print(f"[INFO] alignments loaded for {len(alignments)} frames")

    # ------------------------------------------------------------------ #
    # 3. BUILD cam_params.json
    #    One logical camera, per-frame extrinsics normalized by alignment scale
    # ------------------------------------------------------------------ #
    cam_params = {
        "all_cam_names": ["1"],
        "1": {
            "K": K,
            "D": D,
            "S": S,
            "frames": {}
        }
    }

    cam_scales = []

    for idx, fr in enumerate(colmap_frames):
        frame_name = fr["name"]

        if frame_name in alignments:
            align_key = frame_name
        else:
            alt = f"{idx:05d}.png"
            if alt in alignments:
                align_key = alt
            else:
                raise KeyError(
                    f"No alignment found for frame '{frame_name}' "
                    f"(tried '{alt}' too)"
                )

        A = np.asarray(alignments[align_key], dtype=np.float32)
        s, _, _ = decompose_alignment(A)
        cam_scales.append(s)

        R_mat = Rotation.from_quat(
            [fr['qx'], fr['qy'], fr['qz'], fr['qw']]
        ).as_matrix()

        frame_key = f"{idx:06d}"
        cam_params["1"]["frames"][frame_key] = {
            "R": R_mat.tolist(),
            "T": [[fr['tx'] / s], [fr['ty'] / s], [fr['tz'] / s]]
        }

    cam_params_path = os.path.join(out_dir, "cam_params.json")
    with open(cam_params_path, "w") as f:
        json.dump(cam_params, f, indent=2)

    print(f"[OK] cam_params.json written (1 logical camera, {N} frame-wise extrinsics)")
    print(f"[INFO] camera normalization scale stats: "
          f"min={min(cam_scales):.4f}  max={max(cam_scales):.4f}  mean={np.mean(cam_scales):.4f}")

    # ------------------------------------------------------------------ #
    # 4. COPY IMAGES & MASKS
    # ------------------------------------------------------------------ #
    images_dir = os.path.join(seq_dir, "images")
    masks_dir = os.path.join(seq_dir, "segmentations")
    cam_out = ensure_dir(os.path.join(out_dir, "1"))

    for idx, fr in enumerate(colmap_frames):
        frame_name = fr['name']
        out_img = os.path.join(cam_out, f"{idx:06d}.jpg")
        out_mask = os.path.join(cam_out, f"{idx:06d}.png")

        src_jpg = os.path.join(images_dir, frame_name.replace(".png", ".jpg"))
        src_png = os.path.join(images_dir, frame_name)

        if os.path.exists(src_jpg):
            shutil.copy(src_jpg, out_img)
        elif os.path.exists(src_png):
            from PIL import Image as PILImage
            PILImage.open(src_png).convert("RGB").save(out_img, quality=95)
        else:
            print(f"[WARN] image not found for frame '{frame_name}'")

        src_mask = os.path.join(masks_dir, frame_name)
        if os.path.exists(src_mask):
            shutil.copy(src_mask, out_mask)
        else:
            src_mask_jpg = src_mask.replace(".png", ".jpg")
            if os.path.exists(src_mask_jpg):
                shutil.copy(src_mask_jpg, out_mask)
            else:
                print(f"[WARN] mask not found for frame '{frame_name}'")

    print("[OK] images + masks copied")

    # ------------------------------------------------------------------ #
    # 5. LOAD SMPL DATA
    # ------------------------------------------------------------------ #
    opt_pkl = os.path.join(seq_dir, "smpl_output_optimized.pkl")
    if not os.path.exists(opt_pkl):
        raise FileNotFoundError(f"Not found: {opt_pkl}")

    opt = joblib.load(opt_pkl)
    smpl_data = opt[1]
    poses_list = smpl_data['pose']
    betas_list = smpl_data['betas']

    if len(poses_list) != N:
        raise ValueError(
            f"optimized poses count ({len(poses_list)}) != frames ({N})"
        )

    mean_betas = np.mean(
        np.stack([np.asarray(b, dtype=np.float32) for b in betas_list], axis=0),
        axis=0
    ).astype(np.float32)
    print(f"[INFO] mean betas (first 3): {mean_betas[:3]}")

    # ------------------------------------------------------------------ #
    # 6. RUN SMPL PER FRAME -> models/*.npz
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] SMPL device: {device}")

    bm, faces = load_body_model(args.bm_path, args.faces_npz, device)
    models_dir = ensure_dir(os.path.join(out_dir, "models"))

    scales = []

    for idx, fr in enumerate(colmap_frames):
        frame_name = fr['name']

        if frame_name in alignments:
            align_key = frame_name
        else:
            alt = f"{idx:05d}.png"
            if alt in alignments:
                align_key = alt
            else:
                raise KeyError(
                    f"No alignment found for frame '{frame_name}' "
                    f"(tried '{alt}' too)"
                )

        A = np.asarray(alignments[align_key], dtype=np.float32)
        s, R_align, t = decompose_alignment(A)
        scales.append(s)

        pose72 = np.asarray(poses_list[idx], dtype=np.float32).reshape(72)

        ply_path = (
            os.path.join(models_dir, f"{idx:06d}.ply")
            if args.export_ply else None
        )

        out = smpl_for_frame(
            bm=bm,
            device=device,
            pose72=pose72,
            betas10=mean_betas,
            s=s,
            R_align=R_align,
            t=t,
            faces=faces,
            export_ply=args.export_ply,
            ply_path=ply_path,
        )

        np.savez(os.path.join(models_dir, f"{idx:06d}.npz"), **out)

        if idx % 20 == 0 or idx == N - 1:
            print(
                f"  [SMPL] {idx:04d}/{N}  "
                f"scale={s:.4f}  "
                f"trans_norm=[{out['trans'][0]:.3f}, {out['trans'][1]:.3f}, {out['trans'][2]:.3f}]"
            )

    print(
        f"\n[INFO] scale stats over sequence: "
        f"min={min(scales):.4f}  max={max(scales):.4f}  mean={np.mean(scales):.4f}"
    )

    print("\nNeuMan preprocessing done")
    print(f"   Output : {out_dir}")
    print(f"   cam_params.json  (1 logical camera, {N} frame-wise extrinsics)")
    print("   image/mask folder : 1/")
    print(f"   models/  ({N} .npz files)")


if __name__ == "__main__":
    main()