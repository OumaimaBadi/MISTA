import os
import json
import argparse
import shutil
import glob
import numpy as np

from scipy.spatial.transform import Rotation as R
import torch
import trimesh

# === SMPL / HBP ===
from human_body_prior.body_model.body_model import BodyModel


def to_dict_matrix(mat):
    mat = np.asarray(mat)
    out = {}
    for i in range(mat.shape[0]):
        out[str(i)] = {}
        for j in range(mat.shape[1]):
            out[str(i)][str(j)] = float(mat[i, j])
    return out

def to_dict_vector(vec):
    vec = np.asarray(vec).reshape(-1)
    out = {}
    for i in range(len(vec)):
        out[str(i)] = float(vec[i])
    return out

def load_mapping(mapping_txt):
    """
    mapping.txt contient des couples: <seq_name> <env_name>
    ex: gBR_sBM_cAll_d04_mBR0_ch01  setting7_1
    """
    m = {}
    with open(mapping_txt, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            seq, env = line.strip().split()
            m[seq] = env
    return m

def camera_params_from_setting(setting_json_path, add_size=True, remap_names=True):
    """
    Convertit settingX.json (AIST++) vers dict style ZJU:
      { "1": {K,D,R,T,S}, "2": {...}, ..., "9": {...} }
    - rotation (rotvec) -> R 3x3
    - T et D convertis en listes de listes (comme ZJU)
    - S aussi en [[W],[H]]
    """
    with open(setting_json_path, 'r') as f:
        cams = json.load(f)

    out = {"all_cam_names": [str(i) for i in range(1, 10)]}
    for cam in cams:
        name = cam["name"]   # "c01" ... "c09"
        cam_id = str(int(name[1:])) if remap_names else name

        K = np.array(cam["matrix"], dtype=np.float64)
        D = np.array(cam["distortions"], dtype=np.float64).reshape(-1)
        rvec = np.array(cam["rotation"], dtype=np.float64).reshape(3)
        tvec = np.array(cam["translation"], dtype=np.float64).reshape(3)
        size = np.array(cam["size"], dtype=np.int32).reshape(2)  # [W, H]

        Rmat = R.from_rotvec(rvec).as_matrix()

        # On force D et T à être des listes de listes [[x],[y],[z]]
        D_list = [[float(v)] for v in D]
        T_list = [[float(v)/100.0] for v in tvec]

        cam_dict = {
            "K": K.tolist(),              # reste liste classique (3x3)
            "D": D_list,                  # [[d1],[d2],[d3],[d4],[d5]]
            "R": Rmat.tolist(),           # 3x3
            "T": T_list                   # [[x],[y],[z]]
        }
        if add_size:
            cam_dict["S"] = [[int(size[0])], [int(size[1])]]  # [[W],[H]]

        out[cam_id] = cam_dict

    return out


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def build_body_model(bm_path_neutral='body_models/smpl/neutral/model.pkl',
                     faces_npz='body_models/misc/faces.npz'):
    """
    Charge BodyModel (HBP) et faces SMPL.
    """
    device = torch.device('cuda')
    body_model = BodyModel(
        bm_path=bm_path_neutral,
        num_betas=10,
        batch_size=1
    ).to(device)
    faces = np.load(faces_npz)['faces']
    return body_model, faces, device


def smpl_per_frame_full(body_model, device,
                        pose72, trans3, betas10=None,
                        export_ply=False, faces=None, ply_path=None,
                        scale=1.0):
    pose72 = np.asarray(pose72, dtype=np.float32).reshape(72)
    root_orient = pose72[0:3].copy()
    pose_body   = pose72[3:66].copy()
    pose_hand   = pose72[66:].copy()

    trans = np.asarray(trans3, dtype=np.float32).reshape(3)
    trans_m = trans / 100.0 # cm → m

    betas = np.zeros((10,), dtype=np.float32) if betas10 is None else np.asarray(betas10, dtype=np.float32).reshape(10)
    
    # Torch tensors
    betas_t = torch.from_numpy(betas).to(device)[None, ...]
    ro_t = torch.from_numpy(root_orient).to(device)[None, ...]
    pb_t = torch.from_numpy(pose_body).to(device)[None, ...]
    ph_t = torch.from_numpy(pose_hand).to(device)[None, ...]
    tr_t = torch.from_numpy(trans_m).to(device)[None, ...]

    # minimal_shape (déjà scalé)
    body_min = body_model(betas=betas_t)
    minimal_shape = body_min.v[0].detach().cpu().numpy() * scale

    # Corps complet
    body = body_model(root_orient=ro_t, pose_body=pb_t, pose_hand=ph_t,
                      betas=betas_t, trans=tr_t)

    vertices = body.v[0].detach().cpu().numpy() * scale
    bone_transforms = body.bone_transforms[0].detach().cpu().numpy()
    Jtr_posed = body.Jtr[0].detach().cpu().numpy() * scale

    if export_ply and faces is not None and ply_path is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(ply_path)

    out = {
        "minimal_shape": minimal_shape.astype(np.float32),
        "betas": betas.astype(np.float32),
        "Jtr_posed": Jtr_posed.astype(np.float32),
        "bone_transforms": bone_transforms.astype(np.float32),
        "trans": trans_m.astype(np.float32),
        "root_orient": root_orient.astype(np.float32),
        "pose_body": pose_body.astype(np.float32),
        "pose_hand": pose_hand.astype(np.float32),
    }

    return out


def main():
    ap = argparse.ArgumentParser(description="Preprocess AIST++ (STRICT ZJU style)")
    ap.add_argument("--aist_root", required=True, help="Racine AIST++")
    ap.add_argument("--out_root", required=True, help="Racine de sortie (AIST_preprocessed)")
    ap.add_argument("--seq", required=True, help="Séquence AIST++ (ex: d01/gBR_sBM_cAll_d01_mBR0_ch01)")
    ap.add_argument("--export_ply", action="store_true", help="Exporter un PLY par frame (facultatif)")
    args = ap.parse_args()

    aist_root = args.aist_root
    out_root = args.out_root
    seq_rel = args.seq

    

    # --- 1) CAMERAS
    cams_dir = os.path.join(aist_root, "Annotations", "cameras")
    mapping_txt = os.path.join(cams_dir, "mapping.txt")
    mapping = load_mapping(mapping_txt)

    seq_name = os.path.basename(seq_rel)
    seq_name_full = seq_name 
    parts = seq_name.split('_')
    if parts[3].startswith('d') and parts[3][1:].isdigit():
        parts.pop(3)
    seq_name_out = '_'.join(parts)
    seq_dir_out = ensure_dir(os.path.join(out_root, os.path.dirname(seq_rel), seq_name_out))
    
    print(f"[INFO] Output seq dir: {seq_dir_out}")
    
    if seq_name not in mapping:
        raise FileNotFoundError(f"{seq_name} absent de mapping.txt")
    env_name = mapping[seq_name]
    setting_json = os.path.join(cams_dir, f"{env_name}.json")
    if not os.path.exists(setting_json):
        alt = os.path.join(cams_dir, f"{env_name.replace('_','')}.json")
        if os.path.exists(alt):
            setting_json = alt
        else:
            raise FileNotFoundError(f"Introuvable: {setting_json}")

    cam_params = camera_params_from_setting(setting_json, add_size=True, remap_names=True)
    with open(os.path.join(seq_dir_out, "cam_params.json"), "w") as f:
        json.dump(cam_params, f, indent=2)
    print("[OK] cam_params.json écrit")

    # --- 2) FRAMES & MASKS
    day = seq_rel.split('/')[0]
    parts = seq_name.split('_')
    assert parts[2] == 'cAll', "Le nom de séquence doit contenir cAll"
    seq_base_nocAll = parts.copy()
    seq_base_nocAll[2] = ''
    seq_base_nocAll = '_'.join([p for p in seq_base_nocAll if p != ''])

    for cam_id in range(1, 10):
        ensure_dir(os.path.join(seq_dir_out, str(cam_id)))
        cam_tag = f"c{cam_id:02d}"
        tokens = parts.copy()
        tokens[2] = cam_tag
        seq_cam_name = '_'.join(tokens)

        frames_src = os.path.join(aist_root, "frames", day, seq_cam_name)
        masks_src  = os.path.join(aist_root, "masks", day, seq_cam_name)
        dst_dir    = os.path.join(seq_dir_out, str(cam_id))

        if not os.path.isdir(frames_src):
            print(f"[WARN] frames absents: {frames_src}")
            continue

        jpgs = sorted(glob.glob(os.path.join(frames_src, "*.jpg")))
        for i, jpg in enumerate(jpgs):
            new_name = f"{i:06d}.jpg"
            shutil.copy(jpg, os.path.join(dst_dir, new_name))

            orig_mask = os.path.join(masks_src, os.path.basename(jpg).replace(".jpg", ".png"))
            if os.path.exists(orig_mask):
                new_mask_name = f"{i:06d}.png"
                shutil.copy(orig_mask, os.path.join(dst_dir, new_mask_name))

    print("[OK] frames + masks copiés")

    # --- 3) MOTIONS
    motions_pkl = os.path.join(aist_root, "Annotations", "motions", f"{seq_name}.pkl")
    if not os.path.exists(motions_pkl):
        raise FileNotFoundError(f"Motion pkl introuvable: {motions_pkl}")

    import pickle
    with open(motions_pkl, "rb") as f:
        motion = pickle.load(f)

    smpl_poses = motion["smpl_poses"]
    smpl_trans = motion["smpl_trans"]
    smpl_scaling = motion.get("smpl_scaling", np.ones((len(smpl_trans), 1), dtype=np.float32))

    if smpl_poses.ndim == 3 and smpl_poses.shape[1:] == (24, 3):
        smpl_poses = smpl_poses.reshape(smpl_poses.shape[0], 72)

    N = smpl_poses.shape[0]
    print(f"[INFO] N frames (motion): {N}")

    models_dir = ensure_dir(os.path.join(seq_dir_out, "models"))

    body_model, faces, device = build_body_model()
    scale = (smpl_scaling[0].item() / 100.0) if smpl_scaling is not None else 1.0

    for i in range(N):
        pose72 = smpl_poses[i]
        trans3 = smpl_trans[i]

        out = smpl_per_frame_full(
            body_model=body_model,
            device=device,
            pose72=pose72,
            trans3=trans3,
            betas10=None,
            export_ply=args.export_ply,
            faces=faces,
            ply_path=os.path.join(models_dir, f"{i:06d}.ply") if args.export_ply else None,
            scale=scale
        )
        np.savez(os.path.join(models_dir, f"{i:06d}.npz"), **out)

        if (i % 100) == 0:
            print(f"[SMPL] saved {i:06d}.npz")

    print(" DONE: strict ZJU-style preprocessing for", seq_rel)
    print("   cam_params.json + frames/masks + models/*.npz (complets)")


if __name__ == "__main__":
    main()
