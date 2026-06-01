import os
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

def to_dict_matrix(mat):
    """Convertit une matrice numpy (2D) en dict imbriqué style ZJU."""
    out = {}
    for i in range(mat.shape[0]):
        out[str(i)] = {}
        for j in range(mat.shape[1]):
            out[str(i)][str(j)] = float(mat[i, j])
    return out

def to_dict_vector(vec):
    """Convertit un vecteur numpy (1D) en dict style ZJU."""
    out = {}
    for i in range(len(vec)):
        out[str(i)] = float(vec[i])
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-setting", type=str, required=True,
                        help="Fichier settingX.json dans AIST++/Annotations/cameras/")
    parser.add_argument("--out", type=str, required=True,
                        help="Fichier de sortie cam_params.json")
    args = parser.parse_args()

    # Charger fichier settingX.json
    with open(args.camera_setting, "r") as f:
        cams = json.load(f)

    all_cam_params = {}

    for cam in cams:
        name = cam["name"]  # ex: "c01"
        size = cam["size"]
        K = np.array(cam["matrix"])
        D = np.array(cam["distortions"])
        rot = np.array(cam["rotation"])
        trans = np.array(cam["translation"])

        # Rotation → matrice 3x3
        Rmat = R.from_rotvec(rot).as_matrix()

        cam_params = {
            "K": to_dict_matrix(K),
            "D": {str(i): float(D[i]) for i in range(len(D))},
            "R": to_dict_matrix(Rmat),
            "T": to_dict_vector(trans),
            "S": to_dict_vector(size)  # On garde aussi la taille image
        }

        all_cam_params[name] = cam_params

    # Sauvegarde cam_params.json
    with open(args.out, "w") as f:
        json.dump(all_cam_params, f, indent=2)

    print(f"✅ Caméras converties et sauvegardées dans {args.out}")
