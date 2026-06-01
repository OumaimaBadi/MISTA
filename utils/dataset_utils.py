import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scene.gaussian_model import BasicPointCloud
from plyfile import PlyData, PlyElement

def compute_barycentric_batch(points, face_ids, verts, faces, eps=1e-12):
    tri = faces[face_ids]          # (N,3)
    v0 = verts[tri[:, 0]]
    v1 = verts[tri[:, 1]]
    v2 = verts[tri[:, 2]]

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p  = points - v0

    d00 = np.einsum('ij,ij->i', v0v1, v0v1)
    d01 = np.einsum('ij,ij->i', v0v1, v0v2)
    d11 = np.einsum('ij,ij->i', v0v2, v0v2)
    d20 = np.einsum('ij,ij->i', v0p,  v0v1)
    d21 = np.einsum('ij,ij->i', v0p,  v0v2)

    denom = d00 * d11 - d01 * d01
    safe = np.abs(denom) > eps

    v = np.zeros_like(d00, dtype=np.float32)
    w = np.zeros_like(d00, dtype=np.float32)

    v[safe] = (d11[safe] * d20[safe] - d01[safe] * d21[safe]) / denom[safe]
    w[safe] = (d00[safe] * d21[safe] - d01[safe] * d20[safe]) / denom[safe]
    u = 1.0 - v - w

    bary = np.stack([u, v, w], axis=1).astype(np.float32)

    # petite stabilisation
    bary = np.clip(bary, -1e-4, 1.0 + 1e-4)
    s = bary.sum(axis=1, keepdims=True)
    bary = bary / np.maximum(s, eps)

    if np.any(~safe):
        bary[~safe] = np.array([1/3, 1/3, 1/3], dtype=np.float32)

    return bary

def compute_uv_from_face_bary(face_ids, bary, smpl_uv_npz_path):
    data = np.load(smpl_uv_npz_path)
    uv_coords = data["uv_coords"].astype(np.float32)  # (T,2)
    uv_faces  = data["uv_faces"].astype(np.int64)     # (F,3)

    tri_uv = uv_faces[face_ids]      # (N,3)
    uv0 = uv_coords[tri_uv[:, 0]]
    uv1 = uv_coords[tri_uv[:, 1]]
    uv2 = uv_coords[tri_uv[:, 2]]

    u = bary[:, 0:1]
    v = bary[:, 1:2]
    w = bary[:, 2:3]
    uv = u * uv0 + v * uv1 + w * uv2
    return np.clip(uv, 0.0, 1.0)

def _R(axis, deg):
    return Rotation.from_euler(axis, deg, degrees=True).as_matrix()
def get_soldier_bone_transforms(
    Jtr,
    leg_deg=1.5,          # moins d'espace entre les jambes: descends vers 1.0 puis 0.5 si besoin
    arm_down_deg=55.0,    # bras vers le bas (soldat)
    clav_gap_deg=4.0,     # petit espace bras/torse via clavicule
    down_axis="z",        # axe de "descente" (souvent z ou x selon ton repère)
    kintree_path="body_models/misc/kintree_table.npy",
):
    # parents SMPL
    kintree = np.load(kintree_path)        # shape (2,24)
    parents = kintree[0].astype(int)       # parents[j] = parent index, root=-1

    # rotations relatives (locales)
    rel_R = np.tile(np.eye(3), (24, 1, 1))

    # jambes: petit V
    rel_R[1] = _R("z", +leg_deg)   # left hip
    rel_R[2] = _R("z", -leg_deg)   # right hip

    # bras "soldat":
    # - clavicule: petit angle pour garder un gap
    rel_R[13] = _R("z", +clav_gap_deg)  # left collar
    rel_R[14] = _R("z", -clav_gap_deg)  # right collar

    # - épaule: on descend le bras
    #   signe opposé gauche/droite sinon ça part symétriquement de travers
    rel_R[16] = _R(down_axis, -arm_down_deg)  # left shoulder
    rel_R[17] = _R(down_axis, +arm_down_deg)  # right shoulder

    # coudes/poignets restent identité -> pas de rotation parasite des mains

    # FK: transforms globaux
    T = np.tile(np.eye(4), (24, 1, 1))
    for j in range(24):
        p = parents[j]
        if p == -1:
            T[j, :3, :3] = rel_R[j]
            T[j, :3, 3]  = Jtr[j]
        else:
            T[j, :3, :3] = T[p, :3, :3] @ rel_R[j]
            T[j, :3, 3]  = T[p, :3, :3] @ (Jtr[j] - Jtr[p]) + T[p, :3, 3]

    # correction SMPL: t <- t - R @ J
    for j in range(24):
        T[j, :3, 3] -= T[j, :3, :3] @ Jtr[j]

    return T


# add ZJUMoCAP dataloader
def get_02v_bone_transforms(Jtr,):
    rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()

    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    chain = [1, 4, 7, 10]
    rot = rot45p.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    chain = [2, 5, 8, 11]
    rot = rot45n.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

    return bone_transforms_02v

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

class AABB(torch.nn.Module):
    def __init__(self, coord_max, coord_min):
        super().__init__()
        self.register_buffer("coord_max", torch.from_numpy(coord_max).float())
        self.register_buffer("coord_min", torch.from_numpy(coord_min).float())

    def normalize(self, x, sym=False):
        x = (x - self.coord_min) / (self.coord_max - self.coord_min)
        if sym:
            x = 2 * x - 1.
        return x

    def unnormalize(self, x, sym=False):
        if sym:
            x = 0.5 * (x + 1)
        x = x * (self.coord_max - self.coord_min) + self.coord_min
        return x

    def clip(self, x):
        return x.clip(min=self.coord_min, max=self.coord_max)

    def volume_scale(self):
        return self.coord_max - self.coord_min

    def scale(self):
        return math.sqrt((self.volume_scale() ** 2).sum() / 3.)