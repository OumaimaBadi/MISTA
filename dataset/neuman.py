import os
import glob
import cv2
import numpy as np
import json
import torch
import trimesh

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

from utils.graphics_utils import focal2fov
from utils.dataset_utils import (
    get_02v_bone_transforms,
    fetchPly,
    storePly,
    AABB,
    compute_barycentric_batch,
    compute_uv_from_face_bary,
)
from scene.cameras import Camera


class NeuManDataset(Dataset):
    """
    NeuMan dataset loader (clean version)

    Expected preprocessed structure:
        root_dir/
            subject/
                cam_params.json
                1/
                    000000.jpg
                    000000.png
                    000001.jpg
                    000001.png
                    ...
                models/
                    000000.npz
                    000001.npz
                    ...

    cam_params.json format:
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
    """

    def __init__(self, cfg, split="train"):
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.root_dir = cfg.root_dir
        self.subject = cfg.subject

        self.train_frames = cfg.train_frames
        self.train_cams = cfg.train_views
        self.val_frames = cfg.val_frames
        self.val_cams = cfg.val_views

        self.white_bg = cfg.white_background
        self.H, self.W = cfg.orig_hw       # IMPORTANT: [H, W]
        self.h, self.w = cfg.img_hw        # resized [h, w]

        # NeuMan-specific control flags
        self.use_freeview = bool(cfg.get("freeview", False))
        self.test_camera_mode = cfg.get("test_camera_mode", "native")
        self.test_camera_frame = int(cfg.get("test_camera_frame", 0))
        self.test_camera_frames = list(cfg.get("test_camera_frames", []))
        self.test_num_frames = int(cfg.get("test_num_frames", 0))
        self.predict_camera_frame = int(cfg.get("predict_camera_frame", 0))

        # Kept for compatibility with the same code path as ZJU/AIST++
        self.refine = False

        self.faces = np.load("body_models/misc/faces.npz")["faces"]
        self.skinning_weights = dict(np.load("body_models/misc/skinning_weights_all.npz"))
        self.posedirs = dict(np.load("body_models/misc/posedirs_all.npz"))
        self.J_regressor = dict(np.load("body_models/misc/J_regressors.npz"))

        if split == "train":
            cam_names = self.train_cams
            frames = self.train_frames
        elif split == "val":
            cam_names = self.val_cams
            frames = self.val_frames
        elif split == "test":
            cam_names = self.cfg.test_views[self.cfg.test_mode]
            frames = self.cfg.test_frames[self.cfg.test_mode]
        elif split == "predict":
            cam_names = self.cfg.predict_views
            frames = self.cfg.predict_frames
        else:
            raise ValueError(f"Unknown split: {split}")

        subject_dir = os.path.join(self.root_dir, self.subject)

        with open(os.path.join(subject_dir, "cam_params.json"), "r") as f:
            self.camera_dict = json.load(f)

        if len(cam_names) == 0:
            cam_names = self.camera_dict["all_cam_names"]

        # NeuMan clean preprocessing exposes one logical camera
        if len(cam_names) != 1:
            raise ValueError(
                f"NeuManDataset expects exactly one logical camera, got cam_names={cam_names}"
            )

        self.cam_names = cam_names
        cam_name = cam_names[0]

        start_frame, end_frame, sampling_rate = frames

        # --------------------------------------------------------------
        # Models
        # --------------------------------------------------------------
        # For NeuMan, keep predict simple: use models/*.npz too.
        model_files = sorted(glob.glob(os.path.join(subject_dir, "models", "*.npz")))
        if len(model_files) == 0:
            raise FileNotFoundError(
                f"No model files found in {os.path.join(subject_dir, 'models')}"
            )

        self.model_files = model_files
        frames = list(range(len(model_files)))

        if end_frame == 0:
            end_frame = len(model_files)

        frame_slice = slice(start_frame, end_frame, sampling_rate)
        model_files = model_files[frame_slice]
        frames = frames[frame_slice]

        if split == "test" and self.test_num_frames > 0:
            model_files = model_files[:self.test_num_frames]
            frames = frames[:self.test_num_frames]

        self.frames = frames
        self.model_files_list = model_files

        # --------------------------------------------------------------
        # Optional freeview placeholder
        # --------------------------------------------------------------
        # Freeview with frame-wise extrinsics needs a NeuMan-specific helper.
        # For now, keep a clean explicit error rather than silent wrong behavior.
        if self.use_freeview:
            raise NotImplementedError(
                "freeview is not implemented yet for NeuManDataset with frame-wise extrinsics."
            )

        # --------------------------------------------------------------
        # Build samples
        # --------------------------------------------------------------
        self.data = []
        cam_dir = os.path.join(subject_dir, cam_name)

        img_files = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
        mask_files = sorted(glob.glob(os.path.join(cam_dir, "*.png")))

        if split != "predict":
            img_files = img_files[frame_slice]
            mask_files = mask_files[frame_slice]

            if split == "test" and self.test_num_frames > 0:
                img_files = img_files[:self.test_num_frames]
                mask_files = mask_files[:self.test_num_frames]

            if len(img_files) != len(model_files):
                raise ValueError(
                    f"Mismatch: {len(img_files)} images vs {len(model_files)} model files "
                    f"for subject {self.subject}"
                )
            if len(mask_files) != len(model_files):
                raise ValueError(
                    f"Mismatch: {len(mask_files)} masks vs {len(model_files)} model files "
                    f"for subject {self.subject}"
                )

        for d_idx, f_idx in enumerate(frames):
            model_file = model_files[d_idx]

            # Camera frame selection
            if split == "test":
                if self.test_camera_mode == "native":
                    camera_frame_idx = f_idx
                elif self.test_camera_mode == "fixed":
                    camera_frame_idx = self.test_camera_frame
                elif self.test_camera_mode == "list":
                    if len(self.test_camera_frames) == 0:
                        raise ValueError(
                            "test_camera_mode='list' but test_camera_frames is empty"
                        )
                    camera_frame_idx = int(
                        self.test_camera_frames[d_idx % len(self.test_camera_frames)]
                    )
                else:
                    raise ValueError(
                        f"Unknown test_camera_mode: {self.test_camera_mode}"
                    )
            elif split == "predict":
                camera_frame_idx = self.predict_camera_frame
            else:
                camera_frame_idx = f_idx

            # RGB/mask paths
            if split == "predict":
                img_file = os.path.join(subject_dir, "1", f"{camera_frame_idx:06d}.jpg")
                mask_file = os.path.join(subject_dir, "1", f"{camera_frame_idx:06d}.png")
            else:
                img_file = img_files[d_idx]
                mask_file = mask_files[d_idx]

            self.data.append({
                "cam_idx": 0,
                "cam_name": cam_name,
                "data_idx": d_idx,
                "frame_idx": f_idx,
                "camera_frame_idx": camera_frame_idx,
                "img_file": img_file,
                "mask_file": mask_file,
                "model_file": model_file,
            })

        self.get_metadata()

        self.preload = cfg.get("preload", True)
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def get_metadata(self):
        data_paths = self.model_files_list
        data_path = data_paths[0]

        cano_data = self.get_cano_smpl_verts(data_path)

        base_metadata = {
            "faces": self.faces,
            "posedirs": self.posedirs,
            "J_regressor": self.J_regressor,
            "cameras_extent": self.compute_neuman_cameras_extent(),
        }

        if self.split != "train":
            self.metadata = {}
            self.metadata.update(base_metadata)
            self.metadata.update(cano_data)
            return

        start, end, step = self.train_frames
        frames = list(range(len(data_paths)))
        if end == 0:
            end = len(frames)
        frame_slice = slice(start, end, step)
        frames = frames[frame_slice]

        frame_dict = {frame: i for i, frame in enumerate(frames)}

        self.metadata = {
            **base_metadata,
            "frame_dict": frame_dict,
        }
        self.metadata.update(cano_data)

        if self.cfg.train_smpl:
            self.metadata.update(self.get_smpl_data())

    def compute_neuman_cameras_extent(self):
        cam_name = self.cam_names[0]
        frame_keys = sorted(self.camera_dict[cam_name]["frames"].keys())

        cam_centers = []
        for fk in frame_keys:
            R = np.array(self.camera_dict[cam_name]["frames"][fk]["R"], dtype=np.float32)
            T = np.array(self.camera_dict[cam_name]["frames"][fk]["T"], dtype=np.float32).reshape(3, 1)

            # COLMAP convention: x_cam = R x_world + T
            # camera center in world: C = -R^T T
            C = -R.T @ T
            cam_centers.append(C[:, 0])

        cam_centers = np.stack(cam_centers, axis=0)
        center = cam_centers.mean(axis=0)
        dists = np.linalg.norm(cam_centers - center[None, :], axis=1)

        return float(dists.max())

    def get_cano_smpl_verts(self, data_path):
        """
        Compute star-posed SMPL body vertices.
        To get a consistent canonical space,
        we do not add pose blend shape.
        """
        model_dict = np.load(data_path)
        gender = "neutral"

        minimal_shape = model_dict["minimal_shape"]
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        J_regressor = self.J_regressor[gender]
        Jtr = np.dot(J_regressor, minimal_shape)

        skinning_weights = self.skinning_weights[gender]
        bone_transforms_02v = get_02v_bone_transforms(Jtr)

        T = np.matmul(
            skinning_weights,
            bone_transforms_02v.reshape([-1, 16])
        ).reshape([-1, 4, 4])

        vertices = np.matmul(
            T[:, :3, :3],
            minimal_shape[..., np.newaxis]
        ).squeeze(-1) + T[:, :3, -1]

        coord_max = np.max(vertices, axis=0)
        coord_min = np.min(vertices, axis=0)

        padding_ratio = np.array(self.cfg.padding, dtype=float)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max += padding
        coord_min -= padding

        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.faces)

        return {
            "gender": gender,
            "smpl_verts": vertices.astype(np.float32),
            "minimal_shape": minimal_shape,
            "Jtr": Jtr,
            "skinning_weights": skinning_weights.astype(np.float32),
            "bone_transforms_02v": bone_transforms_02v,
            "cano_mesh": cano_mesh,
            "coord_min": coord_min,
            "coord_max": coord_max,
            "aabb": AABB(coord_max, coord_min),
        }

    def get_smpl_data(self):
        if self.split != "train":
            return {}

        from collections import defaultdict
        smpl_data = defaultdict(list)

        for idx, (frame, model_file) in enumerate(zip(self.frames, self.model_files_list)):
            model_dict = np.load(model_file)

            if idx == 0:
                smpl_data["betas"] = model_dict["betas"].astype(np.float32)

            smpl_data["frames"].append(frame)
            smpl_data["root_orient"].append(model_dict["root_orient"].astype(np.float32))
            smpl_data["pose_body"].append(model_dict["pose_body"].astype(np.float32))
            smpl_data["pose_hand"].append(model_dict["pose_hand"].astype(np.float32))
            smpl_data["trans"].append(model_dict["trans"].astype(np.float32))

        return smpl_data

    # ------------------------------------------------------------------
    # Standard dataset methods
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def getitem(self, idx, data_dict=None):
        if data_dict is None:
            data_dict = self.data[idx]

        cam_name = data_dict["cam_name"]
        frame_idx = data_dict["frame_idx"]
        camera_frame_idx = data_dict["camera_frame_idx"]
        img_file = data_dict["img_file"]
        mask_file = data_dict["mask_file"]
        model_file = data_dict["model_file"]

        K = np.array(self.camera_dict[cam_name]["K"], dtype=np.float32).copy()
        dist = np.array(self.camera_dict[cam_name]["D"], dtype=np.float32).ravel()

        frame_key = f"{camera_frame_idx:06d}"
        if frame_key not in self.camera_dict[cam_name]["frames"]:
            raise KeyError(
                f"Frame key {frame_key} not found in cam_params.json for camera {cam_name}"
            )

        R = np.array(self.camera_dict[cam_name]["frames"][frame_key]["R"], dtype=np.float32)
        T = np.array(self.camera_dict[cam_name]["frames"][frame_key]["T"], dtype=np.float32)

        # Recenter principal point the same way as in ZJU loader
        M = np.eye(3, dtype=np.float32)
        M[0, 2] = (K[0, 2] - self.W / 2) / K[0, 0]
        M[1, 2] = (K[1, 2] - self.H / 2) / K[1, 1]
        K[0, 2] = self.W / 2
        K[1, 2] = self.H / 2

        R = M @ R
        T = M @ T

        R = np.transpose(R)
        T = T[:, 0]

        image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        image = cv2.undistort(image, K, dist, None)
        mask = cv2.undistort(mask, K, dist, None)

        lanczos = self.cfg.get("lanczos", False)
        interpolation = cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_LINEAR

        image = cv2.resize(image, (self.w, self.h), interpolation=interpolation)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        # NeuMan masks: black person / white background
        mask = mask < 128

        # Replace only the background
        image[~mask] = 255.0 if self.white_bg else 0.0

        image = image / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        K[0, :] *= self.w / self.W
        K[1, :] *= self.h / self.H

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        # --------------------------------------------------------------
        # Human params
        # --------------------------------------------------------------
        minimal_shape = self.metadata["minimal_shape"]

        # ---- Human params ----
        model_dict = np.load(model_file)
        trans = model_dict["trans"].astype(np.float32)
        bone_transforms = model_dict["bone_transforms"].astype(np.float32)

        root_orient = model_dict["root_orient"].astype(np.float32)
        pose_body   = model_dict["pose_body"].astype(np.float32)
        pose_hand   = model_dict["pose_hand"].astype(np.float32)


        # ---- FIN DEBUG ----

        pose = np.concatenate([root_orient, pose_body, pose_hand], axis=-1)
        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))
        pose_mat_full = pose.as_matrix()
        pose_mat  = pose_mat_full[1:, ...].copy()
        pose_rot  = np.concatenate(
            [np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0
        ).reshape([-1, 9])

        Jtr = self.metadata["Jtr"]
        center = np.mean(minimal_shape, axis=0)
        minimal_shape_centered = minimal_shape - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding  = (cano_max - cano_min) * 0.05

        Jtr_norm  = Jtr - center
        Jtr_norm  = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.0

        bone_transforms_02v = self.metadata["bone_transforms_02v"]
        bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms_02v)
        bone_transforms = bone_transforms.astype(np.float32)
        bone_transforms[:, :3, 3] += trans   # ← trans ajouté ici correctement

        if frame_idx in [0, 1, 2, 50, 100]:
            R_w2c = np.array(self.camera_dict[cam_name]["frames"][frame_key]["R"])
            T_w2c = np.array(self.camera_dict[cam_name]["frames"][frame_key]["T"]).reshape(3)
            cam_pos_world = -R_w2c.T @ T_w2c

            print(f"[DEBUG frame {frame_idx}]")
            print(f"  trans (world):            {trans}")
            print(f"  final bone_trans[0,:3,3]: {bone_transforms[0,:3,3]}")
            print(f"  camera position (world):  {cam_pos_world}")
            print(f"  dist cam->trans:          {np.linalg.norm(trans - cam_pos_world):.3f} m")

        return Camera(
            frame_id=frame_idx,
            cam_id=int(cam_name),
            K=K, R=R, T=T,
            FoVx=FovX, FoVy=FovY,
            image=image, mask=mask,
            gt_alpha_mask=None,
            image_name=f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
            data_device=self.cfg.data_device,
            rots=torch.from_numpy(pose_rot).float().unsqueeze(0),
            Jtrs=torch.from_numpy(Jtr_norm).float().unsqueeze(0),
            bone_transforms=torch.from_numpy(bone_transforms),
        )

    def __getitem__(self, idx):
        if self.preload:
            return self.cameras[idx]
        return self.getitem(idx)

    # ------------------------------------------------------------------
    # Point cloud init helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _with_numpy_seed(tmp_seed, fn, *args, **kwargs):
        state = np.random.get_state()
        try:
            np.random.seed(int(tmp_seed))
            return fn(*args, **kwargs)
        finally:
            np.random.set_state(state)

    # def readPointCloud(self):
    #     seed = int(getattr(self.cfg, "seed", 123))
    #     n_points = int(self.cfg.get("n_init_points", 50_000))

    #     if self.cfg.get("random_init", False):
    #         ply_path = os.path.join(self.root_dir, self.subject, "random_pc.ply")

    #         aabb = self.metadata["aabb"]
    #         coord_min = aabb.coord_min.unsqueeze(0).numpy()
    #         coord_max = aabb.coord_max.unsqueeze(0).numpy()

    #         if not os.path.exists(ply_path):
    #             def _make_random():
    #                 xyz_norm = np.random.rand(n_points, 3).astype(np.float32)
    #                 xyz = xyz_norm * coord_min + (1.0 - xyz_norm) * coord_max
    #                 rgb = np.ones_like(xyz) * 255
    #                 storePly(ply_path, xyz.astype(np.float32), rgb.astype(np.uint8))

    #             self._with_numpy_seed(seed, _make_random)

    #         return fetchPly(ply_path)

    #     ply_path = os.path.join(self.root_dir, self.subject, "star_smpl.ply")
    #     binding_path = os.path.join(self.root_dir, self.subject, "star_smpl_binding.npz")

    #     verts = self.metadata["smpl_verts"].astype(np.float32)
    #     faces = self.faces.astype(np.int64)
    #     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    #     if (not os.path.exists(ply_path)) or (not os.path.exists(binding_path)):
    #         def _make_cano():
    #             xyz, face_ids = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    #             xyz = xyz.astype(np.float32)
    #             face_ids = face_ids.astype(np.int64)

    #             bary_coords = compute_barycentric_batch(xyz, face_ids, verts, faces)
    #             out = {"face_ids": face_ids, "bary_coords": bary_coords}

    #             uv_npz = self.cfg.get("smpl_uv_npz", None)
    #             if uv_npz is not None and os.path.exists(uv_npz):
    #                 uv = compute_uv_from_face_bary(face_ids, bary_coords, uv_npz).astype(np.float32)
    #                 out["uv"] = uv

    #             rgb = np.ones_like(xyz) * 255
    #             storePly(ply_path, xyz, rgb.astype(np.uint8))
    #             np.savez_compressed(binding_path, **out)

    #         self._with_numpy_seed(seed, _make_cano)

    #     pcd = fetchPly(ply_path)
    #     binding = np.load(binding_path)

    #     pcd.face_ids = binding["face_ids"]
    #     pcd.bary_coords = binding["bary_coords"]
    #     if "uv" in binding.files:
    #         pcd.uv = binding["uv"]

    #     return pcd

    def readPointCloud(self,):
        if self.cfg.get('random_init', False):
            ply_path = os.path.join(self.root_dir, self.subject, 'random_pc.ply')

            aabb = self.metadata['aabb']
            coord_min = aabb.coord_min.unsqueeze(0).numpy()
            coord_max = aabb.coord_max.unsqueeze(0).numpy()
            n_points = 50_000

            # Only create once, deterministically, without touching global RNG state
            if not os.path.exists(ply_path):
                def _make_random():
                    xyz_norm = np.random.rand(n_points, 3)
                    xyz = xyz_norm * coord_min + (1. - xyz_norm) * coord_max
                    rgb = np.ones_like(xyz) * 255
                    storePly(ply_path, xyz, rgb)

                self._with_numpy_seed(getattr(self.cfg, "seed", 123), _make_random)

            pcd = fetchPly(ply_path)

        else:
            ply_path = os.path.join(self.root_dir, self.subject, 'star.ply')

            # Create canonical PLY once, deterministically
            if not os.path.exists(ply_path):
                verts = self.metadata['smpl_verts']
                faces = self.faces
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                n_points = 50_000

                def _make_cano():
                    # trimesh.sample() uses NumPy RNG internally
                    xyz = mesh.sample(n_points)
                    rgb = np.ones_like(xyz) * 255
                    storePly(ply_path, xyz, rgb)

                self._with_numpy_seed(getattr(self.cfg, "seed", 123), _make_cano)

            # Load the (now existing) PLY
            pcd = fetchPly(ply_path)

        return pcd