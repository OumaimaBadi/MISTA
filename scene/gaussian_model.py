#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

import trimesh
import igl

# MIGS: Densification/pruning not used all dynamic point growing methods are disabled.

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, cfg):
        self.cfg = cfg

        # two modes: SH coefficient or feature
        self.use_sh = cfg.use_sh
        self.active_sh_degree = 0
        if self.use_sh:
            self.max_sh_degree = cfg.sh_degree
            self.feature_dim = (self.max_sh_degree + 1) ** 2
        else:
            self.feature_dim = cfg.feature_dim

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        # self.xyz_optimizer = None
        self._opacity = torch.empty(0)
        #self.max_radii2D = torch.empty(0)
        #self.xyz_gradient_accum = torch.empty(0)
        #self.denom = torch.empty(0)
        self.optimizer = None
        #self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._face_ids = None
        self._bary = None
        self._uv = None
        self.setup_functions()

    def clone(self):
        cloned = GaussianModel(self.cfg)

        properties = ["active_sh_degree",
                      "non_rigid_feature",
                      ]
        for property in properties:
            if hasattr(self, property):
                setattr(cloned, property, getattr(self, property))

        parameters = ["_xyz",
                      "_features_dc",
                      "_features_rest",
                      "_scaling",
                      "_rotation",
                      "_opacity"]
        for parameter in parameters:
            setattr(cloned, parameter, getattr(self, parameter) + 0.)

        return cloned

    def set_fwd_transform(self, T_fwd):
        self.fwd_transform = T_fwd

    def color_by_opacity(self):
        cloned = self.clone()
        cloned._features_dc = self.get_opacity.unsqueeze(-1).expand(-1,-1,3)
        cloned._features_rest = torch.zeros_like(cloned._features_rest)
        return cloned

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            #self.max_radii2D,
            #self.xyz_gradient_accum,
            #self.denom,
            #self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    # def restore(self, model_args, training_args):
    #     (self.active_sh_degree, 
    #     self._xyz, 
    #     self._features_dc, 
    #     self._features_rest,
    #     self._scaling, 
    #     self._rotation, 
    #     self._opacity,
    #     #self.max_radii2D, 
    #     #xyz_gradient_accum, 
    #     #denom,
    #     opt_dict, 
    #     self.spatial_lr_scale) = model_args
    #     self.training_setup(training_args)
    #     #self.xyz_gradient_accum = xyz_gradient_accum
    #     #self.denom = denom
    #     self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        if hasattr(self, 'rotation_precomp'):
            return self.covariance_activation(self.get_scaling, scaling_modifier, self.rotation_precomp)
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if not self.use_sh:
            return
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_opacity_loss(self):
        # opacity classification loss
        opacity = self.get_opacity
        eps = 1e-6
        loss_opacity_cls = -(opacity * torch.log(opacity + eps) + (1 - opacity) * torch.log(1 - opacity + eps)).mean()
        return {'opacity': loss_opacity_cls}

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale=1.):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        print("xyz min:", fused_point_cloud.min(dim=0).values)
        print("xyz max:", fused_point_cloud.max(dim=0).values)
        print("xyz mean:", fused_point_cloud.mean(dim=0))

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        if self.use_sh:
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
        else:
            features = torch.zeros((fused_color.shape[0], 1, self.feature_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._xyz = fused_point_cloud
        self._features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        self._features_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        self._scaling = scales
        self._rotation = rots
        # self._scaling = nn.Parameter(scales.requires_grad_(True))
        # self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        # keep mesh binding if provided by dataset ---
        if hasattr(pcd, "face_ids") and pcd.face_ids is not None:
            self._face_ids = torch.from_numpy(np.asarray(pcd.face_ids)).long().cuda()
            print(f"[GaussianModel] Face IDs loaded: {self._face_ids.shape}")
        else:
            self._face_ids = None
            print(f"[GaussianModel] No face IDs in point cloud")

        if hasattr(pcd, "bary_coords") and pcd.bary_coords is not None:
            self._bary = torch.from_numpy(np.asarray(pcd.bary_coords)).float().cuda()
            print(f"[GaussianModel] Barycentric coords loaded: {self._bary.shape}")
        else:
            self._bary = None
            print(f"[GaussianModel] No barycentric coords in point cloud")

        if hasattr(pcd, "uv") and pcd.uv is not None:
            self._uv = torch.from_numpy(np.asarray(pcd.uv)).float().cuda()
            print(f"[GaussianModel] UV coords loaded: {self._uv.shape}")
            print(f"[GaussianModel]  UV range: [{self._uv.min():.4f}, {self._uv.max():.4f}]")
        else:
            self._uv = None
            print(f"[GaussianModel] No UV coords in point cloud")


        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        self._scaling = torch.tensor(scale, dtype=torch.float, device="cuda")
        self._rotation = torch.tensor(rotation, dtype=torch.float, device="cuda")
        self.active_sh_degree = self.max_sh_degree



    def save_ply_playcanvas(self, path, rgb=None, fill_gray=0.5):
        """
        Export a strict 3DGS PLY compatible with PlayCanvas / Supersplat.

        Args:
            path (str): output file path (.ply)
            rgb (np.ndarray, optional): (N,3) array of RGB values in [0,1].
                If None, a constant gray color is used (fill_gray).
            fill_gray (float): fallback gray level if rgb=None.
            
        Notes:
            - f_dc_* : stores base RGB (direct color, no spherical harmonics).
            - f_rest_* : always zero (no higher-order SH or extra features).
            - Attribute order: 
            xyz, scales, rotation (quat), opacity, f_dc_0..2, f_rest_0..44
            - Opacity can remain in logit space (PlayCanvas will apply sigmoid).
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)


        xyz      = self._xyz.detach().cpu().numpy()      # (N,3)
        scale    = self._scaling.detach().cpu().numpy()  # (N,3)
        rotation = self._rotation.detach().cpu().numpy() # (N,4) quaternion
        opacity  = self._opacity.detach().cpu().numpy()  # (N,1) (logits are fine)

        N = xyz.shape[0]

        if rgb is None:
            f_dc = np.full((N, 3), float(fill_gray), dtype=np.float32)  # neutral gray
        else:
            f_dc = np.asarray(rgb, dtype=np.float32)
            assert f_dc.shape == (N, 3), "rgb must be shape (N,3)"
            f_dc = np.clip(f_dc, 0.0, 1.0)  # clamp to [0,1]


        f_rest45 = np.zeros((N, 45), dtype=np.float32)

        prop_names = [
            'x','y','z',
            'scale_0','scale_1','scale_2',
            'rot_0','rot_1','rot_2','rot_3',
            'opacity',
            'f_dc_0','f_dc_1','f_dc_2'
        ] + [f'f_rest_{i}' for i in range(45)]
        dtype_full = [(n, 'f4') for n in prop_names]

        attrib = np.concatenate([xyz, scale, rotation, opacity, f_dc, f_rest45], axis=1)

        elements = np.empty(N, dtype=dtype_full)
        elements[:] = list(map(tuple, attrib))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)

