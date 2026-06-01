# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import os
from os import makedirs
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision
import wandb
from omegaconf import OmegaConf
from tqdm import trange

from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import Evaluator, PSEvaluator, fix_random
import torch.nn as nn
import hydra
from diagnose import diagnose

def get_camera_folder_name(view):
    for attr in ["camera_id", "cam_id", "view_id", "uid"]:
        try:
            val = view.data[attr]
            return f"camera_{val}"
        except (KeyError, AttributeError):
            continue
    # fallback: use image name
    name = getattr(view, "image_name", None)
    if name is not None:
        return f"camera_{name}"
    return "camera_unknown"

def predict(config) -> None:
    """
    Prediction/inference loop for CP, Tucker, TT, and MARS-wrapped TT.
    Supporte aussi tt5d_color_split (branches géo + couleur séparées).
    """
    with torch.set_grad_enabled(False):

        load_ckpt = config.get("load_ckpt", None)
        if load_ckpt is None:
            raise ValueError("Please provide load_ckpt.")

        print("[CHECKPOINT] Loading checkpoint...")
        tmp = torch.load(load_ckpt, map_location="cpu")
        sd  = tmp["migs_module_state_dict"]

        migs_type = tmp.get("migs_type", config.migs.type)
        print("[CHECKPOINT] Detected migs_type =", migs_type)
        config.migs.type = migs_type

        appearance_id = getattr(config, "appearance_identity", None)


        # CASE 1: CP / Tucker
        if migs_type in ("cp", "tucker"):
            print("\n[SCENE] Building CP/Tucker scene...")
            config.migs.skip_init_from_tensor = False

            gaussians = GaussianModel(config.model.gaussian)
            scene = Scene(config, gaussians, config.exp_dir)
            scene.appearance_identity = appearance_id
            scene.eval()

            print("\n[CHECKPOINT] Loading CP/Tucker checkpoint...")
            scene.load_checkpoint(load_ckpt)
            print(" CP/Tucker checkpoint loaded")

        # CASE 2: TT / MARS-wrapped TT  (tt5d, tt5d_color_split, ...)
        else:
            print("\n[TT] Preparing TT checkpoint...")

            has_mars_prefix = any(k.startswith("tensorized_model.tt.") for k in sd.keys())
            prefix = "tensorized_model.tt." if has_mars_prefix else ""
            if has_mars_prefix:
                print("[CHECKPOINT] Detected MARS-wrapped checkpoint")

            # rangs spatiaux communs aux deux variantes
            r1 = sd[f"{prefix}tt_tensor_gpu.0"].shape[-1]
            r2 = sd[f"{prefix}tt_tensor_gpu.1"].shape[-1]
            r3 = sd[f"{prefix}tt_tensor_gpu.2"].shape[-1]
            r4 = sd[f"{prefix}tt_tensor_gpu.3"].shape[-1]
            config.migs.rank = [1, r1, r2, r3, r4, 1]

            n_id = sd[f"{prefix}tt_tensor_gpu.0"].shape[1]
            n1   = sd[f"{prefix}tt_tensor_gpu.1"].shape[1]
            n2   = sd[f"{prefix}tt_tensor_gpu.2"].shape[1]
            n3   = sd[f"{prefix}tt_tensor_gpu.3"].shape[1]

            # lecture des shapes selon le type 
            if migs_type == "tt5d_color_split":
                # branche géo : core4_geo_*
                M_xyz = sd[f"{prefix}core4_geo_xyz"].shape[1]       # 3
                M_scl = sd[f"{prefix}core4_geo_scaling"].shape[1]   # 3
                M_rot = sd[f"{prefix}core4_geo_rotation"].shape[1]  # 4
                M_opa = sd[f"{prefix}core4_geo_opacity"].shape[1]   # 1
                M_geo = M_xyz + M_scl + M_rot + M_opa               # 11

                # branche couleur : compter les identités
                n_color_ids = sum(
                    1 for k in sd
                    if k.startswith(f"{prefix}tt_color_list.")
                    and k.endswith(".0")
                )
                print(f"[CHECKPOINT] tt5d_color_split : n_color_ids={n_color_ids}  M_geo={M_geo}")

                # shapes d'un core couleur (identité 0)
                cc0_shape = sd[f"{prefix}tt_color_list.0.0"].shape  # (1, n1, r1c)
                cc1_shape = sd[f"{prefix}tt_color_list.0.1"].shape  # (r1c, n2, r2c)
                cc2_shape = sd[f"{prefix}tt_color_list.0.2"].shape  # (r2c, n3, rMc)
                cc3_shape = sd[f"{prefix}tt_color_list.0.3"].shape  # (rMc, 32, 1)
                print(f"[CHECKPOINT] color core shapes : {cc0_shape} {cc1_shape} {cc2_shape} {cc3_shape}")

                # tt_shape = (n_id, n1, n2, n3, 43) pour rétrocompatibilité
                config.migs.tt_shape = [n_id, n1, n2, n3, 43]

            else:
                # variante standard tt5d / tt6d / tt4d ...
                try:
                    M_xyz = sd[f"{prefix}core4_xyz"].shape[1]
                    M_scl = sd[f"{prefix}core4_scaling"].shape[1]
                    M_rot = sd[f"{prefix}core4_rotation"].shape[1]
                    M_dc  = sd[f"{prefix}core4_dc"].shape[1]
                    M_rst = sd[f"{prefix}core4_rest"].shape[1]
                    M_opa = sd[f"{prefix}core4_opacity"].shape[1]
                    M = M_xyz + M_scl + M_rot + M_dc + M_rst + M_opa
                except KeyError:
                    M_xyz, M_scl, M_rot, M_dc, M_rst, M_opa = 3, 3, 4, 1, 31, 1
                    M = 43
                config.migs.tt_shape = [n_id, n1, n2, n3, M]

            config.migs.n_identities_ckpt = int(n_id)
            config.migs.use_mars          = False
            config.migs.skip_init_from_tensor = True

            print(f"[CHECKPOINT] rank     = {config.migs.rank}")
            print(f"[CHECKPOINT] tt_shape = {config.migs.tt_shape}")

            # strip MARS prefix si nécessaire
            if has_mars_prefix:
                sd = {
                    k[len(prefix):] if k.startswith(prefix) else k: v
                    for k, v in sd.items()
                }

            # construction de la scène 
            print("\n[SCENE] Building TT scene...")
            gaussians = GaussianModel(config.model.gaussian)
            scene = Scene(config, gaussians, config.exp_dir)
            scene.appearance_identity = appearance_id
            scene.eval()

            # allocation des cores 
            print("\n[TT] Allocating cores...")
            tt_module = scene.migs_module
            tt_module.tt_shape = tuple(config.migs.tt_shape)
            tt_module.tt_rank  = config.migs.rank

            device = "cuda"

            # cores spatiaux géo communs (0..3)
            tt_module.tt_tensor_gpu = nn.ParameterList([
                nn.Parameter(torch.zeros(1,  n_id, r1, device=device)),
                nn.Parameter(torch.zeros(r1, n1,   r2, device=device)),
                nn.Parameter(torch.zeros(r2, n2,   r3, device=device)),
                nn.Parameter(torch.zeros(r3, n3,   r4, device=device)),
            ])

            if migs_type == "tt5d_color_split":
                # dernier core géo (slices)
                tt_module.core4_geo_xyz      = nn.Parameter(torch.zeros(r4, M_xyz, 1, device=device))
                tt_module.core4_geo_scaling  = nn.Parameter(torch.zeros(r4, M_scl, 1, device=device))
                tt_module.core4_geo_rotation = nn.Parameter(torch.zeros(r4, M_rot, 1, device=device))
                tt_module.core4_geo_opacity  = nn.Parameter(torch.zeros(r4, M_opa, 1, device=device))

                # TT couleur : une ParameterList par identité 
                tt_module.tt_color_list = nn.ModuleList([
                    nn.ParameterList([
                        nn.Parameter(torch.zeros(cc0_shape, device=device)),
                        nn.Parameter(torch.zeros(cc1_shape, device=device)),
                        nn.Parameter(torch.zeros(cc2_shape, device=device)),
                        nn.Parameter(torch.zeros(cc3_shape, device=device)),
                    ])
                    for _ in range(n_color_ids)
                ])
                tt_module._n_color_ids = n_color_ids
                print(f"color_split cores allocated  (n_color_ids={n_color_ids})")

            else:
                # tandard tt5d : slices core4 
                tt_module.core4_xyz      = nn.Parameter(torch.zeros(r4, M_xyz, 1, device=device))
                tt_module.core4_scaling  = nn.Parameter(torch.zeros(r4, M_scl, 1, device=device))
                tt_module.core4_rotation = nn.Parameter(torch.zeros(r4, M_rot, 1, device=device))
                tt_module.core4_dc       = nn.Parameter(torch.zeros(r4, M_dc,  1, device=device))
                tt_module.core4_rest     = nn.Parameter(torch.zeros(r4, M_rst, 1, device=device))
                tt_module.core4_opacity  = nn.Parameter(torch.zeros(r4, M_opa, 1, device=device))

            # buffers perm / inv_perm
            G = n1 * n2 * n3
            tt_module.register_buffer("perm",     torch.arange(G, dtype=torch.long, device=device))
            tt_module.register_buffer("inv_perm", torch.arange(G, dtype=torch.long, device=device))
            print(f" TT cores allocated (G={G})")

            # chargement des poids 
            missing, unexpected = tt_module.load_state_dict(sd, strict=False)
            if missing:
                print(f"Missing keys   : {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            print(" TT weights loaded")

            print("\n[CONVERTER] Loading converter state...")
            scene.converter.load_state_dict(tmp["converter_state"])
            print("Converter loaded")


        # COMMON RENDERING LOOP
        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, "renders")
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end   = torch.cuda.Event(enable_timing=True)
        times_ms   = []

        print("\n[RENDERING] Starting predict rendering...")

        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]

            if appearance_id is not None:
                view.person_id = int(appearance_id)
                if idx == 0:
                    print(f"\n[INFO] Using identity {view.person_id} for all frames")

            iter_start.record()

            print("[DEBUG] view.person_id       =", view.person_id)
            print("[DEBUG] appearance_identity  =", appearance_id)
            print("[DEBUG] subject              =", getattr(scene.test_dataset, "subject", "multi"))

            render_pkg = render(
                view,
                config.opt.iterations,
                scene,
                config.pipeline,
                background,
                compute_loss=False,
                return_opacity=False,
            )

            iter_end.record()
            torch.cuda.synchronize()
            elapsed_ms = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            wandb.log({
                "test_images": [
                    wandb.Image(rendering[None], caption=f"render_{view.image_name}")
                ]
            })

            camera_name = get_camera_folder_name(view)
            camera_render_path = os.path.join(render_path, camera_name)
            makedirs(camera_render_path, exist_ok=True)

            torchvision.utils.save_image(
                rendering,
                os.path.join(camera_render_path, f"render_{view.image_name}.png")
            )

            times_ms.append(elapsed_ms)

        mean_ms = float(np.mean(times_ms[1:])) if len(times_ms) > 1 else float(np.mean(times_ms))
        wandb.log({"metrics/time": mean_ms})
        np.savez(os.path.join(config.exp_dir, config.suffix, "results.npz"), time=mean_ms)

        print("\n✅ PREDICT COMPLETE!")
        print(f"   Model type: {migs_type}")
        print(f"   Time:       {mean_ms:.2f} ms/frame")
        print(f"   Renders:    {render_path}")

def _log_nonrigid_mlp_activations(scene: Scene, writer_dir: str) -> None:
    """
    One-shot TensorBoard logging of the non-rigid MLP graph and layer histograms
    to help debug/inspect the deformation network.
    """
    from torch.utils.tensorboard import SummaryWriter  # local import to avoid global dependency

    writer = SummaryWriter(log_dir=writer_dir)

    # Grab MLP and normalized Gaussian coordinates
    non_rigid_mlp = scene.converter.deformer.non_rigid.mlp
    xyz = scene.gaussians.get_xyz
    aabb = scene.converter.deformer.non_rigid.aabb
    xyz_norm = aabb.normalize(xyz, sym=True)

    # Use one sample camera to build pose features
    sample_camera = scene.test_dataset[0]
    rots = sample_camera.rots
    Jtrs = sample_camera.Jtrs
    pose_feat = scene.converter.deformer.non_rigid.pose_encoder(rots, Jtrs)
    pose_feat = pose_feat.expand(xyz_norm.shape[0], -1)

    # Optionally append a per-frame latent code if available
    if hasattr(scene.converter.deformer.non_rigid, "latent"):
        latent_module = scene.converter.deformer.non_rigid.latent
        latent_dim = getattr(latent_module, "embedding_dim", latent_module.weight.shape[1])

        frame_dict = scene.converter.deformer.non_rigid.frame_dict
        frame_id = sample_camera.frame_id
        latent_idx = frame_dict.get(frame_id, len(frame_dict) - 1)
        latent_idx = torch.tensor([latent_idx], dtype=torch.long, device=pose_feat.device)
        latent_code = latent_module(latent_idx).expand(pose_feat.shape[0], -1)
        assert latent_code.shape[1] == latent_dim
        pose_feat = torch.cat([pose_feat, latent_code], dim=1)

    # Adjust conditioning width to match the MLP expected input
    first_layer = non_rigid_mlp.lin0
    expected_in = first_layer.in_features
    coords_emb = non_rigid_mlp.embed_fn(xyz_norm) if non_rigid_mlp.embed_fn else xyz_norm
    expected_cond = expected_in - coords_emb.shape[1]
    if pose_feat.shape[1] < expected_cond:
        diff = expected_cond - pose_feat.shape[1]
        print(f"[WARN] Padding pose_feat by {diff} zeros to match MLP expected conditioning width.")
        pad = torch.zeros((pose_feat.shape[0], diff), device=pose_feat.device)
        pose_feat = torch.cat([pose_feat, pad], dim=1)

    iteration = 0

    class WrappedMLP(torch.nn.Module):
        """Wrap MLP so that TensorBoard graph shows (coords, cond) explicitly."""
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp
        def forward(self, coords, cond):
            return self.mlp(coords, cond)

    with torch.no_grad():
        wrapped = WrappedMLP(non_rigid_mlp)
        writer.add_graph(wrapped, (xyz_norm, pose_feat))

        x = coords_emb
        for l in range(0, non_rigid_mlp.num_layers - 1):
            lin = getattr(non_rigid_mlp, f"lin{l}")

            if l in non_rigid_mlp.config.cond_in:
                x = torch.cat([x, pose_feat], dim=1)

            if l in non_rigid_mlp.config.skip_in:
                x = torch.cat([x, coords_emb], dim=1) / np.sqrt(2)

            x = lin(x)
            writer.add_histogram(f"non_rigid_mlp/layer_{l}_pre_act", x, iteration)

            if l < non_rigid_mlp.num_layers - 2:
                x = non_rigid_mlp.activation(x)
                writer.add_histogram(f"non_rigid_mlp/layer_{l}_post_act", x, iteration)

        output = x
        writer.add_histogram("non_rigid_mlp/output", output, iteration)
        writer.add_scalar("non_rigid_mlp/output_mean", output.mean(), iteration)
        writer.add_scalar("non_rigid_mlp/output_std", output.std(), iteration)

    writer.close()



def test(config) -> None:
    with torch.no_grad():
        load_ckpt = config.get("load_ckpt", None)
        if load_ckpt is None:
            raise ValueError("Please provide load_ckpt when using pruned checkpoint.")

        # Load checkpoint to read shapes and type
        print("[CHECKPOINT] Loading to extract shapes...")
        tmp = torch.load(load_ckpt, map_location="cpu")
        sd = tmp["migs_module_state_dict"]
        migs_type = tmp.get("migs_type", config.migs.type)
        print(f"[CHECKPOINT] Detected migs_type = {migs_type}")

        config.migs.skip_init_from_tensor = True

        # TT-specific shape extraction
        is_tt = migs_type not in ("cp", "tucker")

        if is_tt:
            has_mars_prefix = any(k.startswith("tensorized_model.tt.") for k in sd.keys())
            prefix = "tensorized_model.tt." if has_mars_prefix else ""
            if has_mars_prefix:
                print(f"[CHECKPOINT] Detected MARS-wrapped checkpoint")
            else:
                print(f"[CHECKPOINT] Detected pure TT checkpoint")

            try:
                r1 = sd[f"{prefix}tt_tensor_gpu.0"].shape[-1]
                r2 = sd[f"{prefix}tt_tensor_gpu.1"].shape[-1]
                r3 = sd[f"{prefix}tt_tensor_gpu.2"].shape[-1]
                r4 = sd[f"{prefix}tt_tensor_gpu.3"].shape[-1]
            except KeyError as e:
                raise KeyError(f"Missing TT core in checkpoint. Available keys: {list(sd.keys())[:10]}") from e

            config.migs.rank = [1, r1, r2, r3, r4, 1]
            n_id = sd[f"{prefix}tt_tensor_gpu.0"].shape[1]
            n1   = sd[f"{prefix}tt_tensor_gpu.1"].shape[1]
            n2   = sd[f"{prefix}tt_tensor_gpu.2"].shape[1]
            n3   = sd[f"{prefix}tt_tensor_gpu.3"].shape[1]

            try:
                M_xyz = sd[f"{prefix}core4_xyz"].shape[1]
                M_scl = sd[f"{prefix}core4_scaling"].shape[1]
                M_rot = sd[f"{prefix}core4_rotation"].shape[1]
                M_dc  = sd[f"{prefix}core4_dc"].shape[1]
                M_rst = sd[f"{prefix}core4_rest"].shape[1]
                M_opa = sd[f"{prefix}core4_opacity"].shape[1]
                M = M_xyz + M_scl + M_rot + M_dc + M_rst + M_opa
            except KeyError:
                M_xyz, M_scl, M_rot, M_dc, M_rst, M_opa = 3, 3, 4, 1, 31, 1
                M = 43

            config.migs.tt_shape = [n_id, n1, n2, n3, M]
            config.migs.n_identities_ckpt = int(n_id)
            config.migs.use_mars = False

            print(f"[CHECKPOINT] rank={config.migs.rank}")
            print(f"[CHECKPOINT] tt_shape={config.migs.tt_shape}")


        # Build Scene
        print("\n[SCENE] Building scene...")
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.appearance_identity = config.appearance_identity
        scene.eval()

        # Load weights
        if is_tt:
            print(f"\n[TT] Allocating cores from checkpoint shapes...")
            tt_module = scene.migs_module
            tt_module.tt_shape = tuple(config.migs.tt_shape)
            tt_module.tt_rank  = config.migs.rank

            device = "cuda"
            tt_module.tt_tensor_gpu = nn.ParameterList([
                nn.Parameter(torch.zeros(1,  n_id, r1, device=device)),
                nn.Parameter(torch.zeros(r1, n1,   r2, device=device)),
                nn.Parameter(torch.zeros(r2, n2,   r3, device=device)),
                nn.Parameter(torch.zeros(r3, n3,   r4, device=device)),
            ])
            tt_module.core4_xyz      = nn.Parameter(torch.zeros(r4, M_xyz, 1, device=device))
            tt_module.core4_scaling  = nn.Parameter(torch.zeros(r4, M_scl, 1, device=device))
            tt_module.core4_rotation = nn.Parameter(torch.zeros(r4, M_rot, 1, device=device))
            tt_module.core4_dc       = nn.Parameter(torch.zeros(r4, M_dc,  1, device=device))
            tt_module.core4_rest     = nn.Parameter(torch.zeros(r4, M_rst, 1, device=device))
            tt_module.core4_opacity  = nn.Parameter(torch.zeros(r4, M_opa, 1, device=device))

            G = n1 * n2 * n3
            tt_module.register_buffer("perm",     torch.arange(G, dtype=torch.long, device=device))
            tt_module.register_buffer("inv_perm", torch.arange(G, dtype=torch.long, device=device))
            print(f"TT cores allocated (G={G})")

            print(f"\n[CHECKPOINT] Loading TT weights...")
            if has_mars_prefix:
                sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

            missing, unexpected = tt_module.load_state_dict(sd, strict=False)
            if missing:
                print(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            print(f"TT weights loaded")

            print(f"\n[CONVERTER] Loading converter state...")
            scene.converter.load_state_dict(tmp["converter_state"])
            print(f"Converter loaded")

        else:
            # CP/Tucker: use normal checkpoint loading
            print(f"\n[CHECKPOINT] Loading CP/Tucker checkpoint...")
            scene.load_checkpoint(load_ckpt)
            print(f"Checkpoint loaded")

        # Rendering loop
        print(f"\n[RENDERING] Starting...")

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, "renders")
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end   = torch.cuda.Event(enable_timing=True)

        evaluator = PSEvaluator() if config.dataset.name == "people_snapshot" else Evaluator()

        psnrs:    List[torch.Tensor] = []
        ssims:    List[torch.Tensor] = []
        lpipss:   List[torch.Tensor] = []
        times_ms: List[float]        = []

        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]

            if getattr(config, "appearance_identity", None) is not None:
                view.person_id = config.appearance_identity
                if idx == 0:
                    print(f"\n[INFO] Using identity {view.person_id} for all frames")

            try:
                iter_start.record()
                render_pkg = render(
                    view,
                    config.opt.iterations,
                    scene,
                    config.pipeline,
                    background,
                    compute_loss=False,
                    return_opacity=False,
                )
                iter_end.record()
                torch.cuda.synchronize()

                elapsed_ms = iter_start.elapsed_time(iter_end)
                rendering  = render_pkg["render"]
                gt         = view.original_image[:3, :, :]

                wandb.log({
                    "test_images": [
                        wandb.Image(rendering[None], caption=f"render_{view.image_name}"),
                        wandb.Image(gt[None],        caption=f"gt_{view.image_name}"),
                    ]
                })

                camera_name        = get_camera_folder_name(view)
                camera_render_path = os.path.join(render_path, camera_name)
                makedirs(camera_render_path, exist_ok=True)
                torchvision.utils.save_image(
                    rendering,
                    os.path.join(camera_render_path, f"render_{view.image_name}.png")
                )

                camera_gt_path = os.path.join(config.exp_dir, config.suffix, "gt", camera_name)
                makedirs(camera_gt_path, exist_ok=True)
                torchvision.utils.save_image(
                    gt,
                    os.path.join(camera_gt_path, f"gt_{view.image_name}.png")
                )

                if config.evaluate:
                    m = evaluator(rendering, gt)
                    psnrs.append(m["psnr"])
                    ssims.append(m["ssim"])
                    lpipss.append(m["lpips"])
                else:
                    psnrs.append(torch.tensor([0.0], device=rendering.device))
                    ssims.append(torch.tensor([0.0], device=rendering.device))
                    lpipss.append(torch.tensor([0.0], device=rendering.device))

                times_ms.append(elapsed_ms)

            except RuntimeError as e:
                print(f"\n RENDER ERROR on frame {idx} ({view.image_name}):")
                print(f"   {type(e).__name__}: {e}")
                if idx == 0:
                    print("CRASH ON FIRST FRAME — Aborting.")
                    raise
                print("   Skipping frame.")
                continue

        if len(psnrs) == 0:
            print("\n No frames rendered successfully!")
            return

        psnr_mean  = torch.mean(torch.stack(psnrs))
        ssim_mean  = torch.mean(torch.stack(ssims))
        lpips_mean = torch.mean(torch.stack(lpipss))
        mean_ms    = float(np.mean(times_ms[1:])) if len(times_ms) > 1 else float(np.mean(times_ms))

        wandb.log({
            "metrics/psnr":  psnr_mean,
            "metrics/ssim":  ssim_mean,
            "metrics/lpips": lpips_mean,
            "metrics/time":  mean_ms,
        })
        np.savez(
            os.path.join(config.exp_dir, config.suffix, "results.npz"),
            psnr=psnr_mean.detach().cpu().numpy(),
            ssim=ssim_mean.detach().cpu().numpy(),
            lpips=lpips_mean.detach().cpu().numpy(),
            time=mean_ms,
        )

        print(f"\n RENDERING COMPLETE!")
        print(f"   PSNR:  {psnr_mean:.2f}")
        print(f"   SSIM:  {ssim_mean:.4f}")
        print(f"   LPIPS: {lpips_mean:.4f}")
        print(f"   Time:  {mean_ms:.2f} ms/frame")
        


@hydra.main(version_base=None, config_path="configs", config_name="config_5d")
def main(config) -> None:
    """
    Hydra entry point.
    - Prepares experiment directory, W&B run, and random seeds.
    - Dispatches to test() or predict() according to config.mode.
    """
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    # Experiment directory
    config.exp_dir = config.get("exp_dir") or os.path.join("./exp", config.exp_name)
    os.makedirs(config.exp_dir, exist_ok=True)

    # Human-readable suffix for this run (affects output dirs)
    if config.mode == "test":
        config.suffix = f"{config.mode}-{config.dataset.test_mode}"
    elif config.mode == "predict":
        predict_seq = config.dataset.predict_seq
        if config.dataset.name == "MultiPersonZJUMoCap":
            predict_dict = {
                0: "sameperson",
                1: "TransferMotion",
                2: "dance0",
                3: "dance1",
                4: "flipping",
                5: "canonical",
            }
        else:
            predict_dict = {
                0: "sameperson",
                1: "TransferMotion",
                2: "dance0",
                3: "dance1",
                4: "flipping",
                5: "canonical",
            }
        predict_mode = predict_dict[predict_seq]
        config.suffix = f"{config.mode}-{predict_mode}"
    elif config.mode == "diagnose":
        config.suffix = "diagnose"
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    if config.dataset.freeview:
        config.suffix = f"{config.suffix}-freeview"

    # Weights & Biases
    wandb_name = f"{config.exp_name}-{config.suffix}"
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project="tt5d_mars_motionTransfer_386",
        entity="badioumaima11-insa-rennes",
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method="fork"),
    )

    # Reproducibility
    fix_random(config.seed)

    # Dispatch
    if config.mode == "test":
        test(config)
    elif config.mode == "predict":
        predict(config)
    elif config.mode == "diagnose":  
        diagnose(config)
    else:
        raise ValueError

if __name__ == "__main__":
    main()
