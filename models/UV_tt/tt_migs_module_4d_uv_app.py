import math
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
from tensorly.decomposition import tensor_train
from tensorly.tt_tensor import tt_to_tensor
from omegaconf import ListConfig

tl.set_backend("pytorch")


class TTAppearanceUVModule(nn.Module):
    """
    UV-TT for appearance only:
      W_app = [dc(1), rest(31)] => M_app = 32
    Tensor: (I, Nu, Nv, M_app)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._base_seed = int(getattr(cfg, "seed", 123))
        self.verbose = bool(cfg.migs.get("verbose", False))
        self.tt_rank = cfg.migs.get("rank_app", cfg.migs.get("rank", None))

        # Delay / freeze-unfreeze
        self.tt_delay = cfg.migs.get("delay", 1000)
        if self.tt_delay is None:
            self.tt_delay = cfg.model.gaussian.get("delay", 0)
        self._tt_unfrozen = False

        self.optimizer = None
        self.scheduler = None
        self._opt_cfg = None
        self._needs_opt_rebuild = False
        self._grid_cache = {}

        self.tt_tensor_gpu = nn.ParameterList()

        self.core4_dc = nn.Parameter(torch.zeros(1, 1, 1))
        self.core4_rest = nn.Parameter(torch.zeros(1, 31, 1))

    # ---------------- RNG helpers ----------------
    def _stream(self, tag: str, device) -> torch.Generator:
        h = int.from_bytes(hashlib.md5(tag.encode("utf8")).digest()[:8], "little")
        g = torch.Generator(device=device)
        g.manual_seed(self._base_seed ^ h)
        return g

    def _randn_like(self, ref, tag: str):
        g = self._stream(tag, ref.device)
        return torch.randn(ref.shape, device=ref.device, dtype=ref.dtype, generator=g)

    # ---------------- rank helpers ----------------
    def _to_tt_ranks_4d(self, rank):
        if rank is None:
            R = int(self.cfg.migs.get("rank_app", self.cfg.migs.get("init_rank", 64)))
            return [1, R, R, R, 1]
        if isinstance(rank, ListConfig):
            rank = list(rank)
        if isinstance(rank, (list, tuple)):
            rank = list(map(int, rank))
            assert len(rank) == 5, f"[TT-APP] rank list must have len=5, got {rank}"
            return rank
        R = int(rank)
        return [1, R, R, R, 1]

    @torch.no_grad()
    def _zero_pad_pair_preserve(self, left, right, add, dim_left, dim_right):
        if add <= 0:
            return left, right

        dev = left.device
        dl_shape = list(left.shape)
        dr_shape = list(right.shape)
        dl_shape[dim_left] = add
        dr_shape[dim_right] = add

        left_std = left.detach().std()
        right_std = right.detach().std()
        if (not torch.isfinite(left_std)) or left_std < 1e-8:
            left_std = left.detach().abs().mean()
        if (not torch.isfinite(right_std)) or right_std < 1e-8:
            right_std = right.detach().abs().mean()

        scale = 1e-2
        pad_left = scale * left_std * torch.randn(dl_shape, device=dev, dtype=left.dtype)
        pad_right = scale * right_std * torch.randn(dr_shape, device=dev, dtype=right.dtype)

        new_left = torch.cat([left, pad_left], dim=dim_left)
        new_right = torch.cat([right, pad_right], dim=dim_right)
        return new_left, new_right

    @torch.no_grad()
    def _expand_r1_by_replication(self, r1_target: int):
        c0 = self.tt_tensor_gpu[0]  # (1,I,r1)
        c1 = self.tt_tensor_gpu[1]  # (r1,Nu,r2)
        r1_cur = c0.shape[2]
        if r1_cur >= int(r1_target):
            return

        def _repeat_to(x, dim, target):
            cur = x.shape[dim]
            if cur == target:
                return x
            times = math.ceil(target / cur)
            reps = [1] * x.ndim
            reps[dim] = times
            x_rep = x.repeat(*reps)
            slices = [slice(None)] * x.ndim
            slices[dim] = slice(0, target)
            return x_rep[tuple(slices)]

        r1_target = int(r1_target)
        scale = r1_cur / float(r1_target)

        c0_new = _repeat_to(c0, dim=2, target=r1_target) * scale
        c1_new = _repeat_to(c1, dim=0, target=r1_target)

        self.tt_tensor_gpu[0] = nn.Parameter(c0_new)
        self.tt_tensor_gpu[1] = nn.Parameter(c1_new)
        self._needs_opt_rebuild = True
        self._grid_cache.clear()

    @torch.no_grad()
    def _expand_ranks_to_targets_preserve(self, rank_or_ranks):
        if isinstance(rank_or_ranks, ListConfig):
            rank_or_ranks = list(rank_or_ranks)

        if isinstance(rank_or_ranks, (list, tuple)):
            ranks_target = list(map(int, rank_or_ranks))
            assert len(ranks_target) == 5, f"[TT-APP] ranks_target must be len=5, got {ranks_target}"
        else:
            R = int(rank_or_ranks)
            ranks_target = [1, R, R, R, 1]

        c1 = self.tt_tensor_gpu[1]  # (r1,Nu,r2)
        c2 = self.tt_tensor_gpu[2]  # (r2,Nv,r3)

        # expand r2
        r2_cur = c1.shape[2]
        r2_tgt = int(ranks_target[2])
        if r2_tgt > r2_cur:
            add = r2_tgt - r2_cur
            new_c1, new_c2 = self._zero_pad_pair_preserve(c1, c2, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[1] = nn.Parameter(new_c1)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)

        c2 = self.tt_tensor_gpu[2]

        # expand r3
        r3_cur = c2.shape[2]
        r3_tgt = int(ranks_target[3])
        if r3_tgt > r3_cur:
            add = r3_tgt - r3_cur
            core4_full = self.recombine_core4()  # (r3,32,1)
            new_c2, new_core4 = self._zero_pad_pair_preserve(c2, core4_full, add, dim_left=2, dim_right=0)
            self.tt_tensor_gpu[2] = nn.Parameter(new_c2)

            r3n, M, r_last = new_core4.shape
            assert r_last == 1
            assert M == 32, f"[TT-APP] Expected M=32, got {M}"

            self.core4_dc = nn.Parameter(new_core4[:, 0:1, :])
            self.core4_rest = nn.Parameter(new_core4[:, 1:32, :])

            if len(self.tt_tensor_gpu) >= 4:
                self.tt_tensor_gpu[3] = nn.Parameter(new_core4.detach().clone())

        self.tt_rank = list(map(int, ranks_target))
        self._needs_opt_rebuild = True
        self._grid_cache.clear()

    # ---------------- TT core handling ----------------
    def recombine_core4(self):
        return torch.cat([self.core4_dc, self.core4_rest], dim=1)  # (r3, 32, 1)

    def get_core0(self, identity_idx: int):
        core0 = self.tt_tensor_gpu[0]
        assert 0 <= identity_idx < core0.shape[1], f"identity_idx out of range: {identity_idx}"
        return core0[:, identity_idx:identity_idx + 1, :]

    def get_tt_tensor(self, identity_idx: int):
        return [
            self.get_core0(identity_idx),
            self.tt_tensor_gpu[1],
            self.tt_tensor_gpu[2],
            self.recombine_core4(),
        ]

    def _reconstruct_grid_for_identity(self, identity_idx: int):
        return tt_to_tensor(self.get_tt_tensor(identity_idx))  # (1,Nu,Nv,32)

    def _debug_print_core_shapes(self, tag=""):
        if len(self.tt_tensor_gpu) < 3:
            print(f"[TT-APP]{tag} cores not initialized")
            return

        c0 = self.tt_tensor_gpu[0]
        c1 = self.tt_tensor_gpu[1]
        c2 = self.tt_tensor_gpu[2]
        c4 = self.recombine_core4()

        print(f"[TT-APP]{tag} target ranks = {self.tt_rank}")
        print(f"[TT-APP]{tag} core0: {tuple(c0.shape)}  # (1,I,r1)")
        print(f"[TT-APP]{tag} core1: {tuple(c1.shape)}  # (r1,Nu,r2)")
        print(f"[TT-APP]{tag} core2: {tuple(c2.shape)}  # (r2,Nv,r3)")
        print(f"[TT-APP]{tag} core4: {tuple(c4.shape)}  # (r3,M_app,1)")

    # ---------------- init ----------------
    def init_from_tensor(self, gaussian_model):
        with torch.no_grad():
            device = gaussian_model._xyz.device
            G = gaussian_model._xyz.shape[0]
            print(f"[TT-APP] Initializing from {G} Gaussians using UV grid")
            uv = gaussian_model._uv.detach().float()
            dc = gaussian_model._features_dc.squeeze(-1).detach()      # (G,1)
            rest = gaussian_model._features_rest.squeeze(-1).detach()  # (G,31)

            W = torch.cat([dc, rest], dim=1)  # (G,32)
            M = 32

            self.register_buffer("gaussian_uv", uv)

            if getattr(self.cfg.migs, "uv_resolution", None) is not None:
                res = self.cfg.migs.uv_resolution
                if isinstance(res, (list, tuple, ListConfig)):
                    Nu, Nv = map(int, list(res))
                else:
                    Nu = Nv = int(res)
            else:
                Nu = int(self.cfg.migs.get("uv_Nu", 256))
                Nv = int(self.cfg.migs.get("uv_Nv", 256))

            uv01 = uv.clamp(0.0, 1.0)
            u = uv01[:, 0] * (Nu - 1)
            v = uv01[:, 1] * (Nv - 1)

            iu0 = torch.floor(u).long().clamp(0, Nu - 2)
            iv0 = torch.floor(v).long().clamp(0, Nv - 2)
            iu1 = iu0 + 1
            iv1 = iv0 + 1

            fu = (u - iu0.float()).clamp(0, 1)
            fv = (v - iv0.float()).clamp(0, 1)

            w00 = (1.0 - fu) * (1.0 - fv)
            w01 = (1.0 - fu) * fv
            w10 = fu * (1.0 - fv)
            w11 = fu * fv

            V = Nu * Nv
            grid_flat = torch.zeros(V, M, device=device, dtype=W.dtype)
            counts_flat = torch.zeros(V, 1, device=device, dtype=torch.float32)

            def lin_uv(iu, iv):
                return (iv * Nu + iu).long()

            corners = [(iu0, iv0, w00), (iu0, iv1, w01), (iu1, iv0, w10), (iu1, iv1, w11)]
            for iu_c, iv_c, ww in corners:
                idx = lin_uv(iu_c, iv_c)
                ww = ww.view(-1, 1).to(torch.float32)
                grid_flat.index_add_(0, idx, W * ww.to(W.dtype))
                counts_flat.index_add_(0, idx, ww)

            grid = grid_flat.view(1, Nu, Nv, M)
            counts = counts_flat.view(1, Nu, Nv, 1)
            grid = grid / (counts.to(grid.dtype) + 1e-8)
            print(f"[TT-APP] Resolution: (Nu,Nv)=({Nu},{Nv})  cells={Nu*Nv}")
            ranks_target = self._to_tt_ranks_4d(self.tt_rank)
            self.tt_rank = ranks_target
            tt = tensor_train(grid, rank=ranks_target, verbose=self.verbose)
            self.tt_tensor_gpu = nn.ParameterList([nn.Parameter(c.to(device)) for c in tt.factors])

            core4 = self.tt_tensor_gpu[3]  # (r3,32,1)
            self.core4_dc = nn.Parameter(core4[:, 0:1, :].detach().clone())
            self.core4_rest = nn.Parameter(core4[:, 1:32, :].detach().clone())
            self._debug_print_core_shapes(tag=" after_tt_init") 
            rank_cfg = self.cfg.migs.get("rank_app", self.cfg.migs.get("init_rank", None))
            if rank_cfg is not None:
                ranks_target = self._to_tt_ranks_4d(rank_cfg)
                self._expand_r1_by_replication(ranks_target[1])   # r1
                self._expand_ranks_to_targets_preserve(ranks_target)
                self._debug_print_core_shapes(tag=" after_rank_expand")

            self.register_buffer("NuNv", torch.tensor([Nu, Nv], device=device, dtype=torch.int64))
            self._needs_opt_rebuild = True
            self._grid_cache.clear()
            print(f"[TT-APP] Init complete. ranks={self.tt_rank}  tensor=(I,Nu,Nv,M_app)=(1,{Nu},{Nv},32)")

    # ---------------- sampling ----------------
    def get_W_for_identity(self, identity_idx: int, uv_query=None):
        if uv_query is None:
            uv_query = self.gaussian_uv
        uv_query = uv_query.detach()

        if self.training:
            grid_full = self._reconstruct_grid_for_identity(identity_idx).squeeze(0)  # (Nu,Nv,32)
        else:
            if identity_idx not in self._grid_cache:
                self._grid_cache[identity_idx] = self._reconstruct_grid_for_identity(identity_idx).squeeze(0).detach()
            grid_full = self._grid_cache[identity_idx]

        inp = grid_full.permute(2, 1, 0).unsqueeze(0).contiguous()  # (1,32,Nv,Nu)

        uv01 = uv_query.clamp(0.0, 1.0)
        x = uv01[:, 0] * 2.0 - 1.0
        y = uv01[:, 1] * 2.0 - 1.0
        grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)

        sampled = F.grid_sample(
            inp, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )
        sampled = sampled.view(32, -1).permute(1, 0).contiguous()
        return sampled

    # ---------------- identity expansion ----------------
    @torch.no_grad()
    def expand_first_core(self, n_identities: int):
        if len(self.tt_tensor_gpu) == 0:
            raise RuntimeError("[TT-APP] TT cores must be initialized before expand_first_core.")

        core0 = self.tt_tensor_gpu[0]  # (1,I,r1)
        _, I, _ = core0.shape
        if I >= n_identities:
            return

        base = core0[:, 0:1, :].detach()
        rep = base.repeat(1, n_identities, 1)
        noise = self._randn_like(rep, tag="core0_expand_noise") * 1e-3
        self.tt_tensor_gpu[0] = nn.Parameter(rep + noise)

        self._grid_cache.clear()
        self._needs_opt_rebuild = True
        if self.optimizer is not None and self._opt_cfg is not None:
            self.set_optimizer(self._opt_cfg)

    @torch.no_grad()
    def add_identity(self, noise_scale: float = 0.05, rebuild_optimizer: bool = True) -> int:
        if len(self.tt_tensor_gpu) == 0:
            raise RuntimeError("[TT-APP] TT cores not initialized.")

        core0 = self.tt_tensor_gpu[0]  # (1,I,r1)
        _, I, _ = core0.shape

        U = core0.detach()
        mu = U.mean(dim=1, keepdim=True)
        sig = U.std(dim=1, unbiased=False, keepdim=True).clamp_(min=1e-8)
        eps = self._randn_like(mu, "add_identity").expand_as(mu)
        new_row = mu + noise_scale * sig * eps  # (1,1,r1)

        self.tt_tensor_gpu[0] = nn.Parameter(torch.cat([core0, new_row], dim=1))

        self._grid_cache.clear()
        self._needs_opt_rebuild = True
        if rebuild_optimizer and (self.optimizer is not None) and (self._opt_cfg is not None):
            self.set_optimizer(self._opt_cfg)

        return I

    # ---------------- optimizer / freeze ----------------
    def optimize_parameters(self):
        return [
            self.tt_tensor_gpu[0],
            self.tt_tensor_gpu[1],
            self.tt_tensor_gpu[2],
            self.core4_dc,
            self.core4_rest,
        ]

    def freeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = False

    def unfreeze_tt_parameters(self):
        for p in self.optimize_parameters():
            p.requires_grad = True

    def set_optimizer(self, opt_cfg):
        self._opt_cfg = dict(opt_cfg) if opt_cfg is not None else {}
        lr_init = float(self._opt_cfg.get("position_lr_init", 1.6e-4))
        lr_final = float(self._opt_cfg.get("position_lr_final", 1.6e-6))
        decay_iters = int(self._opt_cfg.get("position_lr_max_steps", 50000))

        param_groups = []
        decayed_idx = []

        for core in [self.tt_tensor_gpu[0], self.tt_tensor_gpu[1], self.tt_tensor_gpu[2]]:
            param_groups.append({
                "params": [core],
                "lr": lr_init,
                "initial_lr": lr_init,
                "final_lr": lr_final,
            })
            decayed_idx.append(len(param_groups) - 1)

        param_groups += [
            {"params": [self.core4_dc], "lr": float(self._opt_cfg.get("feature_lr", 2.5e-3))},
            {"params": [self.core4_rest], "lr": float(self._opt_cfg.get("feature_lr", 2.5e-3))},
        ]

        self.optimizer = torch.optim.Adam(param_groups)

        lr0 = max(lr_init, 1e-20)
        lrf = max(lr_final, 1e-20)
        gamma = (lrf / lr0) ** (1.0 / max(decay_iters, 1))

        def make_lambda(i):
            return (lambda step: gamma ** step) if i in decayed_idx else (lambda step: 1.0)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[make_lambda(i) for i in range(len(self.optimizer.param_groups))]
        )

        self._needs_opt_rebuild = False

    def step(self, iteration=None):
        if self.optimizer is None:
            return

        if (iteration is not None) and (iteration < self.tt_delay):
            self.freeze_tt_parameters()
            self.optimizer.zero_grad(set_to_none=True)
            return

        if (iteration is not None) and (not self._tt_unfrozen) and (iteration >= self.tt_delay):
            self.unfreeze_tt_parameters()
            self._tt_unfrozen = True
            if self.verbose:
                print(f"[TT-APP] Unfrozen at iter {iteration}")

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._grid_cache.clear()
        if self.scheduler is not None:
            self.scheduler.step()