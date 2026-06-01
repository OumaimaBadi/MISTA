import torch.nn as nn
from .tt_migs_module_4d_uv_app import TTAppearanceUVModule
from .tt_migs_module_4d_uv_geo import TTGeometryUVModule


class TTDisentangledUVModule(nn.Module):
    """
    Final disentangled module:
      - appearance: dc + rest
      - geometry: scaling + rotation
      - explicit outside: xyz + opacity
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.appearance = TTAppearanceUVModule(cfg)
        self.geometry = TTGeometryUVModule(cfg)

    def init_from_tensor(self, gaussian_model):
        print("\n" + "="*80)
        print("[TT-DIS] Initializing disentangled UV modules")
        print("="*80)

        self.appearance.init_from_tensor(gaussian_model)
        self.geometry.init_from_tensor(gaussian_model)

        print("[TT-DIS] Appearance + Geometry init done")
        print("="*80 + "\n")

    def expand_first_core(self, n_identities: int):
        self.appearance.expand_first_core(n_identities)
        self.geometry.expand_first_core(n_identities)

    def add_identity(self, noise_scale: float = 0.05, rebuild_optimizer: bool = True):
        new_id_a = self.appearance.add_identity(noise_scale, rebuild_optimizer=False)
        new_id_g = self.geometry.add_identity(noise_scale, rebuild_optimizer=False)
        assert new_id_a == new_id_g
        return new_id_a

    def set_optimizer(self, opt_cfg):
        self.appearance.set_optimizer(opt_cfg)
        self.geometry.set_optimizer(opt_cfg)

    def step(self, iteration=None):
        self.appearance.step(iteration)
        self.geometry.step(iteration)

    def get_app_for_identity(self, identity_idx: int, uv_query=None):
        return self.appearance.get_W_for_identity(identity_idx, uv_query)

    def get_geo_for_identity(self, identity_idx: int, uv_query=None):
        return self.geometry.get_W_for_identity(identity_idx, uv_query)