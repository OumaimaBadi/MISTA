import torch
import torch.nn as nn
from tensorly.tt_tensor import tt_to_tensor

class TensorizedTTAdapter(nn.Module):
    """
    Adapter for applying MARS to TTUltraMIGSModule5D (global TT, no per-block).
    Uses **four masks only** (r1, r2, r3, r4) — no r5.
    """

    def __init__(self, tt_module):
        super().__init__()
        self.tt = tt_module  # instance of TTUltraMIGSModule5D

        # Extract ranks from the TT cores
        c0, c1, c2, c3, c4 = self.tt.get_tt_tensor()

        r1 = c0.shape[2]  # out of core0
        r2 = c1.shape[2]
        r3 = c2.shape[2]
        r4 = c3.shape[2]
        # c4.shape[2] == 1 → ignored, no mask for r5

        # Keep only the four internal TT ranks
        self.ranks = [r1, r2, r3, r4]

        self._pos_to_core_axes = {
            0: [("c0", 2), ("c1", 0)],  # r1
            1: [("c1", 2), ("c2", 0)],  # r2
            2: [("c2", 2), ("c3", 0)],  # r3
            3: [("c3", 2), ("c4", 0)],  # r4
        }

    @property
    def cores(self):
        """
        Return all TT core PARAMETERS for L2 reg in MARS.
        FIXED: Return the actual Parameters, not reconstructed tensors.
        """
        # Return the 4 main cores + the 6 semantic slices of core4
        return [
            self.tt.tt_tensor_gpu[0],  # core0
            self.tt.tt_tensor_gpu[1],  # core1
            self.tt.tt_tensor_gpu[2],  # core2
            self.tt.tt_tensor_gpu[3],  # core3
            # Core4 slices (the actual Parameters)
            self.tt.core4_xyz,
            self.tt.core4_scaling,
            self.tt.core4_rotation,
            self.tt.core4_dc,
            self.tt.core4_rest,
            self.tt.core4_opacity,
        ]

    def _apply_mask_to_core(self, core, axis, mask):
        """Soft mask along an axis."""
        shape = [1] * core.dim()
        shape[axis] = mask.shape[0]
        return core * mask.view(*shape)

    def _index_axis(self, x, axis, mask_bool):
        """Hard mask (boolean indexing)."""
        index = [slice(None)] * x.dim()
        index[axis] = mask_bool
        return x[tuple(index)]

    def _masked_cores(self, cores, masks):
        """
        Apply masks to all TT cores.
        masks: [mask_r1, mask_r2, mask_r3, mask_r4]
        """
        c0, c1, c2, c3, c4 = cores
        hard = not self.training 

        for pos, pairs in self._pos_to_core_axes.items():
            m = masks[pos]
            if not hard:
                m = m.clamp(1e-6, 1.0).sqrt()
            for tag, axis in pairs:
                if tag == "c0":
                    c0 = self._index_axis(c0, axis, m) if hard else self._apply_mask_to_core(c0, axis, m)
                elif tag == "c1":
                    c1 = self._index_axis(c1, axis, m) if hard else self._apply_mask_to_core(c1, axis, m)
                elif tag == "c2":
                    c2 = self._index_axis(c2, axis, m) if hard else self._apply_mask_to_core(c2, axis, m)
                elif tag == "c3":
                    c3 = self._index_axis(c3, axis, m) if hard else self._apply_mask_to_core(c3, axis, m)
                elif tag == "c4":
                    c4 = self._index_axis(c4, axis, m) if hard else self._apply_mask_to_core(c4, axis, m)

        return [c0, c1, c2, c3, c4]

    def reconstruct_with_masks(self, masks, idx_identity=0, original_order=False):
        """Reconstruct (G, M) tensor with masks applied."""
        raw_cores = self.tt.get_tt_tensor(idx_identity)
        masked = self._masked_cores(raw_cores, masks)

        # Contract masked cores to dense tensor
        T = tt_to_tensor(masked)  # (1, n1, n2, n3, M)
        M = T.shape[-1]
        W_perm = T.squeeze(0).contiguous().view(-1, M)  # (G, M)

        if original_order and hasattr(self.tt, "inv_perm"):
            return W_perm[self.tt.inv_perm.to(W_perm.device)]
        return W_perm

    def forward(self, x=None, masks=None, idx_identity=0, original_order=False):
        if masks is None:
            return self.tt.get_W_for_identity(idx_identity, original_order=original_order)
        return self.reconstruct_with_masks(masks, idx_identity=idx_identity, original_order=original_order)

    @torch.no_grad()
    def export_pruned(self, mars):
        masks = [(torch.sigmoid(logits) > 0.5) for logits in mars.phi_logits_list]  # bool
        raw_cores = [c.clone() for c in self.tt.get_tt_tensor()]
        pruned = self._masked_cores(raw_cores, masks)
        return pruned

    def add_identity(self, *args, **kwargs):
        """Propagate add_identity to underlying TT module."""
        return self.tt.add_identity(*args, **kwargs)

    def expand_first_core(self, n_identities: int):
        """Propagate expand_first_core to underlying TT module."""
        return self.tt.expand_first_core(n_identities)
