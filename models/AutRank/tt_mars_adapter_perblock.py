import torch
import torch.nn as nn

class TensorizedTTAdapterPerBlock(nn.Module):
    """
    Adapter for applying MARS to TTUltraMIGSModule5DPerBlock with per-block independence.
    Each parameter block (xyz, scaling, rotation, dc, rest, opacity) has its own
    set of logits → independent masks for each TT rank position.
    """

    def __init__(self, tt_module):
        super().__init__()
        self.tt = tt_module  # instance of TTUltraMIGSModule5DPerBlock

        # --- Create phi_logits PER BLOCK (instead of a single global list) ---
        self.ranks_per_block = {}
        self.phi_logits_per_block = nn.ModuleDict()

        for name, block in self.tt.tt_blocks.items():
            # Each block has 5 TT cores
            c0, c1, c2, c3, c4 = block

            # Extract ranks for this block
            r1 = c0.shape[2]
            r2 = c1.shape[2]
            r3 = c2.shape[2]
            r4 = c3.shape[2]
            rM = c4.shape[2]

            ranks = [r1, r2, r3, r4, rM]
            self.ranks_per_block[name] = ranks

            # Create a ParameterList of logits for this block
            logits_list = nn.ParameterList([
                nn.Parameter(torch.zeros(r)) for r in ranks
            ])
            self.phi_logits_per_block[name] = logits_list

        # Map each rank position to the axes of the cores it connects
        self._pos_to_core_axes = {
            0: [("c0", 2), ("c1", 0)],  # r1
            1: [("c1", 2), ("c2", 0)],  # r2
            2: [("c2", 2), ("c3", 0)],  # r3
            3: [("c3", 2), ("c4", 0)],  # r4
            4: [("c4", 2)],             # rM
        }

    @property
    def cores(self):
        """Return all TT cores (for L2 regularization in MARS)."""
        out = []
        for block in self.tt.tt_blocks.values():
            out.extend(list(block))
        return out

    def _apply_mask_to_core(self, core, axis, mask):
        """Apply a soft mask along a given axis of a core."""
        shape = [1]*core.dim()
        shape[axis] = mask.shape[0]
        return core * mask.view(*shape)

    def _index_axis(self, x, axis, mask_bool):
        """Apply a hard mask (boolean indexing)."""
        index = [slice(None)] * x.dim()
        index[axis] = mask_bool
        return x[tuple(index)]

    def _masked_cores_for_block(self, block_cores, masks):
        """
        Apply masks to all cores of a block.
        masks: [mask_r1, mask_r2, mask_r3, mask_r4, mask_rM] (1D each).
        """
        c0, c1, c2, c3, c4 = block_cores
        hard = all(m.dtype == torch.bool for m in masks) and (not self.training)

        for pos, pairs in self._pos_to_core_axes.items():
            m = masks[pos]
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

    def reconstruct_all_with_masks(self, masks_per_block, idx_identity=0, original_order=False):
        """
        Reconstruct the full (G, M_total) tensor with per-block masks applied.
        """
        mats = []
        for name, _ in self.tt.block_specs:
            raw = self.tt.tt_blocks[name]
            masks = masks_per_block[name]  # ⬅️ block-specific masks
            masked = self._masked_cores_for_block([p for p in raw], masks)
            Wb = self.tt._contract_tt_identity_gemm_block(masked, idx_identity)
            mats.append(Wb)
        W_perm = torch.cat(mats, dim=1)
        if original_order:
            return W_perm[self.tt.inv_perm]
        return W_perm

    def forward(self, x, masks_per_block=None, idx_identity=0, original_order=False):
        """
        Forward pass with optional per-block masks.
        """
        if masks_per_block is None:
            return self.tt.reconstruct_all(idx_identity=idx_identity, original_order=original_order)
        return self.reconstruct_all_with_masks(masks_per_block, idx_identity=idx_identity, original_order=original_order)

    @torch.no_grad()
    def export_pruned(self, mars):
        """
        Export permanently pruned TT cores (after training with MARS).
        """
        pruned = {}
        for name, _ in self.tt.block_specs:
            logits_list = self.phi_logits_per_block[name]
            masks = [(logits > mars.eval_logits_threshold) for logits in logits_list]

            c0,c1,c2,c3,c4 = [p.clone() for p in self.tt.tt_blocks[name]]
            for pos, pairs in self._pos_to_core_axes.items():
                m = masks[pos]
                for tag, axis in pairs:
                    core = {"c0":c0,"c1":c1,"c2":c2,"c3":c3,"c4":c4}[tag]
                    core = self._index_axis(core, axis, m)
                    if   tag=="c0": c0=core
                    elif tag=="c1": c1=core
                    elif tag=="c2": c2=core
                    elif tag=="c3": c3=core
                    else:           c4=core
            pruned[name] = [c0,c1,c2,c3,c4]
        return pruned

    @torch.no_grad()
    def expand_first_core(self, n_identities: int):
        return self.tt.expand_first_core(n_identities)

    @torch.no_grad()
    def add_identity(self, *args, **kwargs):
        return getattr(self.tt, "add_identity", lambda *a, **k: None)(*args, **kwargs)

    @torch.no_grad()
    def get_W_for_identity(self, idx_identity: int, original_order: bool = True):
        return self.tt.get_W_for_identity(idx_identity, original_order)

