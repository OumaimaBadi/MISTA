import torch
import numpy as np

def compute_frobenius_LR(tt_module):
    """
    Importance Frobenius SANS canonisation, alignée avec les ranks TT/MARS.
    Retourne seulement L et R (pas de prod).
    """
    with torch.no_grad():
        cores = [
            tt_module.tt_tensor_gpu[0],  # Core0: (1, n_id, r1)
            tt_module.tt_tensor_gpu[1],  # Core1: (r1, m1, r2)
            tt_module.tt_tensor_gpu[2],  # Core2: (r2, m2, r3)
            tt_module.tt_tensor_gpu[3],  # Core3: (r3, m3, r4)
            tt_module.recombine_core4(), # Core4: (r4, m4, 1)
        ]

        results = {}

        # r1: Core0 ↔ Core1
        results["r1"] = {
            "frob_L": torch.norm(cores[0], dim=(0, 1)).cpu().numpy(),  # (r1,)
            "frob_R": torch.norm(cores[1], dim=(1, 2)).cpu().numpy(),  # (r1,)
        }

        # r2: Core1 ↔ Core2
        results["r2"] = {
            "frob_L": torch.norm(cores[1], dim=(0, 1)).cpu().numpy(),  # (r2,)
            "frob_R": torch.norm(cores[2], dim=(1, 2)).cpu().numpy(),  # (r2,)
        }

        # r3: Core2 ↔ Core3
        results["r3"] = {
            "frob_L": torch.norm(cores[2], dim=(0, 1)).cpu().numpy(),  # (r3,)
            "frob_R": torch.norm(cores[3], dim=(1, 2)).cpu().numpy(),  # (r3,)
        }

        # r4: Core3 ↔ Core4
        results["r4"] = {
            "frob_L": torch.norm(cores[3], dim=(0, 1)).cpu().numpy(),  # (r4,)
            "frob_R": torch.norm(cores[4], dim=(1, 2)).cpu().numpy(),  # (r4,)
        }

    return results


def compute_mars_probs(migs_module):
    """
    Calcule les probabilités MARS (soft masks) = sigmoid(φ / T) pour chaque rang.
    
    Compatible avec :
    - MARS (shared masks)
    - MARSPerBlock (per-block masks, on moyenne sur les blocks)
    
    Args:
        migs_module: scene.migs_module (peut être MARS, MARSPerBlock, ou TT direct)
    
    Returns:
        dict: {"r1": array(n,), "r2": array(n,), "r3": array(n,), "r4": array(n,)}
              ou None si MARS n'est pas actif
    """
    # Check if MARS is active
    if not hasattr(migs_module, "phi_logits_list"):
        print("  [MARS PROBS] No phi_logits_list found → MARS not active")
        return None
    
    # Get temperature
    temperature = float(getattr(migs_module, "temperature", 1.0))
    print(f"  [MARS PROBS] Using temperature = {temperature:.6f}")
    
    results = {}
    rank_names = ["r1", "r2", "r3", "r4"]
    
    with torch.no_grad():
        # Case 1: MARS (shared masks) - phi_logits_list is direct ParameterList
        if hasattr(migs_module.phi_logits_list, "__iter__"):
            for i, (rank_name, logits) in enumerate(zip(rank_names, migs_module.phi_logits_list)):
                # Compute soft mask: sigmoid(φ / T)
                probs = torch.sigmoid(logits / temperature)
                results[rank_name] = probs.cpu().numpy()
                
                print(f"  [MARS PROBS] {rank_name}: shape={probs.shape}, "
                      f"min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
        
        # Case 2: MARSPerBlock - phi_logits_list is dict[block_name -> ParameterList]
        elif isinstance(migs_module.phi_logits_list, dict):
            print("  [MARS PROBS] Detected MARSPerBlock → averaging across blocks")
            
            for rank_idx, rank_name in enumerate(rank_names):
                rank_probs = []
                
                for block_name, block_logits in migs_module.phi_logits_list.items():
                    if rank_idx < len(block_logits):
                        logits = block_logits[rank_idx]
                        probs = torch.sigmoid(logits / temperature)
                        rank_probs.append(probs)
                
                if len(rank_probs) > 0:
                    # Average across blocks
                    avg_probs = torch.stack(rank_probs, dim=0).mean(dim=0)
                    results[rank_name] = avg_probs.cpu().numpy()
                    
                    print(f"  [MARS PROBS] {rank_name}: averaged {len(rank_probs)} blocks, "
                          f"shape={avg_probs.shape}, mean={avg_probs.mean():.4f}")
    
    if len(results) == 4:
        print("  MARS probabilities computed successfully")
        return results
    else:
        print(f"  WARNING: Only {len(results)}/4 ranks computed")
        return None


def add_frobenius_sum(frob_LR):
    """
    Ajoute une clé 'frob_sum' = frob_L + frob_R pour chaque rank.
    """
    out = {}
    for rk, d in frob_LR.items():
        L = d["frob_L"]
        R = d["frob_R"]
        out[rk] = dict(d)
        out[rk]["frob_sum"] = (L + R)
    return out
