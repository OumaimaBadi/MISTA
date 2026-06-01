import torch
import numpy as np
import torch.nn as nn
from torch.distributions.utils import logits_to_probs, probs_to_logits, clamp_probs
from functools import partial


class MARSPerBlock(nn.Module):
    def __init__(self, tensorized_model, 
                 pi=1e-2, alpha=-1.0,
                 temperature=0.1, sigma_inv=0.1, 
                 gamma=-0.1, zeta=1.1,
                 eval_sample=False, ste=False,
                 eval_logits_threshold=0.0):
        """
        MARS per-block wrapping module.
        Works with TensorizedTTAdapterPerBlock: each block has its own logits/masks.
        """

        super().__init__()
        self.tensorized_model = tensorized_model  # instance of TensorizedTTAdapterPerBlock

        self.log_prior_prob = np.log(pi)
        self.log_prior_prob_c = np.log(1.0 - pi)

        self.temperature = temperature
        self.l2_weight = 0.5 * sigma_inv ** 2

        # Hard Concrete constants
        self.gamma = gamma
        self.zeta = zeta
        self.zmg = self.zeta - self.gamma

        self.eval_sample = eval_sample
        self.ste = ste
        self.eval_logits_threshold = eval_logits_threshold

        self.F = partial(logits_to_probs, is_binary=True)  
        self.F_inv = partial(probs_to_logits, is_binary=True)  
        self.warmup = False  

        # Collect per-block logits from the adapter
        self.phi_logits_per_block = self.tensorized_model.phi_logits_per_block  

        # Create a flat view for optimizer/compatibility
        self.ranks = []
        self.phi_logits_list = nn.ParameterList()
        for name, logits_list in self.phi_logits_per_block.items():
            for logits in logits_list:
                self.ranks.append(len(logits))
                self.phi_logits_list.append(logits)

        self.optimizer = None
        self.scheduler = None

    # ----------------------------------------------------
    def get_mask(self, logits):
        "Get mask from phi logits (soft during training, hard at eval)."
        if self.eval_sample or self.training:
            u = clamp_probs(torch.rand(logits.shape, dtype=logits.dtype, device=logits.device))
            logits = logits + self.F_inv(u)  

            if self.training:
                # soft sample
                s = self.F(logits / self.temperature)
                s = s * self.zmg + self.gamma
                s = torch.clamp(s, min=0.0, max=1.0)

                if self.ste:
                    s_ste = torch.round(s)
                    s = (s_ste - s).detach() + s
                return s

        # hard mask
        return logits > self.eval_logits_threshold

    # ----------------------------------------------------
    def forward(self, x):
        if self.warmup:
            return self.tensorized_model(x)

        # ✅ build masks per-block (dict)
        masks_per_block = {}
        for name, logits_list in self.phi_logits_per_block.items():
            masks_per_block[name] = [self.get_mask(logits) for logits in logits_list]

        return self.tensorized_model(x, masks_per_block=masks_per_block)

    # ----------------------------------------------------
    def compute_reg(self):
        "Compute the MARS regularizer term: log p(m) + log p(G)."
        reg = 0.0
        # use flat list for convenience
        probs_list = [self.F(logits) for logits in self.phi_logits_list]

        for probs in probs_list:
            probs_c = 1 - probs
            reg += torch.sum(probs * self.log_prior_prob + probs_c * self.log_prior_prob_c)  

        if self.l2_weight > 0:
            reg -= self.l2_weight * sum(torch.sum(core ** 2) for core in self.tensorized_model.cores)

        return reg

    # ----------------------------------------------------
    @torch.no_grad()
    def get_W_for_identity(self, idx: int, original_order: bool = True):
        return self.tensorized_model.forward(
            x=None, masks_per_block=None, idx_identity=idx, original_order=original_order
        )

    def expand_first_core(self, n_identities: int):
        return self.tensorized_model.expand_first_core(n_identities)

    def add_identity(self, *args, **kwargs):
        return getattr(self.tensorized_model, "add_identity", None)(*args, **kwargs)


    # ---------------------------
    #  PHI-ONLY OPTIMIZATION API
    # ---------------------------
    def set_phi_optimizer(self, opt_cfg):
        """
        EN: Create the optimizer for phi logits ONLY (no TT cores here).
        FR: Crée l'optimizer pour les logits phi UNIQUEMENT (pas de cores TT ici).
        """
        mars_lr = float(opt_cfg.get("mars_lr", 1e-3))
        self.optimizer = torch.optim.Adam(
            [{"params": list(self.phi_logits_list), "lr": mars_lr}]
        )
        self.scheduler = None  # plug a scheduler if you want

    def step_phi(self, iteration=None):
        """
        EN: Single optimization step for phi logits.
        FR: Step d'optimisation pour les logits phi.
        """
        if self.optimizer is None:
            return
        self.optimizer.step()
        self.optimizer.zero_grad()

    # ---------------------------
    #  BACKWARD COMPAT SHIMS
    # ---------------------------
    def set_optimizer(self, opt_cfg):
        """
        EN: Backward-compatible alias. Legacy code that calls set_optimizer()
            will now configure PHI optimizer only.
        FR: Compatibilité descendante: configure seulement l'optimizer PHI.
        """
        self.set_phi_optimizer(opt_cfg)

    def step(self, iteration=None):
        """
        EN: Backward-compatible alias. Legacy code that calls step()
            will now step PHI optimizer only.
        FR: Compat descendante: step uniquement de l'optimizer PHI.
        """
        self.step_phi(iteration)


# --------------------------------------------------------
def compute_cum_reg(model):
    reg = 0.0
    for layer in model.modules():
        if isinstance(layer, (MARSPerBlock,)):  
            reg += layer.compute_reg()
    return reg


class MARSLoss(nn.Module):
    def __init__(self, model, train_size, criterion=None, reg_term_coef=1.0):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.reg_term_coef = reg_term_coef / train_size  

    def forward(self, output, target):
        neg_data_term = self.criterion(output, target)
        reg_term = compute_cum_reg(self.model)
        return neg_data_term - self.reg_term_coef * reg_term


def get_MARS_attr(model, attr_name):
    for layer in model.modules():
        if isinstance(layer, (MARSPerBlock,)):
            return getattr(layer, attr_name)


def set_MARS_attr(model, attr_name, attr_value):
    for layer in model.modules():
        if isinstance(layer, (MARSPerBlock,)):
            setattr(layer, attr_name, attr_value)
