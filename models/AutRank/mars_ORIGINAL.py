import torch
import numpy as np
import torch.nn as nn
from torch.distributions.utils import logits_to_probs, probs_to_logits, clamp_probs
from functools import partial


class MARS(nn.Module):
    def __init__(self, tensorized_model, 
                 pi,
                 initial_mask_prob,
                 temperature,
                 enable_temp_decay,
                 temp_gamma,
                 temp_end,
                 lambda_sparsity,
                 lambda_binary,
                 sigma_inv, 
                 gamma, zeta,
                 eval_sample, ste,
                 eval_logits_threshold,
                 warmup_iterations,
                 mask_warmup_iterations,
                 phi_grad_delay,
                 min_mask_value,
                 grad_clip_value):
        """
        MARS with ALL optimizations for best results.
        
        Key Parameters:
        ---------------
        initial_mask_prob : float (0.95)
            Initial mask probability (high = start gentle).
        temperature : float (0.5)
            Initial temperature for Binary Concrete.
        temp_gamma : float (0.99)
            Temperature decay factor (0.99 = slow decay).
        temp_end : float (0.01)
            Minimum temperature.
        warmup_iterations : int (5000)
            Iterations without masks (normal learning).
        mask_warmup_iterations : int (2000)
            Transition iterations (progressive activation).
        min_mask_value : float (0.05)
            Minimum mask value (prevents total collapse).
        grad_clip_value : float (1.0)
            Gradient clipping threshold.
        """
        super().__init__()

        self.tensorized_model = tensorized_model

        self.log_prior_prob = np.log(pi)
        self.log_prior_prob_c = np.log(1.0 - pi)

        # Temperature with decay
        self.enable_temp_decay= enable_temp_decay
        self.temperature = temperature
        self.temp_init = temperature
        self.temp_gamma = temp_gamma
        self.temp_end = temp_end
        self.lambda_sparsity = 0.0
        self.lambda_sparsity_final = lambda_sparsity      
        self.lambda_binary = lambda_binary

        #self.lambda_multipliers =[3.5,2.5,1.0,0.3] #[3.5, 2.5, 0.8, 0.2] 


        
        self.l2_weight = 0.5 * sigma_inv ** 2

        # Hard Concrete
        self.gamma = gamma
        self.zeta = zeta
        self.zmg = self.zeta - self.gamma

        self.eval_sample = eval_sample
        self.ste = ste
        self.eval_logits_threshold = eval_logits_threshold

        self.F = partial(logits_to_probs, is_binary=True)
        self.F_inv = partial(probs_to_logits, is_binary=True)
        
        # Warmup and transition
        self.warmup = False
        self.warmup_iterations = warmup_iterations
        self.mask_warmup_iterations = mask_warmup_iterations
        self.current_iteration = 0
        self.phi_grad_delay = phi_grad_delay
        self.phi_freeze_iteration = 40000
        # Stability
        self.grad_clip_value = grad_clip_value
        self.min_mask_value = min_mask_value
        
        # Optimal logits initialization
        initial_logits_value = probs_to_logits(
            torch.tensor(initial_mask_prob), is_binary=True
        ).item()

        self.phi_logits_list = []        
        for R in self.tensorized_model.ranks:
            logits = nn.Parameter(torch.Tensor(R))
            logits.data.normal_(initial_logits_value, 1e-3)
            self.phi_logits_list.append(logits)
        self.phi_logits_list = nn.ParameterList(self.phi_logits_list)

    # def decay_temperature(self):
    #     if self.current_iteration >= self.warmup_iterations:
    #         self.temperature = max(self.temp_end, self.temperature * self.temp_gamma)

    # models/AutRank/mars.py, ligne ~80
    def decay_temperature(self):
        """Linear temperature decay."""
        if not self.enable_temp_decay:
            return  

        if self.current_iteration < self.warmup_iterations:
            return
        
        total_iters = 50000
        decay_iters = total_iters - self.warmup_iterations
        progress = (self.current_iteration - self.warmup_iterations) / decay_iters
        progress = min(1.0, max(0.0, progress))
        
        self.temperature = self.temp_init * (1 - progress) + self.temp_end * progress

    def update_lambda_sparsity(self, total_iterations=50000):
        """Rampe lambda_sparsity de 0.0 (iter 5k) → final (iter 50k)"""
        if self.current_iteration < self.warmup_iterations:
            self.lambda_sparsity = 0.0
        elif self.current_iteration < total_iterations:
            ramp_duration = total_iterations - self.warmup_iterations
            progress = (self.current_iteration - self.warmup_iterations) / ramp_duration
            progress = min(1.0, max(0.0, progress))
            self.lambda_sparsity = progress * self.lambda_sparsity_final
        else:
            self.lambda_sparsity = self.lambda_sparsity_final


    # def get_mask(self, logits):
    #     """Get masks with progressive transition and constraints."""
    #     # Compute mask_strength (0 → 1)
    #     if self.current_iteration < self.warmup_iterations:
    #         mask_strength = 0.0
    #     elif self.current_iteration < self.warmup_iterations + self.mask_warmup_iterations:
    #         progress = (self.current_iteration - self.warmup_iterations) / self.mask_warmup_iterations
    #         mask_strength = progress
    #     else:
    #         mask_strength = 1.0
        
    #     if self.eval_sample or self.training:
    #         u = clamp_probs(torch.rand(logits.shape, dtype=logits.dtype, device=logits.device))
    #         logits = logits + self.F_inv(u)

    #         if self.training:
    #             # Binary Concrete with current temperature
    #             s = self.F(logits / self.temperature)
    #             s = s * self.zmg + self.gamma
    #             s = torch.clamp(s, min=0.0, max=1.0)
                
    #             # Minimum constraint
    #             s = torch.clamp(s, min=self.min_mask_value, max=1.0)
                
    #             # Progressive transition
    #             s = (1.0 - mask_strength) * 1.0 + mask_strength * s

    #             if self.ste:
    #                 s_ste = torch.round(s)
    #                 s = (s_ste - s).detach() + s

    #             return s

    #     return logits > self.eval_logits_threshold

    # def get_mask(self, logits):
    #     if self.eval_sample or self.training:
    #         u = clamp_probs(torch.rand_like(logits))
    #         logits_noise = logits + self.F_inv(u)

    #         if self.training:
    #             s = self.F(logits_noise / self.temperature)
    #             s = s * self.zmg + self.gamma
    #             s = torch.clamp(s, 0.0, 1.0)

    #             if self.ste:
    #                 s_ste = torch.round(s)
    #                 s = (s_ste - s).detach() + s

    #             return s

    #     return logits > self.eval_logits_threshold

    def get_mask(self, logits):
        """
        Get masks with STE (Straight-Through Estimator).
        
        Forward: Hard masks (0 or 1) for stability.
        Backward: Soft gradients for optimization.
        """
        if self.training:
            # Compute soft mask
            s_soft = torch.sigmoid(logits / self.temperature)
            
            if self.ste:
                # Round to get hard mask
                s_hard = torch.round(s_soft)
                
                # STE: forward uses hard, backward uses soft gradient
                s = (s_hard - s_soft).detach() + s_soft
                return s
            else:
                # No STE: return soft mask
                return s_soft
        else:
            # EVAL: Hard pruning (boolean threshold)
            return (torch.sigmoid(logits) > 0.5) #return (logits > self.eval_logits_threshold)  # threshold = 0.0 au début


    def get_soft_mask(self, logits):
        """Always return soft mask (for regularization)."""
        return torch.sigmoid(logits / self.temperature)



    def forward(self, x):
        if self.warmup or self.current_iteration < self.warmup_iterations:
            return self.tensorized_model(x)

        masks = [self.get_mask(logits) for logits in self.phi_logits_list]
        return self.tensorized_model(x, masks)
        
    # def compute_reg(self):
    #     """MARS regularization."""
    #     reg = 0.0
    #     probs_list = [self.F(logits) for logits in self.phi_logits_list]
        
    #     for probs in probs_list:
    #         probs_c = 1 - probs
    #         reg += torch.sum(probs * self.log_prior_prob + probs_c * self.log_prior_prob_c)  
            
    #     if self.l2_weight > 0:
    #         # Reduced L2 during warmup
    #         if self.current_iteration < self.warmup_iterations + self.mask_warmup_iterations:
    #             effective_l2 = self.l2_weight * 0.1
    #         else:
    #             effective_l2 = self.l2_weight
            
    #         reg -= effective_l2 * sum(torch.sum(core ** 2) for core in self.tensorized_model.cores)

    #     return reg

    # def compute_reg(self):
    #     reg = 0.0
    #     probs_list = [self.F(logits) for logits in self.phi_logits_list]

    #     for probs in probs_list:
    #         probs_c = 1 - probs
    #         reg += torch.sum(probs * self.log_prior_prob + probs_c * self.log_prior_prob_c)

    #     # pas de L2 pour le moment
    #     return reg

    # def compute_reg(self):
    # #test1: full deterministic
    #     if self.current_iteration < self.warmup_iterations:
    #         return torch.tensor(0.0, device=self.phi_logits_list[0].device)
        
    #     reg_l1 = 0.0
    #     reg_binary = 0.0
        
    #     for logits in self.phi_logits_list:
    #         s = self.get_mask(logits)
    #         reg_l1 += s.sum()
    #         reg_binary += (s * (1 - s)).sum()
        
    #     return self.lambda_sparsity * reg_l1 + self.lambda_binary * reg_binary

    # Dans mars.py, ligne ~180 (remplace compute_reg() complet):

    def compute_reg(self):
        if self.current_iteration < self.warmup_iterations:
            return torch.tensor(0.0, device=self.phi_logits_list[0].device)
        
        reg_l1 = 0.0
        min_active = 0.25
        
        for rank_idx, logits in enumerate(self.phi_logits_list):
            s = self.get_soft_mask(logits)
            
            active_frac = s.mean().detach()
            if active_frac < min_active:
                continue
            
            reg_l1 += self.lambda_sparsity * s.sum()
        
        return reg_l1

    def get_all_masks(self):
        """
        Retourne les masques des 4 ranks pour l'analyse d'importance.
        
        Returns:
            dict: {'r1': tensor, 'r2': tensor, 'r3': tensor, 'r4': tensor}
        """
        masks_dict = {}
        
        for i, logits in enumerate(self.phi_logits_list):
            # ✅ APPELLE get_mask() dans TOUS les cas (train ET eval)
            mask = self.get_mask(logits)
            
            rank_name = f"r{i+1}"
            masks_dict[rank_name] = mask
        
        return masks_dict
    
    # def compute_reg(self):
    #     """
    #     Bernoulli prior regularization (MARS original).
        
    #     Returns:
    #         -log p(z | π) où π est le prior sur les masques
    #     """
    #     if self.current_iteration < self.warmup_iterations:
    #         return torch.tensor(0.0, device=self.phi_logits_list[0].device)
        
    #     reg = 0.0
        
    #     for logits in self.phi_logits_list:
    #         # IMPORTANT : Utilise sigmoid(logits) PAS sigmoid(logits/temp)
    #         # Car le prior est sur les probabilités marginales, pas conditionnelles
    #         probs = torch.sigmoid(logits)
    #         probs_c = 1 - probs
            
    #         reg += torch.sum(
    #             probs * self.log_prior_prob +      # p(z=1) · log(π)
    #             probs_c * self.log_prior_prob_c    # p(z=0) · log(1-π)
    #         )
        
    #     # Retourne -reg car on MINIMISE (c'est un prior favorable = reg négatif)
    #     return -reg


    def get_W_for_identity(self, idx: int, original_order: bool = True):
        """Get W for a specific identity with proper masking."""
        # Warmup phase: no masks
        if self.warmup or self.current_iteration < self.warmup_iterations:
            return self.tensorized_model.forward(
                x=None, masks=None, idx_identity=idx, original_order=original_order
            )

        # ✅ APPELLE get_mask() dans TOUS les cas (train ET eval)
        masks = [self.get_mask(logits) for logits in self.phi_logits_list]

        return self.tensorized_model.forward(
            x=None, masks=masks, idx_identity=idx, original_order=original_order
        )
        
    def expand_first_core(self, n_identities: int):
        return self.tensorized_model.expand_first_core(n_identities)

    def add_identity(self, *args, **kwargs):
        method = getattr(self.tensorized_model, "add_identity", None)
        if method is None:
            raise AttributeError(f"{type(self.tensorized_model).__name__} has no add_identity method")
        return method(*args, **kwargs)
        
    def set_phi_optimizer(self, opt_cfg):
        """Optimizer for φ."""
        mars_lr = float(opt_cfg.get("mars_lr", 1e-3))
        self.optimizer = torch.optim.Adam(
            [{"params": list(self.phi_logits_list), "lr": mars_lr}]
        )
        self.scheduler = None

    def step_phi(self, iteration=None):
        if self.optimizer is None:
            return
        
        if iteration is not None:
            self.current_iteration = iteration
        
        # Phase 1 : φ gelé avant delay
        phi_grad_delay = getattr(self, "phi_grad_delay", 0)
        if iteration is not None and iteration < phi_grad_delay:
            for logits in self.phi_logits_list:
                if logits.grad is not None:
                    logits.grad.zero_()
            self.optimizer.zero_grad()
            return
        
        # Phase 3 : φ gelé en fin de training
        if self.phi_freeze_iteration and iteration is not None and iteration >= self.phi_freeze_iteration:
            for logits in self.phi_logits_list:
                if logits.grad is not None:
                    logits.grad.zero_()
            self.optimizer.zero_grad()
            return
        
        # ===== NORMALISATION PAR RANG =====
        for logits in self.phi_logits_list:
            if logits.grad is not None:
                grad_norm = logits.grad.norm() + 1e-8
                logits.grad.data.div_(grad_norm)
        # ==================================
        
        if self.grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.phi_logits_list, self.grad_clip_value
            )
        
        self.optimizer.step()
        self.optimizer.zero_grad()

    # def clip_tt_gradients(self):
    #     """Clip gradients of TT cores."""
    #     if self.grad_clip_value > 0:
    #         params = list(self.tensorized_model.cores)
    #         torch.nn.utils.clip_grad_norm_(params, self.grad_clip_value)

    def set_optimizer(self, opt_cfg):
        self.set_phi_optimizer(opt_cfg)

    def step(self, iteration=None):
        self.step_phi(iteration)


def compute_cum_reg(model):
    """Compute cumulative MARS regularizer."""
    reg = 0.0
    for layer in model.modules():
        if isinstance(layer, MARS):
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
        
        if torch.isnan(neg_data_term):
            print("WARNING: NaN in data term!")
        if torch.isnan(reg_term):
            print("WARNING: NaN in regularization!")
        
        loss = neg_data_term - self.reg_term_coef * reg_term
        
        if torch.isnan(loss):
            print("ERROR: NaN loss! Returning data term only.")
            return neg_data_term
        
        return loss


def get_MARS_attr(model, attr_name):
    for layer in model.modules():
        if isinstance(layer, MARS):
            return getattr(layer, attr_name)


def set_MARS_attr(model, attr_name, attr_value):
    for layer in model.modules():
        if isinstance(layer, MARS):
            setattr(layer, attr_name, attr_value)