"""
Learned-Geometry Wasserstein Barycenter (Thread 3)
===================================================

Learns bin embeddings φ_θ : {0,…,K-1} → ℝ^d jointly with the barycenter
computation, trained to minimize out-of-sample RPS.

Key classes
-----------
BinEmbedding(K, d, monotone)
    Embedding table K → ℝ^d with optional monotone constraint (d=1 only).
    Regularization: centering + scale normalization applied in forward().

SinkhornLayer(n_iter, eps)
    Differentiable Wasserstein barycenter via Sinkhorn iterations.
    Accepts a panel of distributions and a cost matrix; returns the
    regularized barycenter PMF.

LearnedWassersteinBarycenter(K, d, n_sinkhorn, eps_sinkhorn)
    End-to-end model: BinEmbedding → cost matrix → SinkhornLayer → PMF.

WeightNetwork(K, d_h)
    Deep Sets permutation-invariant forecaster weight network (optional).
    Returns softmax weights over n forecasters.

Loss functions
--------------
rps_loss(cdf_pred, y_bin, K)
    Ranked Probability Score (Brier-based, quadratic in CDF error).

geometric_regularization(phi, K, lambda_smooth)
    Smoothness + centering penalty on the embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ────────────────────────────────────────────────────────────────────────────
# Bin Embedding
# ────────────────────────────────────────────────────────────────────────────

class BinEmbedding(nn.Module):
    """
    Learnable embedding φ_θ : {0,…,K-1} → ℝ^d.

    Initialized to natural spacing b/(K-1) in each dimension.
    A residual perturbation η_θ(b) is learned on top.

    Parameters
    ----------
    K         : number of bins
    d         : embedding dimension (1 = ordered line; 2+ = higher-dim geometry)
    monotone  : if True and d=1, enforce strict monotonicity via cumsum(softplus)
    """

    def __init__(self, K: int = 10, d: int = 1, monotone: bool = True):
        super().__init__()
        self.K = K
        self.d = d
        self.monotone = monotone and (d == 1)

        # Natural spacing initialization
        natural = torch.linspace(0.0, 1.0, K).unsqueeze(1).expand(K, d)  # (K, d)

        if self.monotone:
            # Parameterize as cumulative sum of softplus values → strictly increasing
            # Initialize so that cumsum(softplus(raw)) ≈ natural spacing
            # softplus(0) = log(2) ≈ 0.693; spacing = 1/(K-1) ≈ 0.111
            # We want cumsum(softplus(raw)) * scale ≈ natural
            # Simple init: raw = softplus_inv(1/(K-1)) repeated
            target_step = 1.0 / (K - 1)
            # softplus_inv(x) = log(exp(x) - 1) for x > 0
            raw_init = torch.log(
                torch.exp(torch.tensor(target_step)) - 1.0
            ).item()
            self.raw_steps = nn.Parameter(
                torch.full((K,), raw_init)  # (K,), cumsum gives positions
            )
            self.scale = nn.Parameter(torch.tensor(1.0))
        else:
            # Free embedding table, initialized to natural spacing
            self.phi = nn.Parameter(natural.clone())

    def forward(self) -> torch.Tensor:
        """
        Returns φ_θ ∈ ℝ^{K × d}, with centering and unit-scale normalization.
        """
        if self.monotone:
            # Strictly increasing positions in ℝ
            steps = F.softplus(self.raw_steps)         # (K,) all positive
            positions = torch.cumsum(steps, dim=0)     # (K,) strictly increasing
            positions = positions - positions.mean()   # center
            # Normalize to unit RMS
            rms = positions.pow(2).mean().sqrt().clamp(min=1e-6)
            positions = positions / rms
            phi = positions.unsqueeze(1)               # (K, 1)
        else:
            phi = self.phi                             # (K, d)
            phi = phi - phi.mean(dim=0, keepdim=True)  # center each dim
            rms = phi.pow(2).mean().sqrt().clamp(min=1e-6)
            phi = phi / rms

        return phi  # (K, d)

    def cost_matrix(self, p: int = 1) -> torch.Tensor:
        """
        C[a, b] = ||φ_θ(a) - φ_θ(b)||^p  →  (K, K) tensor.

        p=1 (default): L¹ cost.  Under natural spacing, the Sinkhorn W₁
            barycenter converges to the CDF-median (Proposition 2.2) as ε→0.
        p=2: squared-L² cost.  Gives the W₂ (Fréchet mean) barycenter.
        """
        phi = self.forward()                          # (K, d)
        diff = phi.unsqueeze(1) - phi.unsqueeze(0)   # (K, K, d)
        if p == 1:
            C = diff.abs().sum(dim=-1)               # (K, K)
        else:
            C = (diff ** 2).sum(dim=-1)              # (K, K)
        return C

    def smoothness_penalty(self) -> torch.Tensor:
        """Sum of squared adjacent-bin distances in embedding space."""
        phi = self.forward()                         # (K, d)
        diffs = phi[1:] - phi[:-1]                  # (K-1, d)
        return (diffs ** 2).sum()


# ────────────────────────────────────────────────────────────────────────────
# Sinkhorn Barycenter Layer
# ────────────────────────────────────────────────────────────────────────────

class SinkhornLayer(nn.Module):
    """
    Differentiable Wasserstein barycenter via Sinkhorn–Knopp iterations.

    Given a panel P = {p^(i)}_{i=1}^n of distributions over K bins (each row
    of `panel` is a distribution), and a cost matrix C (K×K), computes the
    regularized barycenter:

        b* = argmin_{b ∈ Δ^K} Σ_i w_i W²_ε(p^(i), b; C)

    Solved via the fixed-point iteration of Cuturi & Doucet (2014) / Benamou
    et al. (2015):  T Sinkhorn iterations with Gibbs kernel K_ε = exp(-C/ε).

    Parameters
    ----------
    n_iter : number of Sinkhorn iterations (50 is usually sufficient)
    eps    : regularization strength (0.1 → moderate; 0.01 → near-exact OT)
    """

    def __init__(self, n_iter: int = 50, eps: float = 0.1):
        super().__init__()
        self.n_iter = n_iter
        self.eps = eps

    def forward(
        self,
        panel: torch.Tensor,        # (n, K)  forecaster distributions
        cost: torch.Tensor,         # (K, K)  ground cost matrix
        weights: Optional[torch.Tensor] = None,  # (n,) or None (equal)
    ) -> torch.Tensor:
        """
        Returns barycenter PMF b* ∈ Δ^K as (K,) tensor.
        """
        n, K = panel.shape
        if weights is None:
            weights = torch.ones(n, device=panel.device) / n
        else:
            weights = weights / weights.sum()

        # Gibbs kernel: K_ε[a,b] = exp(-C[a,b]/ε)
        log_K = -cost / self.eps   # (K, K)

        # Initialize dual variables
        log_b = torch.zeros(K, device=panel.device)   # log of barycenter

        # Sinkhorn iterations (log-domain for numerical stability)
        log_panel = torch.log(panel.clamp(min=1e-10))  # (n, K)

        for _ in range(self.n_iter):
            # For each source i, compute log u^(i):
            #   log u^(i)[a] = log p^(i)[a] - log( Σ_b K_ε[a,b] exp(log_b[b]) )
            # i.e., log u^(i) = log_p^(i) - log_softmax_style stabilization

            # log( K_ε @ exp(log_b) ) computed stably:
            #   = logsumexp_b( log_K[a,b] + log_b[b] )
            log_Kb = torch.logsumexp(
                log_K + log_b.unsqueeze(0), dim=1
            )  # (K,) — same for all i since K_ε and b are shared

            log_u = log_panel - log_Kb.unsqueeze(0)  # (n, K)

            # Update log_b:
            #   log b[a] = Σ_i w_i log( Σ_j K_ε[j,a] exp(log u^(i)[j]) )
            # = weighted average over i of log( K_ε^T @ u^(i) )
            log_KT = log_K.t()  # (K, K) transposed

            log_KTu = torch.logsumexp(
                log_KT.unsqueeze(0) + log_u.unsqueeze(2), dim=1
            )  # (n, K): log_KTu[i, a] = log Σ_j K_ε[j,a] u^(i)[j]

            log_b = (weights.unsqueeze(1) * log_KTu).sum(dim=0)  # (K,)

        # Convert log_b to normalized PMF
        b = torch.exp(log_b - log_b.max())
        b = b / b.sum().clamp(min=1e-10)
        return b


# ────────────────────────────────────────────────────────────────────────────
# Deep Sets Weight Network (optional)
# ────────────────────────────────────────────────────────────────────────────

class WeightNetwork(nn.Module):
    """
    Permutation-invariant forecaster weight network (Deep Sets).

    Architecture:
        h_θ : ℝ^K → ℝ^{d_h}    (per-forecaster feature map)
        ρ_θ : ℝ^{d_h} → ℝ^n    (score each forecaster given pooled summary)

    w_i = softmax(ρ_θ(Σ_j h_θ(p^(j))))[i]

    Parameters
    ----------
    K   : number of bins (input dimension per forecaster)
    d_h : hidden dimension for the pooled representation
    """

    def __init__(self, K: int = 10, d_h: int = 16):
        super().__init__()
        self.h = nn.Sequential(
            nn.Linear(K, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
        )
        # Score each forecaster: concat(pool, h(p_i)) → scalar
        self.score = nn.Sequential(
            nn.Linear(2 * d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, 1),
        )

    def forward(self, panel: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        panel : (n, K) distributions

        Returns
        -------
        weights : (n,) softmax weights
        """
        h_vals = self.h(panel)                    # (n, d_h)
        pool = h_vals.sum(dim=0, keepdim=True)    # (1, d_h)
        pool_rep = pool.expand(h_vals.shape[0], -1)  # (n, d_h)
        combined = torch.cat([pool_rep, h_vals], dim=1)  # (n, 2*d_h)
        scores = self.score(combined).squeeze(1)  # (n,)
        return F.softmax(scores, dim=0)           # (n,)


# ────────────────────────────────────────────────────────────────────────────
# Full Model
# ────────────────────────────────────────────────────────────────────────────

class LearnedWassersteinBarycenter(nn.Module):
    """
    End-to-end learned-geometry Wasserstein barycenter.

    Combines:
      - BinEmbedding: learns the ground metric C_θ
      - SinkhornLayer: differentiable barycenter computation
      - WeightNetwork (optional): permutation-invariant forecaster weights

    Parameters
    ----------
    K             : number of outcome bins
    d             : embedding dimension (1 = ordered line geometry)
    n_sinkhorn    : Sinkhorn iterations
    eps_sinkhorn  : Sinkhorn regularization
    monotone      : enforce monotone ordering for d=1
    use_weights   : if True, learn forecaster weights via WeightNetwork
    d_h           : hidden dim for WeightNetwork
    """

    def __init__(
        self,
        K: int = 10,
        d: int = 1,
        n_sinkhorn: int = 50,
        eps_sinkhorn: float = 0.05,
        cost_p: int = 1,
        monotone: bool = True,
        use_weights: bool = False,
        d_h: int = 16,
    ):
        super().__init__()
        self.K = K
        self.embedding = BinEmbedding(K=K, d=d, monotone=monotone)
        self.sinkhorn = SinkhornLayer(n_iter=n_sinkhorn, eps=eps_sinkhorn)
        self.cost_p = cost_p
        self.use_weights = use_weights
        if use_weights:
            self.weight_net = WeightNetwork(K=K, d_h=d_h)

    def forward(
        self,
        panel: torch.Tensor,  # (n, K) forecaster distributions
    ) -> torch.Tensor:
        """
        Returns aggregate PMF b* ∈ Δ^K as (K,) tensor.
        """
        cost = self.embedding.cost_matrix(p=self.cost_p)   # (K, K)

        if self.use_weights:
            weights = self.weight_net(panel)  # (n,)
        else:
            weights = None

        b = self.sinkhorn(panel, cost, weights)  # (K,)
        return b

    def predict_cdf(self, panel: torch.Tensor) -> torch.Tensor:
        """Returns CDF of the barycenter (K-1 values, cumsum up to K-1)."""
        pmf = self.forward(panel)             # (K,)
        cdf = torch.cumsum(pmf, dim=0)        # (K,)
        return cdf[:-1]                       # (K-1,) CDF at first K-1 bins

    def smoothness_penalty(self) -> torch.Tensor:
        return self.embedding.smoothness_penalty()


# ────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ────────────────────────────────────────────────────────────────────────────

def rps_loss(
    pmf: torch.Tensor,     # (K,) predicted PMF
    y_bin: int,            # realized bin index (0-based)
    K: int,
) -> torch.Tensor:
    """
    Ranked Probability Score: RPS = Σ_{k=0}^{K-2} (CDF_pred(k) - 1{y ≤ k})²

    Lower is better.
    """
    cdf_pred = torch.cumsum(pmf, dim=0)[:-1]   # (K-1,)
    cdf_true = torch.zeros(K - 1, device=pmf.device)
    cdf_true[y_bin:] = 1.0
    return ((cdf_pred - cdf_true) ** 2).sum()


def geometric_regularization(
    model: LearnedWassersteinBarycenter,
    lambda_smooth: float = 0.01,
    lambda_natural: float = 0.0,
) -> torch.Tensor:
    """
    Regularization on the learned geometry.

    Two components:
    1. Smoothness: penalty on adjacent-bin distance variation (lambda_smooth).
       Centering and scale normalization are handled inside BinEmbedding.forward().
    2. Proximity to natural (lambda_natural): L2 distance from the normalized
       natural spacing φ_natural(b) = (b - mean) / rms.
       Acts as a prior anchoring the geometry to the natural ordering.
    """
    reg = lambda_smooth * model.smoothness_penalty()

    if lambda_natural > 0.0:
        K = model.K
        phi = model.embedding.forward()                    # (K, d)
        # Natural spacing, normalized the same way BinEmbedding normalizes
        nat = torch.linspace(0.0, 1.0, K, device=phi.device)
        nat = nat - nat.mean()
        nat = nat / nat.pow(2).mean().sqrt().clamp(min=1e-6)
        nat = nat.unsqueeze(1).expand_as(phi)             # (K, d)
        reg = reg + lambda_natural * ((phi - nat) ** 2).mean()

    return reg


# ────────────────────────────────────────────────────────────────────────────
# Simpler architecture: Learned Quantile Level (Q-Level) Model
# ────────────────────────────────────────────────────────────────────────────

class QuantileLevelBarycenter(nn.Module):
    """
    Simplest learned generalization of the CDF-median (BC).

    Instead of always aggregating at the median quantile (q=0.5), this model
    learns the optimal quantile level q_θ ∈ (0, 1), optionally conditioned
    on a scalar covariate (e.g., cross-sectional dispersion D_t).

    Architecture
    ------------
    q_θ(d) = σ(α + β · d)

    Parameters: α (intercept), β (dispersion slope).
    With β=0, α=0: q=0.5 exactly → recovers BC.

    The aggregated CDF at each level k is the soft q_θ-th quantile of
    {F^(i)(k)}_i, computed via the pinball loss gradient:

        Ĝ_θ(k) = Σ_i w_i · F^(i)(k)  where w_i ∝ softmax(s_θ(F^(i)(k)))
        and  s_θ(u) = q_θ · I[u > median] - (1-q_θ) · I[u ≤ median]

    In practice we use the soft-sort / relaxed order-statistics approach:
    the q-th quantile is approximated as a soft linear combination of the
    sorted values, differentiable everywhere.

    Relation to OT
    --------------
    The q=0.5 case is the W₁ barycenter (CDF-median, Proposition 2.2).
    For q ≠ 0.5, this model selects a different central tendency in the
    Wasserstein sense — q < 0.5 favors lower-inflation forecasters,
    q > 0.5 favors higher-inflation forecasters.

    Parameters
    ----------
    use_dispersion : if True, learn β ≠ 0 (covariate-conditioned quantile)
    """

    def __init__(self, use_dispersion: bool = True):
        super().__init__()
        # logit(0.5) = 0 → initialize to median
        self.alpha = nn.Parameter(torch.tensor(0.0))
        if use_dispersion:
            self.beta = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("beta", torch.tensor(0.0))
        self.use_dispersion = use_dispersion

    def quantile_level(self, dispersion: float = 0.0) -> torch.Tensor:
        """Returns q_θ ∈ (0, 1)."""
        return torch.sigmoid(self.alpha + self.beta * dispersion)

    def forward(
        self,
        panel: torch.Tensor,      # (n, K) PMF panel
        dispersion: float = 0.0,  # scalar covariate (e.g., cross-sectional std)
    ) -> torch.Tensor:
        """
        Returns aggregate PMF (K,) via soft-quantile aggregation of CDFs.
        """
        n, K = panel.shape
        q = self.quantile_level(dispersion)   # scalar ∈ (0, 1)

        # Convert panel PMFs to CDFs (K-1 values each)
        cdfs = torch.cumsum(panel, dim=1)[:, :-1]   # (n, K-1)

        # Soft q-th quantile of each CDF level via differentiable sort
        # We compute the soft q-th order statistic:
        #   Q_q({x_i}) ≈ Σ_i w_i(q) · x_(i:n)
        # where w_i(q) = weight of the i-th order statistic at quantile q.
        # Using temperature-scaled softmax over sorted positions:

        agg_cdf = torch.zeros(K - 1, device=panel.device)
        for k in range(K - 1):
            x = cdfs[:, k]             # (n,) CDF values at level k
            # Sort ascending
            sorted_x, _ = torch.sort(x)
            # Index positions for soft quantile: position q*(n-1)
            # Use linear interpolation weights
            pos = q * (n - 1)          # float position in [0, n-1]
            lo = torch.floor(pos).long().clamp(0, n - 2)
            hi = lo + 1
            frac = pos - lo.float()
            agg_cdf[k] = (1.0 - frac) * sorted_x[lo] + frac * sorted_x[hi]

        # Ensure agg_cdf is non-decreasing (should be by construction at each level,
        # but small floating-point errors can violate this)
        agg_cdf = torch.cummax(agg_cdf, dim=0).values
        agg_cdf = agg_cdf.clamp(0.0, 1.0)

        # CDF → PMF via differencing
        pmf = torch.diff(agg_cdf, prepend=torch.zeros(1, device=panel.device),
                         append=torch.ones(1, device=panel.device))
        pmf = pmf.clamp(min=1e-10)
        pmf = pmf / pmf.sum()
        return pmf


def predict_qlevel_numpy(
    model: QuantileLevelBarycenter,
    panel_np: np.ndarray,
    dispersion: float = 0.0,
    device: str = "cpu",
) -> np.ndarray:
    """Numpy convenience wrapper for QuantileLevelBarycenter."""
    model.eval()
    with torch.no_grad():
        panel_t = torch.from_numpy(panel_np.astype(np.float32)).to(device)
        pmf_t = model(panel_t, dispersion=dispersion)
    return pmf_t.cpu().numpy()


# ────────────────────────────────────────────────────────────────────────────
# Numpy interface (for inference after training)
# ────────────────────────────────────────────────────────────────────────────

def predict_numpy(
    model: LearnedWassersteinBarycenter,
    panel_np: np.ndarray,         # (n, K) float32
    device: str = "cpu",
) -> np.ndarray:
    """
    Convenience wrapper: numpy panel → numpy aggregate PMF.
    """
    model.eval()
    with torch.no_grad():
        panel_t = torch.from_numpy(panel_np.astype(np.float32)).to(device)
        pmf_t = model(panel_t)
    return pmf_t.cpu().numpy()


def get_cost_matrix_numpy(
    model: LearnedWassersteinBarycenter,
    device: str = "cpu",
) -> np.ndarray:
    """
    Returns the learned cost matrix C_θ as a (K, K) numpy array.
    """
    model.eval()
    with torch.no_grad():
        C = model.embedding.cost_matrix()
    return C.cpu().numpy()


def get_embedding_numpy(
    model: LearnedWassersteinBarycenter,
    device: str = "cpu",
) -> np.ndarray:
    """
    Returns φ_θ as a (K, d) numpy array.
    """
    model.eval()
    with torch.no_grad():
        phi = model.embedding.forward()
    return phi.cpu().numpy()
