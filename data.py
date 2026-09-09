"""
Everything needed to generate synthetic (z, x) data and its analytic Fisher
information: priors over z, forward models z -> x, and
SimulationData class which ties a chosen prior + forward model together.

x = mean(z) + noise, noise ~ N(0, sigma), where "mean" comes from a
ForwardModel and z is drawn from a Prior.

"z" denotes the simulator parameter (the flow's conditioning variable); 
"x" denotes the noisy observation (the flow's modeled variable).

This "z" is unrelated to the flow's own internal Gaussian latent variable, which
cinn.py also calls "z".

To add a new prior or forward model, subclass `Prior` / `ForwardModel` below.
PRIOR_REGISTRY / FORWARD_MODEL_REGISTRY (name -> class) exist so that they can be instantiated from the yaml file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def build_covariance(diag: Optional[list] = None, matrix: Optional[list] = None) -> np.ndarray:
    """
    Build a covariance matrix from either its diagonal entries (independent
    dimensions) or an explicit full matrix (to introduce correlations
    between dimensions). If both are given, `matrix` takes precedence.
    Used by GaussianPrior (std -> diag, or cov -> matrix directly).
    """
    if matrix is not None:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"matrix must be a square 2D covariance matrix, got shape {matrix.shape}")
        return matrix
    if diag is not None:
        return np.diag(np.asarray(diag, dtype=np.float32)).astype(np.float32)
    raise ValueError("Provide either `diag` or `matrix`.")

# ============================================================
# Priors: distributions over z
# ============================================================


class Prior(ABC):
    """A distribution over the simulator parameter z, used to generate training data."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of z."""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n` samples of z, shape (n, dim)."""

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({attrs})"


class GaussianPrior(Prior):
    """
    z ~ Normal(mu, cov). Provide either `std` (independent per dimension --
    the common case) or a full `cov` covariance matrix directly (to
    introduce correlations between z dimensions). If both are given, `cov`
    takes precedence.
    """

    def __init__(
        self,
        mu: list[float],
        std: Optional[list[float]] = None,
        cov: Optional[list[list[float]]] = None,
    ):
        self.mu = np.asarray(mu, dtype=np.float32)
        diag = [s ** 2 for s in std] if std is not None else None
        self.cov = build_covariance(diag=diag, matrix=cov)
        if self.cov.shape != (len(self.mu), len(self.mu)):
            raise ValueError(f"cov must have shape ({len(self.mu)}, {len(self.mu)}), got {self.cov.shape}")

    @property
    def dim(self) -> int:
        return len(self.mu)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.multivariate_normal(mean=self.mu, cov=self.cov, size=n).astype(np.float32)


class UniformPrior(Prior):
    """z_i ~ Uniform(low[i], high[i]), independent per dimension."""

    def __init__(self, low: list[float], high: list[float]):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        if self.low.shape != self.high.shape:
            raise ValueError(f"low and high must have the same shape, got {self.low.shape} and {self.high.shape}")

    @property
    def dim(self) -> int:
        return len(self.low)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=(n, self.dim)).astype(np.float32)


PRIOR_REGISTRY: dict[str, type[Prior]] = {
    "gaussian": GaussianPrior,
    "uniform": UniformPrior,
}


def build_prior(cfg: dict) -> Prior:
    """Instantiate a Prior from a config dict, e.g. {"type": "gaussian", "mu": [...], "std": [...]}."""
    cfg = dict(cfg)
    prior_type = cfg.pop("type", None)
    cls = PRIOR_REGISTRY.get(prior_type)
    if cls is None:
        raise ValueError(f"Unknown prior type '{prior_type}'. Available: {sorted(PRIOR_REGISTRY)}")
    return cls(**cfg)


# ============================================================
# Forward models: z -> x
# ============================================================


class ForwardModel(ABC):
    """A deterministic simulator mean function x = mean(z)."""

    @abstractmethod
    def mean(self, z: np.ndarray) -> np.ndarray:
        """Evaluate the forward model. Accepts a single z (shape (dim,)) or a batch (shape (n, dim))."""

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """
        Analytic Jacobian d(mean)/dz at a single z (shape (dim,)), used for
        the closed-form Fisher information. Override for an exact result;
        the default falls back to a central-difference numeric approximation.
        """
        return self._numeric_jacobian(z)

    def _numeric_jacobian(self, z: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        x0 = self.mean(z)
        jac = np.zeros((len(x0), len(z)))
        for i in range(len(z)):
            step = np.zeros_like(z)
            step[i] = eps
            jac[:, i] = (self.mean(z + step) - self.mean(z - step)) / (2 * eps)
        return jac

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({attrs})"

class LinearForwardModel(ForwardModel):
    """
    A simple affine (constant-Jacobian) transform:
 
        x1 = z1
        x2 = z2 + coupling * z1
    """
 
    def __init__(self, coupling: float = 0.7):
        self.coupling = coupling
 
    def mean(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        z1, z2 = z[..., 0], z[..., 1]
        x1 = z1
        x2 = z2 + self.coupling * z1
        return np.stack([x1, x2], axis=-1).astype(np.float32)
 
    def jacobian(self, z: np.ndarray) -> np.ndarray:
        return np.array([
            [1.0, 0.0],
            [self.coupling, 1.0],
        ])

class PolynomialForwardModel(ForwardModel):
    """
    The original toy forward model: mean(z) = [3*z1^3, 0.5*z2 - 1]. Has an
    exact analytic Jacobian.
    """

    def mean(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        z1, z2 = z[..., 0], z[..., 1]
        x1 = 3.0 * z1**3
        x2 = 0.5 * z2 - 1.0
        return np.stack([x1, x2], axis=-1).astype(np.float32)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        return np.array([
            [9.0 * z[0] ** 2, 0.0],
            [0.0, 0.5],
        ])

class CoupledPolynomialForwardModel(ForwardModel):
    """
    A simple coupled polynomial transform:

        x1 = z1 + 0.8 * z2^2
        x2 = z2 + 0.5 * z1 * z2
    """
    def mean(self, z):
        z = np.asarray(z)
        z1, z2 = z[..., 0], z[..., 1]
        x1 = z1 + 0.8 * z2**2
        x2 = z2 + 0.5 * z1 * z2
        return np.stack([x1, x2], axis=-1).astype(np.float32)

    def jacobian(self, z):
        z = np.asarray(z)
        z1, z2 = z
        return np.array([
            [1.0,       1.6 * z2],
            [0.5 * z2,  1.0 + 0.5 * z1],
        ], dtype=np.float32)

class SpiralForwardModel(ForwardModel):
    """
    Classic two-armed spiral transform, a common toy dataset for flows and
    density estimators. Interprets z = (radius, angle) and maps it to
    Cartesian coordinates:

        x1 = radius * cos(turns * angle)
        x2 = radius * sin(turns * angle)

    `turns` controls how many full rotations the spiral makes over the angle
    range covered by the prior. 
    
    No closed-form Jacobian is implemented, the base class's numeric 
    Jacobian is used for the analytic Fisher information comparison.
    """

    def __init__(self, turns: float = 1.5):
        self.turns = turns

    def mean(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        radius, angle = z[..., 0], z[..., 1]
        phi = self.turns * angle
        x1 = radius * np.cos(phi)
        x2 = radius * np.sin(phi)
        return np.stack([x1, x2], axis=-1).astype(np.float32)


class BananaForwardModel(ForwardModel):
    """
    The classic "banana" (twisted Gaussian) transform:

        x1 = z1
        x2 = z2 + curvature * z1^2
    """

    def __init__(self, curvature: float = 0.5):
        self.curvature = curvature

    def mean(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        z1, z2 = z[..., 0], z[..., 1]
        x1 = z1
        x2 = z2 + self.curvature * z1**2
        return np.stack([x1, x2], axis=-1).astype(np.float32)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        return np.array([
            [1.0, 0.0],
            [2.0 * self.curvature * z[0], 1.0],
        ])


class TanhForwardModel(ForwardModel):
    """
    A saturating, per-dimension transform:

        x_i = scale * tanh(z_i / width)

    Has a diagonal Jacobian.
    """

    def __init__(self, scale: float = 1.0, width: float = 2.0):
        self.scale = scale
        self.width = width

    def mean(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        return (self.scale * np.tanh(z / self.width)).astype(np.float32)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        slope = (self.scale / self.width) * (1.0 - np.tanh(z / self.width) ** 2)
        return np.diag(slope)


FORWARD_MODEL_REGISTRY: dict[str, type[ForwardModel]] = {
    "linear": LinearForwardModel,
    "polynomial": PolynomialForwardModel,
    "spiral": SpiralForwardModel,
    "banana": BananaForwardModel,
    "coupled_polynomial": CoupledPolynomialForwardModel,
    "tanh": TanhForwardModel,
}


def build_forward_model(cfg: dict) -> ForwardModel:
    """Instantiate a ForwardModel from a config dict, e.g. {"type": "spiral", "turns": 1.5}."""
    cfg = dict(cfg)
    model_type = cfg.pop("type", None)
    cls = FORWARD_MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown forward model type '{model_type}'. Available: {sorted(FORWARD_MODEL_REGISTRY)}")
    return cls(**cfg)


# ============================================================
# Simulation data
# ============================================================


def map_over_grid(z1_range: np.ndarray, z2_range: np.ndarray, dim: int, fn, on_point=None) -> np.ndarray:
    """
    Evaluate fn(z) -> (dim, dim) matrix at every point of a z1 x z2 grid,
    returning the full stack: shape (len(z2_range), len(z1_range), dim, dim).
    If given, on_point(count, total) is called after each point (e.g. for
    progress logging when fn is expensive). Shared by
    SimulationData.analytic_fisher_grid and eval.FisherEstimator.estimate_grid.
    """
    z1_grid, z2_grid = np.meshgrid(z1_range, z2_range)
    matrix_grid = np.zeros(z1_grid.shape + (dim, dim), dtype=np.float32)
    total = z1_grid.size
    for count, (i, j) in enumerate(np.ndindex(z1_grid.shape), start=1):
        z = np.array([z1_grid[i, j], z2_grid[i, j]], dtype=np.float32)
        matrix_grid[i, j] = fn(z)
        if on_point is not None:
            on_point(count, total)
    return matrix_grid


class SimulationData:
    """
    Generates synthetic (z, x) pairs from a Prior and a ForwardModel, and
    provides train/val/test DataLoaders, single-z conditional DataLoaders
    (for Fisher information estimation), the closed-form Fisher information
    for the chosen forward model, and z-range-aware evaluation grid axes.

    Every DataLoader produced by this class yields (z, x) tuples: z (the
    conditioning variable) first, x (the flow's modeled variable) second --
    matching `model.log_prob(x, z)` / `model.batch_loss(x, z)`.
    """

    def __init__(
        self,
        prior: Prior,
        forward_model: ForwardModel,
        sigma: np.ndarray,
        n_samples: int,
        batch_size: int,
        rng: np.random.Generator,
    ):
        self.prior = prior
        self.forward_model = forward_model
        self.sigma = sigma
        self.batch_size = batch_size
        self.rng = rng

        self.z = self.prior.sample(n_samples, rng)
        self.x = self._sample_observations(self.z)

    @property
    def z_dim(self) -> int:
        return self.prior.dim

    @property
    def x_dim(self) -> int:
        return self.x.shape[-1]

    # ---- analytic Fisher information ----

    def analytic_fisher_information(self, z: np.ndarray) -> np.ndarray:
        """F(z) = J(z)^T sigma^{-1} J(z), where J is the forward model's Jacobian at z."""
        jacobian = self.forward_model.jacobian(z)
        return jacobian.T @ np.linalg.inv(self.sigma) @ jacobian

    def z_grid_axes(self, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Build z1/z2 grid axes spanning the observed range of the generated z
        data (assumes a 2D z). Used for both the analytic and the
        flow-estimated Fisher information grids, so the two are always
        evaluated over the same region.
        """
        z1_range = np.linspace(self.z[:, 0].min(), self.z[:, 0].max(), grid_size)
        z2_range = np.linspace(self.z[:, 1].min(), self.z[:, 1].max(), grid_size)
        return z1_range, z2_range

    def analytic_fisher_grid(self, z1_range: np.ndarray, z2_range: np.ndarray) -> np.ndarray:
        """
        Evaluate the analytic F(z) over a z1 x z2 grid (see z_grid_axes),
        returning the full matrix at every point: shape
        (len(z2_range), len(z1_range), z_dim, z_dim). For just the
        Frobenius norm at each point, use
        np.linalg.norm(grid, axis=(-2, -1)) on the result.
        """
        return map_over_grid(z1_range, z2_range, self.z_dim, self.analytic_fisher_information)

    # ---- sampling ----

    def _sample_noise(self, n: int) -> np.ndarray:
        return self.rng.multivariate_normal(mean=np.zeros(self.sigma.shape[0]), cov=self.sigma, size=n).astype(np.float32)

    def _sample_observations(self, z: np.ndarray) -> np.ndarray:
        """Draw x = mean(z) + noise, noise ~ N(0, sigma)."""
        mu = self.forward_model.mean(z).astype(np.float32)
        return (mu + self._sample_noise(len(z))).astype(np.float32)

    # ---- dataloaders ----

    def dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Split (z, x) pairs into train/val/test DataLoaders (70/20/10)."""
        dataset = TensorDataset(torch.from_numpy(self.z), torch.from_numpy(self.x))

        n_train = int(0.7 * len(dataset))
        n_val = int(0.2 * len(dataset))
        n_test = len(dataset) - n_train - n_val
        train_data, val_data, test_data = random_split(dataset, [n_train, n_val, n_test])

        return (
            DataLoader(train_data, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_data, batch_size=self.batch_size, shuffle=False),
            DataLoader(test_data, batch_size=self.batch_size, shuffle=False),
        )

    def conditional_loader(self, z: np.ndarray, n_samples: int) -> DataLoader:
        """Draw `n_samples` observations x ~ p(x | z) for one fixed z."""
        mu = self.forward_model.mean(z).astype(np.float32)
        x_samples = (mu + self._sample_noise(n_samples)).astype(np.float32)
        z_tiled = np.tile(z, (n_samples, 1)).astype(np.float32)

        dataset = TensorDataset(torch.from_numpy(z_tiled), torch.from_numpy(x_samples))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)