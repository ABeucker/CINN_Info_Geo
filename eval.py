"""
Estimate the Fisher information matrix of a trained conditional density
model via its score function.

The flow models the observation likelihood p(x | z), so we call
`model.log_prob(x, z)` and differentiate with respect to z (the conditioning
variable) to get the score grad_z log p(x | z). 
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SimulationData, map_over_grid

logger = logging.getLogger(__name__)


class FisherEstimator:
    """
    Estimates the Fisher information matrix

        F(z) = E_{x ~ p(x|z)} [ grad_z log p(x|z) grad_z log p(x|z)^T ]

    by averaging the outer product of the score function over samples drawn
    from a trained conditional density model.
    """

    def __init__(self, model, device: torch.device, dataloader: Optional[DataLoader] = None):
        self.model = model.to(device)
        self.device = device
        self.data_loader = dataloader

        self.model.flow.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def estimate_fisher_information(self, dataloader: Optional[DataLoader] = None) -> torch.Tensor:
        """
        Estimate F(z) by averaging the score outer product over all (z, x)
        pairs in `dataloader` (or the estimator's default dataloader if none
        is given).
        """
        loader = dataloader if dataloader is not None else self.data_loader
        if loader is None:
            raise ValueError("No dataloader provided or set on the estimator.")

        fisher = torch.zeros(self.model.cond_dims, self.model.cond_dims, device=self.device)
        n_samples = 0

        with torch.enable_grad():
            for z, x in loader:
                x = x.to(self.device)
                z = z.to(self.device).requires_grad_(True)

                log_p = self.model.log_prob(x, z)
                score, = torch.autograd.grad(log_p.sum(), z)

                fisher += score.T @ score
                n_samples += x.shape[0]

        fisher /= n_samples
        logger.debug("Estimated Fisher information from %d samples:\n%s", n_samples, fisher)
        return fisher

    def estimate_grid(
        self,
        data: SimulationData,
        z1_range: np.ndarray,
        z2_range: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """
        Evaluate the flow-estimated F(z) over a z1 x z2 grid, returning the
        full matrix at every point: shape (len(z2_range), len(z1_range),
        cond_dims, cond_dims). For just the Frobenius norm at each point,
        use np.linalg.norm(grid, axis=(-2, -1)) on the result.
        """
        log_every = max(1, len(z1_range) * len(z2_range) // 20)

        def compute(z: np.ndarray) -> np.ndarray:
            loader = data.conditional_loader(z, n_samples=n_samples)
            return self.estimate_fisher_information(loader).detach().cpu().numpy()

        def progress(count: int, total: int) -> None:
            if count % log_every == 0:
                logger.info("Estimated Fisher information for %d/%d grid points", count, total)

        return map_over_grid(z1_range, z2_range, self.model.cond_dims, compute, on_point=progress)