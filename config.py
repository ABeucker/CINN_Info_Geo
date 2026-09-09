"""
Experiment configuration, loaded from a YAML file (see config.yaml).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from data import build_covariance


@dataclass
class DataConfig:
    """Synthetic data generation settings."""

    n_samples: int = 100_000
    batch_size: int = 256

    sigma_diag: Optional[list[float]] = field(default_factory=lambda: [0.09, 0.25])
    sigma_matrix: Optional[list[list[float]]] = None

    prior: dict = field(default_factory=lambda: {
        "type": "gaussian", "mu": [0.0, 1.0], "std": [1.0, 2.0],
    })

    forward_model: dict = field(default_factory=lambda: {"type": "polynomial"})

    @property
    def sigma(self) -> np.ndarray:
        """Observation-noise covariance matrix."""
        return build_covariance(diag=self.sigma_diag, matrix=self.sigma_matrix)


@dataclass
class ModelConfig:
    """
    Architecture parameters for cinn.ConditionalInvertibleBlock.

    `n_dim` and `cond_dims` are not configured here, they're
    derived from the chosen forward model's output dimension
    and the chosen prior's dimension (see main.build_model), so swapping
    either in config.yaml can never leave a mismatched dimension.
    """

    n_blocks: int = 8
    n_nodes: int = 128
    subnet_depth: int = 2
    act: str = "relu"
    coupling: str = "affine"


@dataclass
class TrainingConfig:
    """Training settings."""

    train_model: bool = True  
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = 5.0

    warmup_epochs: int = 0

    optimizer: str = "Adam"
    optimizer_kwargs: dict = field(default_factory=dict)

    scheduler: Optional[str] = None
    scheduler_kwargs: dict = field(default_factory=dict)

    tensorboard: bool = True

    lambda_score: Optional[float] = None


@dataclass
class EvaluationConfig:
    """
    Fisher information grid evaluation settings.

    By default (grid_range: null), the grid's z1/z2 axes are derived from
    the actual range of the generated z data (see
    data.SimulationData.z_grid_axes), so the analytic and flow-estimated
    Fisher information are evaluated over the same, data-relevant region.
    Set grid_range to a fixed [min, max] to use that on both axes instead.
    """

    grid_size: int = 100
    grid_range: Optional[tuple[float, float]] = None
    fisher_samples_per_point: int = 1_000

    def __post_init__(self) -> None:
        if self.grid_range is not None:
            self.grid_range = tuple(self.grid_range)


@dataclass
class ExperimentConfig:
    """General experiment configuration."""

    seed: int = 42
    output_dir: Optional[str] = None
    model_location: str = "../outputs/21916/cinn_best.pt"

    # Guards against accidentally running the pipeline twice with one submission, since this was problem on HPC server. 
    # Was fixed with explicitly setting #SBATCH --ntasks=1 in SLURM script.
    use_run_lock: bool = False

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def resolved_output_dir(self) -> Path:
        """Resolve (and create) the output directory, falling back to a SLURM-job-id-based path."""
        path = Path(self.output_dir) if self.output_dir else Path(
            f"../outputs/{os.environ.get('SLURM_JOB_ID', 'local')}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def model_params(self, load: bool, n_dim: int, cond_dims: int) -> dict:
        """Build the flat dict expected by cinn.ConditionalInvertibleBlock."""
        return {
            "n_dim": n_dim,
            "n_blocks": self.model.n_blocks,
            "n_nodes": self.model.n_nodes,
            "subnet_depth": self.model.subnet_depth,
            "cond_dims": cond_dims,
            "act": self.model.act,
            "coupling": self.model.coupling,
            "load": load,
            "model_location": self.model_location,
        }


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file, falling back to defaults for any omitted keys."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return ExperimentConfig(
        seed=raw.get("seed", 42),
        output_dir=raw.get("output_dir"),
        model_location=raw.get("model_location", "../outputs/21916/cinn_best.pt"),
        use_run_lock=raw.get("use_run_lock", True),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )