"""
Estimate the Fisher information matrix of a simulated forward model using a
conditional invertible neural network (cINN) that approximates the
observation likelihood p(x | z).

Pipeline:
    1. Generate synthetic (z, x) pairs from a prior and forward
       model (data.py: SimulationData, priors, forward models).
    2. Train a cINN approximating p(x | z) (train.Trainer), or load a
       previously trained checkpoint, see ReadME.md and yaml config file.
    3. Compare the Fisher information estimated from the flow's score
       function (eval.FisherEstimator) against the analytic
       Fisher information for the chosen forward model over a grid. Both
       the full per-point matrix and its Frobenius norm are saved.

All experiment settings live in config.yaml.
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cinn import ConditionalInvertibleBlock
from config import ExperimentConfig, load_config
from data import SimulationData, build_forward_model, build_prior
from eval import FisherEstimator
from train import Trainer

logger = logging.getLogger(__name__)


def _acquire_run_lock(output_dir: Path) -> bool:
    """
    Had a problem with two SLURM tasks launched for one job. This prevents this. 

    However was fixed with explicitly setting #SBATCH --ntasks=1 in SLURM script.
    """
    lock_path = output_dir / ".run.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


@contextmanager
def run_lock(output_dir: Path, enabled: bool):
    """
    Had a problem with two SLURM tasks launched for one job. This prevents this. 

    However was fixed with explicitly setting #SBATCH --ntasks=1 in SLURM script.
    """
    if not enabled or _acquire_run_lock(output_dir):
        try:
            yield True
        finally:
            if enabled:
                (output_dir / ".run.lock").unlink(missing_ok=True)
    else:
        yield False


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """--config and --no-lock, shared by main.py's and main_sweep.py's CLIs."""
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML config file. Default: config.yaml.",
    )
    parser.add_argument("--no-lock", action="store_true", help="Skip the run lock (useful for local testing).")


def _format_config(cfg: ExperimentConfig) -> str:
    """Read config and return a human-readable string for logging."""
    return "\n".join([
        f"seed={cfg.seed}",
        f"output_dir={cfg.output_dir}",
        f"model_location={cfg.model_location}",
        f"use_run_lock={cfg.use_run_lock}",
        f"data={cfg.data}",
        f"model={cfg.model}",
        f"training={cfg.training}",
        f"evaluation={cfg.evaluation}",
    ])


def build_model(cfg: ExperimentConfig, data: SimulationData) -> ConditionalInvertibleBlock:
    """
    Construct the cINN: initialized if about to be trained,
    otherwise loaded from desired location. `n_dim`/`cond_dims` come from the data
    itself, not config, so they always match the chosen prior/forward model.
    """
    params = cfg.model_params(load=not cfg.training.train_model, n_dim=data.x_dim, cond_dims=data.z_dim)
    return ConditionalInvertibleBlock(params)


def main(config_path: str = "config.yaml", use_run_lock: Optional[bool] = None) -> None:
    """`use_run_lock`, if given, overrides the config's `use_run_lock` (e.g. from --no-lock)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_config(config_path)
    logger.info("Using config (%s):\n%s", config_path, _format_config(cfg))

    output_dir = cfg.resolved_output_dir()
    lock_enabled = cfg.use_run_lock if use_run_lock is None else use_run_lock

    with run_lock(output_dir, lock_enabled) as acquired:
        if not acquired:
            logger.warning(
                "Another process already holds the run lock for %s -- skipping this run to avoid "
                "duplicate work (e.g. two tasks launched for one job). If a previous run crashed "
                "without cleaning up, delete %s and re-run.",
                output_dir, output_dir / ".run.lock",
            )
            return
        _run(cfg, output_dir)


def _run(cfg: ExperimentConfig, output_dir: Path) -> None:
    def save(name: str, array: np.ndarray) -> None:
        np.save(output_dir / name, array)

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    logger.info("Generating synthetic data (N=%d)", cfg.data.n_samples)
    prior = build_prior(cfg.data.prior)
    forward_model = build_forward_model(cfg.data.forward_model)
    data = SimulationData(
        prior=prior,
        forward_model=forward_model,
        sigma=cfg.data.sigma,
        n_samples=cfg.data.n_samples,
        batch_size=cfg.data.batch_size,
        rng=rng,
    )
    save("z.npy", data.z)
    save("x.npy", data.x)

    train_loader, val_loader, test_loader = data.dataloaders()

    if cfg.evaluation.grid_range is not None:
        axis = np.linspace(*cfg.evaluation.grid_range, cfg.evaluation.grid_size)
        z1_range, z2_range = axis, axis
    else:
        z1_range, z2_range = data.z_grid_axes(cfg.evaluation.grid_size)
    save("fisher_grid_z1_axis.npy", z1_range)
    save("fisher_grid_z2_axis.npy", z2_range)

    logger.info(
        "Computing analytic Fisher information over a %dx%d grid",
        cfg.evaluation.grid_size, cfg.evaluation.grid_size,
    )
    analytic_matrix_grid = data.analytic_fisher_grid(z1_range, z2_range)
    analytic_norm_grid = np.linalg.norm(analytic_matrix_grid, axis=(-2, -1))
    save("analytic_fisher_matrix_grid.npy", analytic_matrix_grid)
    save("analytic_fisher_norm_grid.npy", analytic_norm_grid)

    model = build_model(cfg, data)

    if cfg.training.train_model:
        logger.info("Training cINN for %d epochs", cfg.training.epochs)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.training.epochs,
            lr=cfg.training.lr,
            checkpoint_dir=output_dir,
            weight_decay=cfg.training.weight_decay,
            grad_clip=cfg.training.grad_clip,
            warmup_epochs=cfg.training.warmup_epochs,
            optimizer=cfg.training.optimizer,
            optimizer_kwargs=cfg.training.optimizer_kwargs,
            scheduler=cfg.training.scheduler,
            scheduler_kwargs=cfg.training.scheduler_kwargs,
            tensorboard=cfg.training.tensorboard,
            lambda_score=cfg.training.lambda_score,
        )
        model = trainer.train()
    else:
        logger.info("Loaded pre-trained cINN from %s", cfg.model_location)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimator = FisherEstimator(model=model, device=device, dataloader=test_loader)

    logger.info(
        "Estimating Fisher information from the flow over a %dx%d grid (%d samples/point)",
        cfg.evaluation.grid_size, cfg.evaluation.grid_size, cfg.evaluation.fisher_samples_per_point,
    )
    estimated_matrix_grid = estimator.estimate_grid(
        data, z1_range, z2_range, n_samples=cfg.evaluation.fisher_samples_per_point
    )
    estimated_norm_grid = np.linalg.norm(estimated_matrix_grid, axis=(-2, -1))
    save("estimated_fisher_matrix_grid.npy", estimated_matrix_grid)
    save("estimated_fisher_norm_grid.npy", estimated_norm_grid)

    logger.info("Done. Outputs written to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    main(args.config, use_run_lock=(False if args.no_lock else None))