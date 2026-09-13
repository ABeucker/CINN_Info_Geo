"""
Estimate the Fisher information matrix of a simulated forward model using a
conditional invertible neural network (cINN) that approximates the
observation likelihood p(x | z).

Pipeline:
    1. Generate synthetic (z, x) pairs from a pluggable prior and forward
       model (data.py: SimulationData, priors, forward models).
    2. Train a cINN approximating p(x | z) (train.Trainer), or load a
       previously trained checkpoint -- see `training.train_model` in
       config.yaml.
    3. Compare the Fisher information estimated from the flow's score
       function (eval.FisherEstimator) against the closed-form analytic
       Fisher information for the chosen forward model, over a z1-z2 grid
       (data-driven by default, or fixed via `evaluation.grid_range`). Both
       the full per-point matrix and its Frobenius norm are saved for each.

"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from cinn import ConditionalInvertibleBlock
from config import ExperimentConfig, load_config
from data import SimulationData, build_forward_model, build_prior
from eval import FisherEstimator
from train import Trainer

logger = logging.getLogger(__name__)

_CHECKPOINT_RE = re.compile(r"cinn_weights_ep(\d+)\.pt$")


def _find_last_checkpoint(output_dir: Path) -> Optional[Path]:
    """
    The highest-numbered checkpoint saved during training, if any. Since
    Trainer only ever writes a checkpoint on a validation-loss improvement,
    this is also the best-performing one from the run, not merely the most
    recently written. Returns None if training never improved past its
    warmup period (see `training.warmup_epochs`).
    """
    candidates = []
    for p in output_dir.glob("cinn_weights_ep*.pt"):
        m = _CHECKPOINT_RE.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    return max(candidates)[1] if candidates else None


def _epoch_from_path(path: str) -> Optional[int]:
    """Extract the epoch number from a `cinn_weights_ep<N>.pt`-style path, if it matches; else None."""
    m = _CHECKPOINT_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


def _fisher_grid_name(kind: str, epoch: Optional[int]) -> str:
    """
    `kind` is 'matrix' or 'norm'. Epoch-tagged (`estimated_fisher_matrix_
    grid_ep<N>.npy`) whenever the evaluated checkpoint's epoch is known --
    the same naming convention evaluate.py's cache looks for, so a
    checkpoint you copy down locally and re-evaluate there is recognized
    as already computed instead of silently redone under a different name.
    Falls back to the plain (untagged) name only when no specific epoch
    applies (e.g. no checkpoint was ever saved during training).
    """
    suffix = f"_ep{epoch:04d}" if epoch is not None else ""
    return f"estimated_fisher_{kind}_grid{suffix}.npy"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """--config, shared by main.py's and main_sweep.py's CLIs."""
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML config file. Default: config.yaml.",
    )


def _format_config(cfg: ExperimentConfig) -> str:
    """One line per top-level section -- readable without needing a full recursive pretty-printer."""
    return "\n".join([
        f"seed={cfg.seed}",
        f"output_dir={cfg.output_dir}",
        f"model_location={cfg.model_location}",
        f"data={cfg.data}",
        f"model={cfg.model}",
        f"training={cfg.training}",
        f"evaluation={cfg.evaluation}",
    ])


def _to_yaml_safe(obj):
    """Recursively convert tuples to lists so yaml.safe_dump can represent the config (e.g. grid_range)."""
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_safe(v) for v in obj]
    return obj


def _save_run_config(cfg: ExperimentConfig, output_dir: Path) -> None:
    """
    Write the fully-resolved config to `output_dir/run_config.yaml`, purely
    as a record of exactly what settings produced this run's outputs (for
    your own future reference -- nothing else in this project reads it).
    """
    raw = _to_yaml_safe(dataclasses.asdict(cfg))
    with open(output_dir / "run_config.yaml", "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


def build_model(cfg: ExperimentConfig, data: SimulationData) -> ConditionalInvertibleBlock:
    """
    Construct the cINN: freshly initialized if we're about to train it,
    otherwise loaded from disk. `n_dim`/`cond_dims` come from the data
    itself, not config, so they always match the chosen prior/forward model.
    """
    params = cfg.model_params(load=not cfg.training.train_model, n_dim=data.x_dim, cond_dims=data.z_dim)
    return ConditionalInvertibleBlock(params)


def main(config_path: str = "config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_config(config_path)
    logger.info("Using config (%s):\n%s", config_path, _format_config(cfg))

    output_dir = cfg.resolved_output_dir()
    _run(cfg, output_dir)


def _run(cfg: ExperimentConfig, output_dir: Path) -> None:
    def save(name: str, array: np.ndarray) -> None:
        np.save(output_dir / name, array)

    _save_run_config(cfg, output_dir)

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

    # Shared by both the analytic and the flow-estimated grid below, so they
    # cover the same z1-z2 region and are directly comparable. Data-driven by
    # default; fixed to evaluation.grid_range (same range on both axes) if set.
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
        trainer.train()  # mutates `model` in place; also writes checkpoints to output_dir

        checkpoint_path = _find_last_checkpoint(output_dir)
        if checkpoint_path is None:
            logger.warning(
                "No checkpoint was saved during training (epochs=%d, warmup_epochs=%d) -- "
                "evaluating the final in-training weights instead of a reloaded checkpoint.",
                cfg.training.epochs, cfg.training.warmup_epochs,
            )
            checkpoint_epoch = None
        else:
            checkpoint_epoch = _epoch_from_path(str(checkpoint_path))
            logger.info("Evaluating last saved checkpoint: %s", checkpoint_path.name)
            params = cfg.model_params(load=True, n_dim=data.x_dim, cond_dims=data.z_dim)
            params["model_location"] = str(checkpoint_path)
            model = ConditionalInvertibleBlock(params)
    else:
        logger.info("Loaded pre-trained cINN from %s", cfg.model_location)
        checkpoint_epoch = _epoch_from_path(cfg.model_location)

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
    save(_fisher_grid_name("matrix", checkpoint_epoch), estimated_matrix_grid)
    save(_fisher_grid_name("norm", checkpoint_epoch), estimated_norm_grid)

    logger.info("Done. Outputs written to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    main(args.config)