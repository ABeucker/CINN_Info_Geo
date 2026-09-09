"""
Trainer for the conditional invertible neural network (cINN).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    """Maximum-likelihood trainer for a conditional normalizing flow."""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        lr: float = 1e-3,
        checkpoint_dir: Union[str, Path] = "../checkpoints",
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = 5.0,
        warmup_epochs: int = 0,
        optimizer: str = "Adam",
        optimizer_kwargs: Optional[dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[dict] = None,
        tensorboard: bool = True,
        lambda_score: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Checkpoints are written to `checkpoint_dir/cinn_weights_ep<epoch>.pt`
        (not overwritten) whenever validation loss improves, after the first
        `warmup_epochs` epochs.

        `optimizer` is any class name in `torch.optim` (e.g. "Adam", "AdamW",
        "SGD", "RMSprop"). `scheduler` is any class name in
        `torch.optim.lr_scheduler` (e.g. "ReduceLROnPlateau", "StepLR"), or
        None for no scheduler. Both are resolved via getattr, so this works
        for any standard first-order optimizer/scheduler out of the box;
        exotic ones with a different calling convention (e.g. LBFGS, which
        needs a closure) are not supported by this training loop.

        If `tensorboard` is True (default), per-epoch loss/LR/timing are
        logged to `checkpoint_dir/tensorboard` via
        torch.utils.tensorboard.SummaryWriter.

        If `lambda_score` is set, the training loss becomes
        L = L_NLL + lambda_score * mean(||grad_z log p(x|z)||^2). 
        Adds a second backward pass per batch. 
        Validation loss stays plain NLL for comparisons across runs.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.lambda_score = lambda_score

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        optimizer_cls = getattr(torch.optim, optimizer)
        self.optimizer = optimizer_cls(
            self.model.parameters(), lr=lr, weight_decay=weight_decay, **(optimizer_kwargs or {})
        )

        self.scheduler = None
        if scheduler is not None:
            scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler)
            self.scheduler = scheduler_cls(self.optimizer, **(scheduler_kwargs or {}))

        self.train_history: list[float] = []
        self.val_history: list[float] = []

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.checkpoint_dir / "tensorboard"))

    def train(self):
        """Run the full training loop, checkpointing on validation-loss improvements after warmup."""
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            epoch_start = time.perf_counter()
            train_loss, compute_time = self.train_epoch()
            val_loss = self.val_epoch()
            epoch_time = time.perf_counter() - epoch_start

            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            self._step_scheduler(val_loss)

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %04d/%04d | Train Loss %.6f | Val Loss %.6f | LR %.2e | Time %.3fs",
                epoch + 1, self.epochs, train_loss, val_loss, lr, epoch_time,
            )
            if self.writer is not None:
                self.writer.add_scalar("loss/train", train_loss, epoch)
                self.writer.add_scalar("loss/val", val_loss, epoch)
                self.writer.add_scalar("lr", lr, epoch)
                self.writer.add_scalar("time/epoch_seconds", epoch_time, epoch)
                self.writer.add_scalar("time/train_compute_seconds", compute_time, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if epoch >= self.warmup_epochs:
                    checkpoint_path = self.checkpoint_dir / f"cinn_weights_ep{epoch + 1:04d}.pt"
                    torch.save(self.model.state_dict(), checkpoint_path)
                    logger.info("Saved new best checkpoint to %s", checkpoint_path)

        if self.writer is not None:
            self.writer.close()
        self.save_history()
        return self.model

    def _step_scheduler(self, val_loss: float) -> None:
        """ReduceLROnPlateau needs the metric it's monitoring; other schedulers just need step()."""
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _batch_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Plain NLL, or NLL + lambda_score * mean(||grad_z log p(x|z)||^2) if lambda_score is set."""
        if self.lambda_score is None:
            return self.model.batch_loss(x, z)

        z = z.requires_grad_(True)
        log_p = self.model.log_prob(x, z)
        nll_loss = -log_p.mean() / self.model.params["n_dim"]

        score, = torch.autograd.grad(log_p.sum(), z, create_graph=True)
        score_penalty = score.pow(2).sum(dim=1).mean()

        return nll_loss + self.lambda_score * score_penalty

    def train_epoch(self) -> tuple[float, float]:
        """Run one training epoch. Returns (mean training loss, time spent computing -- forward/backward/step)."""
        self.model.flow.train()

        running_loss = 0.0
        samples_seen = 0
        compute_time = 0.0

        for z, x in self.train_loader:
            z = z.to(self.device)
            x = x.to(self.device)

            step_start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._batch_loss(x, z)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            compute_time += time.perf_counter() - step_start

            batch_size = z.shape[0]
            running_loss += loss.item() * batch_size
            samples_seen += batch_size

        return running_loss / samples_seen, compute_time

    def val_epoch(self) -> float:
        """Run one validation epoch and return the mean validation loss (always plain NLL)."""
        self.model.flow.eval()

        running_loss = 0.0
        samples_seen = 0

        with torch.no_grad():
            for z, x in self.val_loader:
                z = z.to(self.device)
                x = x.to(self.device)

                loss = self.model.batch_loss(x, z)

                batch_size = z.shape[0]
                running_loss += loss.item() * batch_size
                samples_seen += batch_size

        return running_loss / samples_seen

    def save_history(self) -> None:
        """Save training and validation loss history as .npy arrays."""
        np.save(self.checkpoint_dir / "train_loss_history.npy", np.array(self.train_history, dtype=np.float32))
        np.save(self.checkpoint_dir / "val_loss_history.npy", np.array(self.val_history, dtype=np.float32))