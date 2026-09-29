"""Reusable final-paper neural-model training utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ArrayLike: TypeAlias = np.ndarray | torch.Tensor
ModelInputs: TypeAlias = ArrayLike | tuple[ArrayLike, ArrayLike]


@dataclass(frozen=True)
class TrainingProtocol:
    """Numerical settings for one authoritative final-paper training path."""

    batch_size: int = 32
    learning_rate: float = 1e-4
    huber_delta: float = 1.0
    max_epochs: int = 100
    patience: int = 10
    max_grad_norm: float = 1.0
    improvement_threshold: float = 0.0
    train_loader_seed: int | None = None
    zero_grad_before_forward: bool = True
    check_finite_batches: bool = False
    check_finite_gradients: bool = False


MULTISCALE_PROTOCOL = TrainingProtocol(
    improvement_threshold=1e-12,
    train_loader_seed=42,
    zero_grad_before_forward=False,
    check_finite_batches=True,
    check_finite_gradients=True,
)

NEURAL_BASELINE_PROTOCOL = TrainingProtocol()


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True)
class TrainingResult:
    """A best-checkpoint-restored model and concise training metadata."""

    model: nn.Module
    best_epoch: int
    epochs_trained: int
    best_validation_loss: float
    history: tuple[EpochMetrics, ...]


def set_seed(seed: int = 42) -> None:
    """Apply the seed settings shared by the final-paper neural runners."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(policy: str | torch.device = "auto") -> torch.device:
    """Resolve the explicit CPU policy or the baseline runners' auto policy."""
    if isinstance(policy, torch.device):
        return policy
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(policy)


def _as_tensor(values: ArrayLike) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)


def _input_tuple(inputs: ModelInputs) -> tuple[ArrayLike, ...]:
    return inputs if isinstance(inputs, tuple) else (inputs,)


def _tensor_dataset(inputs: ModelInputs, targets: ArrayLike) -> TensorDataset:
    input_tensors = tuple(_as_tensor(values) for values in _input_tuple(inputs))
    target_tensor = _as_tensor(targets)
    lengths = {len(target_tensor), *(len(values) for values in input_tensors)}
    if len(lengths) != 1:
        raise ValueError(f"Input and target lengths differ: {sorted(lengths)}")
    if not lengths or next(iter(lengths)) == 0:
        raise ValueError("Training and validation datasets must be non-empty.")
    return TensorDataset(*input_tensors, target_tensor)


def make_data_loaders(
    train_inputs: ModelInputs,
    train_targets: ArrayLike,
    validation_inputs: ModelInputs,
    validation_targets: ArrayLike,
    protocol: TrainingProtocol,
) -> tuple[DataLoader, DataLoader]:
    """Create shuffled Train and ordered Validation loaders."""
    train_dataset = _tensor_dataset(train_inputs, train_targets)
    validation_dataset = _tensor_dataset(validation_inputs, validation_targets)
    generator = None
    if protocol.train_loader_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(protocol.train_loader_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=protocol.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=protocol.batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader


def build_optimizer(
    model: nn.Module,
    protocol: TrainingProtocol,
) -> torch.optim.Adam:
    """Construct the final-paper Adam optimizer without a scheduler."""
    return torch.optim.Adam(model.parameters(), lr=protocol.learning_rate)


def build_loss(protocol: TrainingProtocol) -> nn.HuberLoss:
    return nn.HuberLoss(delta=protocol.huber_delta)


def _forward_batch(
    model: nn.Module,
    batch: list[torch.Tensor] | tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    *features, targets = batch
    features = [feature.to(device) for feature in features]
    targets = targets.to(device)
    return model(*features), targets


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    protocol: TrainingProtocol,
) -> float:
    """Run one sample-weighted training epoch."""
    model.train()
    loss_total = 0.0
    sample_count = 0
    for batch in loader:
        if protocol.zero_grad_before_forward:
            optimizer.zero_grad()
        prediction, targets = _forward_batch(model, batch, device)
        loss = loss_function(prediction, targets)
        if protocol.check_finite_batches and not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        if not protocol.zero_grad_before_forward:
            optimizer.zero_grad()
        loss.backward()
        if protocol.check_finite_gradients:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Non-finite gradient for {name}")
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=protocol.max_grad_norm,
        )
        optimizer.step()
        batch_size = targets.shape[0]
        loss_total += float(loss.item()) * batch_size
        sample_count += int(batch_size)
    mean_loss = loss_total / sample_count
    if not math.isfinite(mean_loss):
        raise RuntimeError(f"Non-finite mean training loss: {mean_loss}")
    return mean_loss


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    check_finite_batches: bool = False,
) -> float:
    """Compute sample-weighted loss without updating the model."""
    model.eval()
    loss_total = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            prediction, targets = _forward_batch(model, batch, device)
            loss = loss_function(prediction, targets)
            if check_finite_batches and not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss")
            batch_size = targets.shape[0]
            loss_total += float(loss.item()) * batch_size
            sample_count += int(batch_size)
    mean_loss = loss_total / sample_count
    if not math.isfinite(mean_loss):
        raise RuntimeError(f"Non-finite mean validation loss: {mean_loss}")
    return mean_loss


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def train_model(
    model: nn.Module,
    train_inputs: ModelInputs,
    train_targets: ArrayLike,
    validation_inputs: ModelInputs,
    validation_targets: ArrayLike,
    protocol: TrainingProtocol,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Train, early-stop on Validation, and restore the best checkpoint."""
    resolved_device = resolve_device(device)
    train_loader, validation_loader = make_data_loaders(
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        protocol,
    )
    model = model.to(resolved_device)
    optimizer = build_optimizer(model, protocol)
    loss_function = build_loss(protocol)

    best_state = _clone_state(model)
    best_validation_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: list[EpochMetrics] = []

    for epoch in range(1, protocol.max_epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            resolved_device,
            protocol,
        )
        validation_loss = evaluate_loss(
            model,
            validation_loader,
            loss_function,
            resolved_device,
            check_finite_batches=protocol.check_finite_batches,
        )
        history.append(EpochMetrics(epoch, train_loss, validation_loss))

        if validation_loss < best_validation_loss - protocol.improvement_threshold:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = _clone_state(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= protocol.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        epochs_trained=len(history),
        best_validation_loss=best_validation_loss,
        history=tuple(history),
    )
