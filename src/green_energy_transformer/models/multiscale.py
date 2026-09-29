"""Final-paper adaptive multi-scale Transformer architecture."""

from __future__ import annotations

import math

import torch
from torch import nn

INPUT_DIM = 5
D_MODEL = 32
NHEAD = 4
DIM_FEEDFORWARD = 64
DROPOUT = 0.1
ROUTER_HIDDEN = 32
DAILY_PATCH_SIZES = (5, 10, 20, 30)
WEEKLY_PATCH_SIZES = (2, 4, 8, 13)
SCALE_MODES = ("single", "fixed", "static", "adaptive")
INPUT_CONFIGURATIONS = ("daily_only", "weekly_only", "daily_weekly")


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-(math.log(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional encoding "
                f"capacity {self.pe.size(1)}."
            )
        return self.dropout(x + self.pe[:, : x.size(1), :].to(x.device))


def masked_mean(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """Average only valid observations along ``dim``."""
    valid_mask = valid_mask.to(dtype=torch.bool)
    masked = values * valid_mask.unsqueeze(-1).to(values.dtype)
    denominator = valid_mask.sum(dim=dim, keepdim=True).clamp_min(1).to(values.dtype)
    return masked.sum(dim=dim) / denominator.squeeze(dim).unsqueeze(-1)


def occupancy_weighted_patch_mean(
    patch_tokens: torch.Tensor,
    valid_patch_counts: torch.Tensor,
) -> torch.Tensor:
    """Pool patch tokens in proportion to their valid-observation counts."""
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"patch_tokens must have shape [B, P, D], got {tuple(patch_tokens.shape)}"
        )
    if valid_patch_counts.shape != patch_tokens.shape[:2]:
        raise ValueError(
            "valid_patch_counts shape mismatch: expected "
            f"{tuple(patch_tokens.shape[:2])}, got {tuple(valid_patch_counts.shape)}"
        )
    counts = valid_patch_counts.to(patch_tokens.dtype)
    numerator = (patch_tokens * counts.unsqueeze(-1)).sum(dim=1)
    denominator = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    result = numerator / denominator
    if result.ndim != 2:
        raise ValueError(
            f"occupancy_weighted_patch_mean output must be [B, D], got {tuple(result.shape)}"
        )
    return result


class DualAttentionScaleExpert(nn.Module):
    """One patch scale with intra-patch and inter-patch attention."""

    def __init__(
        self,
        input_dim: int,
        patch_size: int,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        assert input_dim == d_model
        assert d_model % nhead == 0
        assert patch_size > 0

        self.input_dim = input_dim
        self.d_model = d_model
        self.patch_size = int(patch_size)
        self.nhead = nhead
        self.intra_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.intra_norm = nn.LayerNorm(d_model)
        self.patch_positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=1024,
            dropout=0.0,
        )
        self.inter_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.inter_norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.inter_norm2 = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def patchify(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad and reshape a sequence into non-overlapping patches."""
        batch_size, sequence_length, embedding_dim = h.shape
        patch_size = self.patch_size
        if valid_mask is None:
            valid = torch.ones(
                batch_size,
                sequence_length,
                device=h.device,
                dtype=torch.bool,
            )
        else:
            if valid_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    f"valid_mask shape {tuple(valid_mask.shape)} does not match "
                    f"input shape {(batch_size, sequence_length)}."
                )
            valid = valid_mask.to(device=h.device, dtype=torch.bool)

        padded_length = int(math.ceil(sequence_length / patch_size) * patch_size)
        pad_length = padded_length - sequence_length
        if pad_length > 0:
            padding = torch.zeros(
                batch_size,
                pad_length,
                embedding_dim,
                device=h.device,
                dtype=h.dtype,
            )
            h_padded = torch.cat([h, padding], dim=1)
            valid = torch.cat(
                [
                    valid,
                    torch.zeros(
                        batch_size,
                        pad_length,
                        device=h.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
            )
        else:
            h_padded = h

        h_padded = h_padded * valid.unsqueeze(-1).to(h_padded.dtype)
        patch_count = padded_length // patch_size
        patches = h_padded.reshape(
            batch_size,
            patch_count,
            patch_size,
            embedding_dim,
        )
        valid_patches = valid.reshape(batch_size, patch_count, patch_size)
        return patches, valid_patches

    def forward(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        patches, valid_patches = self.patchify(h, valid_mask=valid_mask)
        batch_size, patch_count, patch_size, embedding_dim = patches.shape
        flat_patches = patches.reshape(
            batch_size * patch_count,
            patch_size,
            embedding_dim,
        )
        flat_valid = valid_patches.reshape(batch_size * patch_count, patch_size)

        intra_output, _ = self.intra_attn(
            flat_patches,
            flat_patches,
            flat_patches,
            key_padding_mask=~flat_valid,
            need_weights=False,
        )
        flat_patches = self.intra_norm(flat_patches + intra_output)
        flat_patches = flat_patches * flat_valid.unsqueeze(-1).to(flat_patches.dtype)

        patch_tokens = masked_mean(flat_patches, flat_valid, dim=1)
        patch_tokens = patch_tokens.reshape(
            batch_size,
            patch_count,
            embedding_dim,
        )
        patch_tokens = self.patch_positional_encoding(patch_tokens)
        inter_output, _ = self.inter_attn(
            patch_tokens,
            patch_tokens,
            patch_tokens,
            need_weights=False,
        )
        patch_tokens = self.inter_norm1(patch_tokens + inter_output)
        patch_tokens = self.inter_norm2(patch_tokens + self.ffn(patch_tokens))

        valid_counts = valid_patches.sum(dim=-1).to(patch_tokens.dtype)
        expert_representation = occupancy_weighted_patch_mean(
            patch_tokens,
            valid_counts,
        )
        if expert_representation.shape[1] != patch_tokens.shape[-1]:
            raise ValueError(
                f"Occupancy pooling output shape mismatch: {tuple(expert_representation.shape)}"
            )
        return self.output_norm(expert_representation)


class ScaleAggregator(nn.Module):
    """Aggregate scales within one resolution branch."""

    def __init__(
        self,
        scale_mode: str,
        patch_sizes: list[int] | tuple[int, ...],
        d_model: int = D_MODEL,
        router_hidden: int = ROUTER_HIDDEN,
        selected_patch_size: int | None = None,
    ):
        super().__init__()
        if scale_mode not in SCALE_MODES:
            raise ValueError(f"Unsupported scale_mode={scale_mode}")
        self.scale_mode = scale_mode
        self.patch_sizes = list(patch_sizes)
        self.d_model = d_model
        self.K = len(self.patch_sizes)
        self.selected_patch_size = selected_patch_size

        if scale_mode == "single":
            if selected_patch_size is None:
                raise ValueError("scale_mode='single' requires explicit selected_patch_size.")
            self.selected_patch_size = int(selected_patch_size)
            if self.selected_patch_size not in self.patch_sizes:
                raise ValueError(
                    f"selected_patch_size={selected_patch_size} not in {self.patch_sizes}."
                )
        elif scale_mode == "static":
            self.scale_logits = nn.Parameter(torch.zeros(self.K, dtype=torch.float32))
        elif scale_mode == "adaptive":
            self.router = nn.Sequential(
                nn.Linear(self.K * d_model, router_hidden),
                nn.ReLU(),
                nn.Linear(router_hidden, self.K),
            )

    def _single_weights(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones(batch_size, 1, device=device, dtype=torch.float32)

    def _fixed_weights(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (batch_size, self.K),
            1.0 / self.K,
            device=device,
            dtype=torch.float32,
        )

    def _static_weights(self, batch_size: int, device: torch.device) -> torch.Tensor:
        weights = torch.softmax(self.scale_logits, dim=0)
        return weights.view(1, -1).expand(batch_size, -1)

    def _adaptive_weights(
        self,
        expert_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        logits = self.router(torch.cat(expert_outputs, dim=-1))
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        expert_outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not expert_outputs:
            raise ValueError("No expert outputs available for scale aggregation.")
        batch_size = expert_outputs[0].shape[0]
        device = expert_outputs[0].device

        if self.scale_mode == "single":
            return expert_outputs[0], self._single_weights(batch_size, device)
        if self.scale_mode == "fixed":
            weights = self._fixed_weights(batch_size, device)
            representation = sum(
                weight * expert
                for weight, expert in zip(weights[0], expert_outputs)
            )
            return representation, weights
        if self.scale_mode == "static":
            weights = self._static_weights(batch_size, device)
            representation = sum(
                weight * expert
                for weight, expert in zip(weights[0], expert_outputs)
            )
            return representation, weights
        if self.scale_mode == "adaptive":
            weights = self._adaptive_weights(expert_outputs)
            representation = sum(
                weight.unsqueeze(-1) * expert
                for weight, expert in zip(weights.T, expert_outputs)
            )
            return representation, weights
        raise ValueError(f"Unsupported scale_mode={self.scale_mode}")


class FrequencyAdaptivePathFormerBranch(nn.Module):
    """Daily or Weekly branch with resolution-specific patch experts."""

    def __init__(
        self,
        frequency: str,
        input_dim: int,
        patch_sizes: list[int] | tuple[int, ...],
        scale_mode: str,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        selected_patch_size: int | None = None,
    ):
        super().__init__()
        assert frequency in {"daily", "weekly"}
        assert len(patch_sizes) == 4
        self.frequency = frequency
        self.input_dim = input_dim
        self.patch_sizes = list(patch_sizes)
        self.scale_mode = scale_mode
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=1024,
            dropout=0.0,
        )

        if scale_mode == "single":
            if selected_patch_size is None:
                raise ValueError(
                    "scale_mode='single' requires an explicit selected_patch_size."
                )
            if int(selected_patch_size) not in self.patch_sizes:
                raise ValueError(
                    f"selected_patch_size={selected_patch_size} not in {self.patch_sizes}."
                )
            self.selected_patch_size = int(selected_patch_size)
            self.experts = nn.ModuleList(
                [
                    DualAttentionScaleExpert(
                        input_dim=d_model,
                        patch_size=self.selected_patch_size,
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                    )
                ]
            )
        else:
            self.experts = nn.ModuleList(
                [
                    DualAttentionScaleExpert(
                        input_dim=d_model,
                        patch_size=patch_size,
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                    )
                    for patch_size in patch_sizes
                ]
            )
            self.selected_patch_size = None

        self.aggregator = ScaleAggregator(
            scale_mode=scale_mode,
            patch_sizes=patch_sizes,
            d_model=d_model,
            router_hidden=ROUTER_HIDDEN,
            selected_patch_size=selected_patch_size,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.positional_encoding(self.input_proj(x))
        expert_outputs = [expert(hidden) for expert in self.experts]
        return self.aggregator(expert_outputs)


class SingleFrequencyAdaptivePathFormerRegressor(nn.Module):
    """Daily-only or Weekly-only final-paper regressor."""

    def __init__(
        self,
        frequency: str,
        scale_mode: str,
        input_dim: int = INPUT_DIM,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
        selected_patch_size: int | None = None,
    ):
        super().__init__()
        if frequency == "daily":
            patch_sizes = DAILY_PATCH_SIZES
        elif frequency == "weekly":
            patch_sizes = WEEKLY_PATCH_SIZES
        else:
            raise ValueError(f"Unsupported frequency={frequency}")
        self.frequency = frequency
        self.scale_mode = scale_mode
        self.branch = FrequencyAdaptivePathFormerBranch(
            frequency=frequency,
            input_dim=input_dim,
            patch_sizes=patch_sizes,
            scale_mode=scale_mode,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            selected_patch_size=selected_patch_size,
        )
        self.head = nn.Linear(d_model, 1)

    def forward_with_router(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation, weights = self.branch(x)
        return self.head(representation).squeeze(-1), weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prediction, _ = self.forward_with_router(x)
        return prediction

    def get_router_weights(self, x: torch.Tensor) -> torch.Tensor:
        _, weights = self.branch(x)
        return weights


class DualFrequencyAdaptivePathFormerRegressor(nn.Module):
    """Daily+Weekly late-fusion final-paper regressor."""

    def __init__(
        self,
        scale_mode: str,
        daily_selected_patch_size: int | None = None,
        weekly_selected_patch_size: int | None = None,
        input_dim: int = INPUT_DIM,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.scale_mode = scale_mode
        self.daily_branch = FrequencyAdaptivePathFormerBranch(
            frequency="daily",
            input_dim=input_dim,
            patch_sizes=DAILY_PATCH_SIZES,
            scale_mode=scale_mode,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            selected_patch_size=daily_selected_patch_size,
        )
        self.weekly_branch = FrequencyAdaptivePathFormerBranch(
            frequency="weekly",
            input_dim=input_dim,
            patch_sizes=WEEKLY_PATCH_SIZES,
            scale_mode=scale_mode,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            selected_patch_size=weekly_selected_patch_size,
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward_with_router(
        self,
        x_daily: torch.Tensor,
        x_weekly: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        daily_representation, daily_weights = self.daily_branch(x_daily)
        weekly_representation, weekly_weights = self.weekly_branch(x_weekly)
        fused = torch.cat([daily_representation, weekly_representation], dim=-1)
        prediction = self.fusion(fused).squeeze(-1)
        return prediction, daily_weights, weekly_weights

    def forward(
        self,
        x_daily: torch.Tensor,
        x_weekly: torch.Tensor,
    ) -> torch.Tensor:
        prediction, _, _ = self.forward_with_router(x_daily, x_weekly)
        return prediction

    def get_router_weights(
        self,
        x_daily: torch.Tensor,
        x_weekly: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, daily_weights = self.daily_branch(x_daily)
        _, weekly_weights = self.weekly_branch(x_weekly)
        return daily_weights, weekly_weights


def build_multiscale_model(
    input_configuration: str,
    scale_mode: str = "adaptive",
    selected_daily_patch: int | None = None,
    selected_weekly_patch: int | None = None,
) -> nn.Module:
    """Build one final-paper input/aggregation configuration."""
    if input_configuration == "daily_only":
        selected_patch = selected_daily_patch if selected_daily_patch is not None else 20
        return SingleFrequencyAdaptivePathFormerRegressor(
            frequency="daily",
            scale_mode=scale_mode,
            selected_patch_size=selected_patch if scale_mode == "single" else None,
        )
    if input_configuration == "weekly_only":
        selected_patch = selected_weekly_patch if selected_weekly_patch is not None else 4
        return SingleFrequencyAdaptivePathFormerRegressor(
            frequency="weekly",
            scale_mode=scale_mode,
            selected_patch_size=selected_patch if scale_mode == "single" else None,
        )
    if input_configuration == "daily_weekly":
        return DualFrequencyAdaptivePathFormerRegressor(
            scale_mode=scale_mode,
            daily_selected_patch_size=(
                selected_daily_patch if scale_mode == "single" else None
            ),
            weekly_selected_patch_size=(
                selected_weekly_patch if scale_mode == "single" else None
            ),
        )
    raise ValueError(f"Unsupported input_configuration={input_configuration}")


class AdaptiveMultiScaleTransformer(nn.Module):
    """Public wrapper for Daily-only, Weekly-only, or Daily+Weekly models."""

    def __init__(
        self,
        input_configuration: str,
        scale_mode: str = "adaptive",
        selected_daily_patch: int | None = None,
        selected_weekly_patch: int | None = None,
    ):
        super().__init__()
        self.input_configuration = input_configuration
        self.scale_mode = scale_mode
        self.model = build_multiscale_model(
            input_configuration=input_configuration,
            scale_mode=scale_mode,
            selected_daily_patch=selected_daily_patch,
            selected_weekly_patch=selected_weekly_patch,
        )

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.model(*inputs)

    def forward_with_router(self, *inputs: torch.Tensor):
        return self.model.forward_with_router(*inputs)

    def get_router_weights(self, *inputs: torch.Tensor):
        return self.model.get_router_weights(*inputs)
