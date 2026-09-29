"""Final-paper Ridge, LSTM, Vanilla Transformer, and SWiM baselines."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

RIDGE_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 1
TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 128
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_ACTIVATION = "relu"
FIXED_MAX_LEN = 512
SWIM_NUM_BLOCKS = 2
SWIM_WINDOW_CONFIG = {
    "daily": {"window_size": 10, "shift_size": 5},
    "weekly": {"window_size": 4, "shift_size": 2},
}


def flatten_ridge_features(
    frequency: str,
    normalized: np.ndarray | tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Flatten normalized model inputs exactly as in the final Ridge run."""
    if frequency in ("daily_only", "weekly_only"):
        values = normalized
        if not isinstance(values, np.ndarray):
            raise TypeError("Single-frequency Ridge input must be one array.")
        return values.reshape(values.shape[0], -1)
    if frequency == "daily_weekly":
        daily, weekly = normalized
        return np.concatenate(
            [
                daily.reshape(daily.shape[0], -1),
                weekly.reshape(weekly.shape[0], -1),
            ],
            axis=1,
        )
    raise ValueError(f"Unknown frequency setting: {frequency}")


def fit_ridge_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    alpha_grid: Iterable[float] = RIDGE_ALPHA_GRID,
) -> tuple[Ridge, float, float]:
    """Select Ridge alpha by validation MSE using the final-paper procedure."""
    best_alpha: float | None = None
    best_validation_mse = np.inf
    best_model: Ridge | None = None
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        validation_prediction = model.predict(X_validation)
        validation_mse = float(
            np.mean((validation_prediction - y_validation) ** 2)
        )
        if validation_mse < best_validation_mse:
            best_alpha = float(alpha)
            best_validation_mse = validation_mse
            best_model = model
    if best_model is None or best_alpha is None:
        raise ValueError("alpha_grid must contain at least one value.")
    return best_model, best_alpha, best_validation_mse


class LSTMRegressor(nn.Module):
    """Final-paper single-resolution LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class DualFrequencyLSTM(nn.Module):
    """Final-paper Daily+Weekly late-fusion LSTM."""

    def __init__(
        self,
        daily_input_size: int,
        weekly_input_size: int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
    ):
        super().__init__()
        self.daily_lstm = nn.LSTM(
            input_size=daily_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.weekly_lstm = nn.LSTM(
            input_size=weekly_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        x_daily: torch.Tensor,
        x_weekly: torch.Tensor,
    ) -> torch.Tensor:
        daily_output, _ = self.daily_lstm(x_daily)
        weekly_output, _ = self.weekly_lstm(x_weekly)
        fused = torch.cat(
            [daily_output[:, -1, :], weekly_output[:, -1, :]],
            dim=1,
        )
        return self.mlp(fused).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding shared by Transformer baselines."""

    def __init__(
        self,
        d_model: int,
        max_len: int = FIXED_MAX_LEN,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-np.log(10000.0) / d_model)
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


class TransformerBranch(nn.Module):
    """One final-paper Vanilla Transformer resolution branch."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=FIXED_MAX_LEN,
            dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=TRANSFORMER_ACTIVATION,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.positional_encoding(self.input_proj(x))
        hidden = self.encoder(hidden)
        return hidden[:, -1, :]


class SingleFrequencyTransformerRegressor(nn.Module):
    """Final-paper Daily-only or Weekly-only Vanilla Transformer."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        self.branch = TransformerBranch(
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.branch(x)).squeeze(-1)


class DualFrequencyTransformerRegressor(nn.Module):
    """Final-paper Daily+Weekly late-fusion Vanilla Transformer."""

    def __init__(
        self,
        daily_feature_dim: int,
        weekly_feature_dim: int,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        self.daily_branch = TransformerBranch(
            feature_dim=daily_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.weekly_branch = TransformerBranch(
            feature_dim=weekly_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x_daily: torch.Tensor,
        x_weekly: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat(
            [self.daily_branch(x_daily), self.weekly_branch(x_weekly)],
            dim=1,
        )
        return self.fusion(fused).squeeze(-1)


def window_partition_1d(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition ``[B, T, C]`` into non-overlapping temporal windows."""
    batch_size, sequence_length, channels = x.shape
    assert sequence_length % window_size == 0
    window_count = sequence_length // window_size
    return x.reshape(
        batch_size,
        window_count,
        window_size,
        channels,
    ).reshape(batch_size * window_count, window_size, channels)


def window_reverse_1d(
    windows: torch.Tensor,
    window_size: int,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    """Reverse :func:`window_partition_1d`."""
    window_count = sequence_length // window_size
    channels = windows.shape[-1]
    return windows.reshape(
        batch_size,
        window_count,
        window_size,
        channels,
    ).reshape(batch_size, sequence_length, channels)


def compute_pad_len(sequence_length: int, window_size: int) -> int:
    return (window_size - (sequence_length % window_size)) % window_size


def compute_shift_region_ids(
    padded_length: int,
    window_size: int,
    shift_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Label shifted-window regions to block cyclic wrap-around attention."""
    region_id = torch.zeros(padded_length, dtype=torch.long, device=device)
    count = 0
    for region_slice in (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    ):
        region_id[region_slice] = count
        count += 1
    return region_id


def build_window_masks(
    valid_mask: torch.Tensor,
    window_size: int,
    shift_size: int,
) -> torch.Tensor:
    """Build the exact additive validity and shifted-boundary attention mask."""
    device = valid_mask.device
    padded_length = valid_mask.shape[0]
    window_count = padded_length // window_size
    attention_mask = torch.full(
        (window_count, window_size, window_size),
        -torch.inf,
        dtype=torch.float32,
        device=device,
    )
    if shift_size > 0:
        valid_shifted = torch.roll(valid_mask, shifts=-shift_size, dims=0)
    else:
        valid_shifted = valid_mask
    query_valid = valid_shifted.view(window_count, window_size)
    key_valid = valid_shifted.view(window_count, window_size)

    if shift_size > 0:
        region_id = compute_shift_region_ids(
            padded_length,
            window_size,
            shift_size,
            device,
        )
        region_windows = region_id.view(window_count, window_size)
        region_block = region_windows.unsqueeze(2) != region_windows.unsqueeze(1)
    else:
        region_block = torch.zeros(
            (window_count, window_size, window_size),
            dtype=torch.bool,
            device=device,
        )

    allowed_real = query_valid.unsqueeze(2) & key_valid.unsqueeze(1) & ~region_block
    attention_mask.masked_fill_(allowed_real, 0.0)
    padded_rows = ~query_valid
    if padded_rows.any():
        dummy_key = torch.zeros_like(query_valid, dtype=torch.bool)
        dummy_key[:, 0] = True
        attention_mask.masked_fill_(
            padded_rows.unsqueeze(2) & dummy_key.unsqueeze(1),
            0.0,
        )
    return attention_mask


class SWiMWindowBlock(nn.Module):
    """One local or shifted temporal window-attention block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        window_size: int,
        shift_size: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        assert 0 <= shift_size < window_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    @staticmethod
    def _hard_zero(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.view(1, -1, 1).expand_as(x)
        return torch.where(
            mask,
            x,
            torch.zeros((), dtype=x.dtype, device=x.device),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, padded_length, _ = x.shape
        window_count = padded_length // self.window_size
        window_mask = build_window_masks(
            valid_mask,
            self.window_size,
            self.shift_size,
        )
        attention_mask = window_mask.unsqueeze(0).expand(
            batch_size,
            window_count,
            self.window_size,
            self.window_size,
        ).reshape(
            batch_size * window_count,
            self.window_size,
            self.window_size,
        )
        attention_mask = attention_mask.repeat_interleave(
            self.attn.num_heads,
            dim=0,
        )

        residual = x
        shifted = (
            torch.roll(x, shifts=-self.shift_size, dims=1)
            if self.shift_size > 0
            else x
        )
        windows = window_partition_1d(shifted, self.window_size)
        attention_output, _ = self.attn(
            windows,
            windows,
            windows,
            attn_mask=attention_mask,
            need_weights=False,
        )
        attention_output = window_reverse_1d(
            attention_output,
            self.window_size,
            batch_size,
            padded_length,
        )
        if self.shift_size > 0:
            attention_output = torch.roll(
                attention_output,
                shifts=self.shift_size,
                dims=1,
            )
        attention_output = self._hard_zero(attention_output, valid_mask)
        x = self.norm1(residual + self.dropout1(attention_output))
        x = self._hard_zero(x, valid_mask)
        feed_forward_output = self._hard_zero(self.ffn(x), valid_mask)
        x = self.norm2(x + self.dropout2(feed_forward_output))
        return self._hard_zero(x, valid_mask)


class SWiMBranch(nn.Module):
    """Projection, positional encoding, two SWiM blocks, and masked mean."""

    def __init__(
        self,
        feature_dim: int,
        frequency_key: str,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TRANSFORMER_DROPOUT,
    ):
        super().__init__()
        if frequency_key not in SWIM_WINDOW_CONFIG:
            raise ValueError(f"Unknown frequency_key: {frequency_key}")
        configuration = SWIM_WINDOW_CONFIG[frequency_key]
        self.window_size = configuration["window_size"]
        self.shift_size = configuration["shift_size"]
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=FIXED_MAX_LEN,
            dropout=dropout,
        )
        assert SWIM_NUM_BLOCKS == 2
        self.blocks = nn.ModuleList(
            [
                SWiMWindowBlock(
                    d_model,
                    nhead,
                    self.window_size,
                    0,
                    dim_feedforward,
                    dropout,
                ),
                SWiMWindowBlock(
                    d_model,
                    nhead,
                    self.window_size,
                    self.shift_size,
                    dim_feedforward,
                    dropout,
                ),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ):
        batch_size, sequence_length, _ = x.shape
        hidden = self.positional_encoding(self.input_proj(x))
        pad_length = compute_pad_len(sequence_length, self.window_size)
        padded_length = sequence_length + pad_length
        valid_mask = torch.ones(
            padded_length,
            dtype=torch.bool,
            device=x.device,
        )
        if pad_length > 0:
            hidden = torch.cat(
                [
                    hidden,
                    torch.zeros(
                        batch_size,
                        pad_length,
                        hidden.shape[-1],
                        device=hidden.device,
                        dtype=hidden.dtype,
                    ),
                ],
                dim=1,
            )
            valid_mask[sequence_length:] = False
        for block in self.blocks:
            hidden = block(hidden, valid_mask)
        if return_sequence:
            return hidden, valid_mask
        return hidden.sum(dim=1) / float(sequence_length)


class SingleFrequencySWiMRegressor(nn.Module):
    """Final-paper Daily-only or Weekly-only SWiM regressor."""

    def __init__(self, feature_dim: int, frequency_key: str):
        super().__init__()
        self.branch = SWiMBranch(
            feature_dim=feature_dim,
            frequency_key=frequency_key,
        )
        self.head = nn.Linear(TRANSFORMER_D_MODEL, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.branch(x)).squeeze(-1)


class DualFrequencySWiMRegressor(nn.Module):
    """Final-paper Daily+Weekly late-fusion SWiM regressor."""

    def __init__(self, daily_feature_dim: int, weekly_feature_dim: int):
        super().__init__()
        self.daily_branch = SWiMBranch(
            feature_dim=daily_feature_dim,
            frequency_key="daily",
        )
        self.weekly_branch = SWiMBranch(
            feature_dim=weekly_feature_dim,
            frequency_key="weekly",
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * TRANSFORMER_D_MODEL, TRANSFORMER_D_MODEL),
            nn.ReLU(),
            nn.Linear(TRANSFORMER_D_MODEL, 1),
        )

    def forward(
        self,
        x_daily: torch.Tensor,
        x_weekly: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat(
            [self.daily_branch(x_daily), self.weekly_branch(x_weekly)],
            dim=1,
        )
        return self.fusion(fused).squeeze(-1)


def build_lstm_model(
    input_configuration: str,
    daily_feature_dim: int = 5,
    weekly_feature_dim: int = 5,
) -> nn.Module:
    if input_configuration == "daily_only":
        return LSTMRegressor(daily_feature_dim)
    if input_configuration == "weekly_only":
        return LSTMRegressor(weekly_feature_dim)
    if input_configuration == "daily_weekly":
        return DualFrequencyLSTM(daily_feature_dim, weekly_feature_dim)
    raise ValueError(f"Unknown input configuration: {input_configuration}")


def build_transformer_model(
    input_configuration: str,
    daily_feature_dim: int = 5,
    weekly_feature_dim: int = 5,
) -> nn.Module:
    if input_configuration == "daily_only":
        return SingleFrequencyTransformerRegressor(daily_feature_dim)
    if input_configuration == "weekly_only":
        return SingleFrequencyTransformerRegressor(weekly_feature_dim)
    if input_configuration == "daily_weekly":
        return DualFrequencyTransformerRegressor(
            daily_feature_dim,
            weekly_feature_dim,
        )
    raise ValueError(f"Unknown input configuration: {input_configuration}")


def build_swim_model(
    input_configuration: str,
    daily_feature_dim: int = 5,
    weekly_feature_dim: int = 5,
) -> nn.Module:
    if input_configuration == "daily_only":
        return SingleFrequencySWiMRegressor(daily_feature_dim, "daily")
    if input_configuration == "weekly_only":
        return SingleFrequencySWiMRegressor(weekly_feature_dim, "weekly")
    if input_configuration == "daily_weekly":
        return DualFrequencySWiMRegressor(
            daily_feature_dim,
            weekly_feature_dim,
        )
    raise ValueError(f"Unknown input configuration: {input_configuration}")
