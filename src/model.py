"""
model.py

LSTM encoder-decoder for trajectory imputation.

Design (review against what Sir expects before trusting it):
  - Task framing: given a windowed sequence where some GPS (lat/lon) points
    are masked (zeroed + flagged via a mask channel), reconstruct the full
    lat/lon trajectory. IMU channels (accel/gyro) are assumed always-present
    context and are NOT masked.
  - Input per timestep: [lat, lon, alt, speed_2d, speed_3d, accel_*, gyro_*,
    mask_flag] — mask_flag is 1 where lat/lon was artificially zeroed for
    training, 0 otherwise. This lets the model learn "trust IMU more when
    mask_flag=1".
  - Encoder: multi-layer LSTM compresses the windowed sequence into a
    hidden state.
  - Decoder: multi-layer LSTM, seeded with the encoder's final hidden state,
    reconstructs a lat/lon sequence of the same length.
  - Output: only 2 values per timestep (lat, lon) — evaluate.py would then
    compute RMSE (in meters, via haversine) only at masked positions.

This does NOT include the masking/windowing logic itself — that belongs in
train.py's data preparation (turning a merged route CSV into (input, target,
mask) windows). This file is architecture only.
"""

import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    """
    LSTM encoder-decoder for imputing masked GPS points in a trajectory,
    conditioned on IMU (accel/gyro) context.

    Args:
        n_features: number of input features per timestep (lat, lon, alt,
            speed_2d, speed_3d, accel_*, gyro_*, mask_flag). Varies by route
            depending on how many sensor positions (back/front/handlebar)
            are present — pass the actual column count from train.py.
        hidden_size: LSTM hidden dimension.
        num_layers: number of stacked LSTM layers in encoder and decoder.
        dropout: dropout between LSTM layers (only applied if num_layers > 1).
        output_size: number of values predicted per timestep. Default 2
            (lat, lon). Set to n_features - 1 if you want the model to
            reconstruct all channels instead of just GPS.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 2,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder consumes its own previous output autoregressively at
        # train time via teacher forcing (see forward()); at inference
        # you'd typically feed zeros or the masked input again.
        self.decoder = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features) — full input sequence, GPS
               channels zeroed at masked positions, mask_flag set accordingly.
            target: (batch, seq_len, output_size) — ground-truth lat/lon,
                used for teacher forcing during training. None at inference.
            teacher_forcing_ratio: probability of feeding ground truth
                (vs. the model's own previous prediction) as the next
                decoder input step. Ignored if target is None.

        Returns:
            (batch, seq_len, output_size) reconstructed lat/lon sequence.
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        _, (hidden, cell) = self.encoder(x)

        # First decoder input: zeros (no "previous" prediction yet)
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=device)
        outputs = []

        for t in range(seq_len):
            dec_out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.output_head(dec_out)  # (batch, 1, output_size)
            outputs.append(pred)

            use_teacher_forcing = (
                target is not None
                and self.training
                and torch.rand(1).item() < teacher_forcing_ratio
            )
            if use_teacher_forcing:
                decoder_input = target[:, t:t + 1, :]
            else:
                decoder_input = pred.detach()

        return torch.cat(outputs, dim=1)


def masked_rmse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    RMSE computed only at masked (imputed) positions — the positions that
    matter for evaluating imputation quality, not the whole sequence.

    Args:
        pred:   (batch, seq_len, 2) predicted lat/lon
        target: (batch, seq_len, 2) ground-truth lat/lon
        mask:   (batch, seq_len) 1 where the point was masked (i.e. is an
                imputation target), 0 where it was real input.
    """
    mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1), broadcasts over lat/lon
    sq_err = (pred - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return torch.sqrt(sq_err.sum() / (denom * pred.shape[-1]))


if __name__ == "__main__":
    # Quick smoke test with dummy data — confirms shapes flow correctly
    # before wiring in real data via train.py.
    n_features = 24  # 23 sensor cols + 1 mask flag, adjust to match your data
    model = TrajectoryLSTM(n_features=n_features)

    batch, seq_len = 4, 50
    x = torch.randn(batch, seq_len, n_features)
    target = torch.randn(batch, seq_len, 2)
    mask = torch.randint(0, 2, (batch, seq_len))

    pred = model(x, target=target)
    loss = masked_rmse_loss(pred, target, mask)
    print(f"Output shape: {pred.shape} (expected: [{batch}, {seq_len}, 2])")
    print(f"Smoke-test loss: {loss.item():.4f}") 