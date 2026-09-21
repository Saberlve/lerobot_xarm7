"""Pure PyTorch NEXT regressors adapted from ``philiphan0109/factr2_next``.

The upstream project is Apache-2.0 licensed. The layer layout and stateless
sliding-window semantics are intentionally kept checkpoint-compatible; ROS2
transport is not part of this module.
"""

from __future__ import annotations


def _torch_modules():
    try:
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "NEXT requires PyTorch; install the project with the 'next' extra"
        ) from exc
    return nn


def build_model(model_cfg, input_size: int, output_size: int, history: int):
    """Build the official MLP/GRU/LSTM NEXT architecture."""
    nn = _torch_modules()
    model_type = str(model_cfg.get("type", "lstm")).lower()
    hidden_size = int(model_cfg.get("hidden_size", 128))
    num_layers = int(model_cfg.get("num_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.0))

    if model_type == "mlp":
        layers = [nn.Flatten()]
        width = int(input_size) * int(history)
        for _ in range(max(1, num_layers)):
            layers.extend((nn.Linear(width, hidden_size), nn.ReLU()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = hidden_size
        layers.append(nn.Linear(width, int(output_size)))
        return nn.Sequential(*layers)

    state_mode = str(model_cfg.get("state_mode", "stateless")).lower()
    if state_mode != "stateless":
        raise NotImplementedError(
            "NEXT state_mode must be stateless; online context comes from the rolling history"
        )
    if model_type not in ("gru", "lstm"):
        raise ValueError("NEXT model type must be one of: mlp, gru, lstm")

    recurrent_type = nn.GRU if model_type == "gru" else nn.LSTM
    recurrent = recurrent_type(
        input_size=int(input_size),
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bool(model_cfg.get("bidirectional", False)),
        dropout=dropout if num_layers > 1 else 0.0,
    )
    head_input = hidden_size * (2 if bool(model_cfg.get("bidirectional", False)) else 1)
    head_hidden = int(model_cfg.get("head_hidden", 256))
    head_layers = int(model_cfg.get("head_layers", 2))
    head = []
    width = head_input
    for _ in range(max(0, head_layers - 1)):
        head.extend((nn.Linear(width, head_hidden), nn.ReLU()))
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        width = head_hidden
    head.append(nn.Linear(width, int(output_size)))

    class RecurrentRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = recurrent
            # Preserve official state-dict names for existing NEXT checkpoints.
            setattr(self, model_type, recurrent)
            self.head = nn.Sequential(*head)

        def forward(self, x):
            output, _ = self.recurrent(x)
            return self.head(output[:, -1])

    model = RecurrentRegressor()
    # Registering the same module twice creates duplicate state-dict keys.
    # Remove the generic registration while keeping the forward reference.
    model._modules.pop("recurrent")
    object.__setattr__(model, "recurrent", getattr(model, model_type))
    return model
