"""Train a ROS-free xArm7 NEXT model from contact-free diagnostic CSV files."""

import argparse
from pathlib import Path

from ..utils.next_training import TrainOptions, train_next


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data", type=Path, nargs="+", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--history", type=int, default=50)
    value.add_argument("--model", choices=("lstm", "gru", "mlp"), default="lstm")
    value.add_argument("--epochs", type=int, default=20)
    value.add_argument("--batch-size", type=int, default=2048)
    value.add_argument("--learning-rate", type=float, default=1e-3)
    value.add_argument("--val-fraction", type=float, default=0.1)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--device", default="cpu")
    value.add_argument("--plot", action="store_true", help="save loss.png (requires matplotlib)")
    return value


def main():
    args = parser().parse_args()
    options = TrainOptions(
        history=args.history,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        plot=args.plot,
    )
    train_next(args.data, args.output, options)
    print(f"Saved NEXT artifacts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
