from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lithos_ssl.data.datamodule import LithosDataModule  # noqa: E402
from lithos_ssl.eval.embed import extract_embeddings  # noqa: E402
from lithos_ssl.eval.knn import knn_accuracy  # noqa: E402
from lithos_ssl.eval.linear_probe import linear_probe_accuracy  # noqa: E402
from lithos_ssl.models import FROZEN_MODEL_REGISTRY, MODEL_REGISTRY  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(set(MODEL_REGISTRY) | set(FROZEN_MODEL_REGISTRY)),
        help="Which architecture this checkpoint is, or which frozen baseline to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a .ckpt from scripts/train.py. Required for trained models, ignored "
        "for frozen baselines (e.g. dinov3_frozen).",
    )
    parser.add_argument("--checkpoint-id", default=None, help="HF model id override for frozen baselines.")
    parser.add_argument("--data-root", default="data/raw/lithos")
    parser.add_argument("--csv-name", default="patch_index.csv")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--label-column", default="Multiclass_label")
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(args: argparse.Namespace, device: str):
    if args.model in FROZEN_MODEL_REGISTRY:
        kwargs = {"checkpoint_id": args.checkpoint_id} if args.checkpoint_id else {}
        model = FROZEN_MODEL_REGISTRY[args.model](**kwargs)
    else:
        if not args.checkpoint:
            raise SystemExit(f"--checkpoint is required for trained model '{args.model}'")
        model_cls = MODEL_REGISTRY[args.model]
        # Lightning reconstructs the model from the hyperparameters every model class
        # in this project saves via self.save_hyperparameters(), so this works without
        # us having to separately track each run's config.
        model = model_cls.load_from_checkpoint(args.checkpoint, map_location="cpu")
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    device = pick_device()
    model = load_model(args, device)

    datamodule = LithosDataModule(
        root=args.data_root,
        method=None,  # evaluation only: no SSL multi-view transform needed
        csv_name=args.csv_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_column=args.label_column,
    )
    datamodule.setup()

    # The labelled, capped train subset (`train_eval_dataloader`) is the "reference
    # bank" / linear-probe training set; the held-out test split is what both metrics
    # are reported on.
    train_embeddings, train_labels = extract_embeddings(model, datamodule.train_eval_dataloader(), device)
    test_embeddings, test_labels = extract_embeddings(model, datamodule.test_dataloader(), device)

    knn_acc = knn_accuracy(train_embeddings, train_labels, test_embeddings, test_labels, k=args.knn_k)
    linear_acc = linear_probe_accuracy(train_embeddings, train_labels, test_embeddings, test_labels)
    num_params = sum(p.numel() for p in model.parameters())

    result = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "num_params": int(num_params),
        "embedding_dim": int(train_embeddings.shape[1]),
        "knn_accuracy": knn_acc,
        "linear_probe_accuracy": linear_acc,
        "n_train_reference": int(len(train_labels)),
        "n_test": int(len(test_labels)),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": args.notes,
    }

    runs_dir = Path(args.results_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = runs_dir / f"{args.model}_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")
    print("Run `python scripts/aggregate_results.py` to rebuild results/RESULTS.md.")


if __name__ == "__main__":
    main()
