import argparse
import os
import sys

import wandb


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a W&B artifact model for g2048.")
    parser.add_argument("--entity", type=str, default="kywch")
    parser.add_argument("--project", type=str, default="pufferlib")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--version", type=str, default="latest")
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory to place the downloaded artifact.",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    wandb.init(id=args.run_id, project=args.project, entity=args.entity)
    try:
        artifact = wandb.use_artifact(f"{args.run_id}:{args.version}")
        data_dir = artifact.download(root=args.outdir)
    finally:
        wandb.finish()

    files = sorted(os.listdir(data_dir))
    if not files:
        print(f"No files found in artifact directory: {data_dir}", file=sys.stderr)
        return 1

    model_file = files[-1]
    model_path = os.path.join(data_dir, model_file)
    print(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
