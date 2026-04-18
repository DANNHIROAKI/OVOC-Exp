from __future__ import annotations

import argparse
from pathlib import Path

from .runner import load_runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenViking × OpenClaw three-group benchmark runner",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "experiment.yaml"),
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing artifacts/work dirs before running",
    )

    sub = parser.add_subparsers(dest="command", required=False)
    sub.add_parser("prepare", help="Clone/install dependencies and create snapshots")
    sub.add_parser("run", help="Execute all blocks from prepared snapshots")
    sub.add_parser("summarize", help="Aggregate block outputs into final tables")
    sub.add_parser("full-run", help="Prepare, run, aggregate, and summarize")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "full-run"

    runner = load_runner(args.config, fresh=args.fresh)

    if command == "prepare":
        runner.prepare()
    elif command == "run":
        runner.run_all_blocks()
    elif command == "summarize":
        runner.aggregate_outputs()
        runner.finalize_manifest()
    elif command == "full-run":
        runner.full_run()
    else:
        parser.error(f"Unsupported command: {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
