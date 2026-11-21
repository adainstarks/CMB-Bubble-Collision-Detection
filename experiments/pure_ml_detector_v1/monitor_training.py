"""Monitor pure-ML detector and segmentation training runs from a terminal."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Print live status for a pure-ML detector training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dir", type=str)
    parser.add_argument("--lines", type=int, default=8)
    parser.add_argument("--watch", type=float, default=0.0, help="Refresh interval in seconds. Use 0 for one-shot.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load JSON if present."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_list(path: Path) -> list[dict]:
    """Load a JSON list if present."""

    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def tail_lines(path: Path, n: int) -> list[str]:
    """Return last n text lines."""

    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-max(int(n), 0):]


def process_alive(pid: int | None) -> bool:
    """Return whether a process id is alive."""

    if pid is None:
        return False
    try:
        subprocess.run(["ps", "-p", str(int(pid))], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def nvidia_smi() -> str:
    """Return compact GPU utilization if nvidia-smi exists."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except Exception:
        return "nvidia-smi unavailable"


def fmt_float(value: object, digits: int = 4) -> str:
    """Format numeric values without crashing on missing fields."""

    if value is None:
        return "-"
    try:
        return f"{float(value):.{int(digits)}f}"
    except Exception:
        return str(value)


def fmt_rate(value: object) -> str:
    """Format rates for terminal display."""

    if value is None:
        return "-"
    try:
        value_f = float(value)
    except Exception:
        return str(value)
    if value_f >= 100.0:
        return f"{value_f:.0f}/s"
    if value_f >= 10.0:
        return f"{value_f:.1f}/s"
    return f"{value_f:.2f}/s"


def fmt_duration(seconds: object) -> str:
    """Format seconds as a compact wall-clock duration."""

    if seconds is None:
        return "-"
    try:
        total = max(0, int(round(float(seconds))))
    except Exception:
        return str(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def progress_bar(percent: object, width: int = 28) -> str:
    """Return an ASCII progress bar."""

    try:
        pct = min(100.0, max(0.0, float(percent)))
    except Exception:
        pct = 0.0
    filled = int(round(width * pct / 100.0))
    return "[" + "#" * filled + "." * (width - filled) + f"] {pct:5.1f}%"


def print_progress(progress: dict, *, active: bool) -> None:
    """Print active step progress when available."""

    title = "Active progress" if active else "Latest progress"
    if not progress:
        print(f"\n{title}:")
        print("(no step progress yet)")
        return
    phase = progress.get("phase", "-")
    epoch = progress.get("epoch", "-")
    step = progress.get("step", "-")
    total_steps = progress.get("total_steps", "-")
    print(f"\n{title}:")
    print(
        f"{phase} | epoch {epoch} | step {step}/{total_steps} "
        f"{progress_bar(progress.get('percent'))}"
    )
    if progress.get("avg_loss") is not None:
        print(
            "loss "
            f"avg={fmt_float(progress.get('avg_loss'))} "
            f"last={fmt_float(progress.get('last_loss'))} | "
            f"focal={fmt_float(progress.get('avg_bce_or_focal'))} "
            f"rank={fmt_float(progress.get('avg_ranking_loss'))} | "
            f"lr={fmt_float(progress.get('learning_rate'), 6)}"
        )
    print(
        "speed "
        f"samples={progress.get('samples', '-')}/{progress.get('total_samples', '-')} "
        f"({fmt_rate(progress.get('samples_per_sec'))}) | "
        f"elapsed={fmt_duration(progress.get('elapsed_sec'))} "
        f"eta={fmt_duration(progress.get('eta_sec'))}"
    )


def print_status(run_dir: Path, lines: int) -> None:
    """Print one status snapshot."""

    if (run_dir / "history.json").exists() and (run_dir / "run_config.json").exists() and not (run_dir / "status.json").exists():
        print_unet_status(run_dir, lines)
        return

    status = load_json(run_dir / "status.json")
    pid = status.get("pid")
    alive = process_alive(pid if pid is not None else None)
    print("=" * 88)
    print(f"run_dir: {run_dir}")
    print(f"state: {status.get('state')} | pid: {pid} | alive: {alive}")
    print(
        f"arch: {status.get('architecture')} | gpu_ids: {status.get('gpu_ids')} | "
        f"data_parallel: {status.get('data_parallel')} | amp: {status.get('amp')}"
    )
    completed_epoch = status.get("completed_epoch", status.get("current_epoch"))
    active_epoch = status.get("active_epoch")
    epoch_text = f"completed: {completed_epoch} / {status.get('final_epoch')}"
    if active_epoch is not None:
        epoch_text += f" | active: {active_epoch}"
    print(f"epoch: {epoch_text} | best_recall: {status.get('best_recall')} | updated: {status.get('updated_utc')}")
    last = status.get("last_epoch_metrics")
    if last:
        print(
            "last: "
            f"epoch={last.get('epoch')} "
            f"train_loss={fmt_float(last.get('train_loss'))} "
            f"val_loss={fmt_float(last.get('val_loss'))} "
            f"val_auroc={fmt_float(last.get('val_auroc'))} "
            f"recall={fmt_float(last.get('target_recall'))} "
            f"precision={fmt_float(last.get('target_precision'))} "
            f"fpr={fmt_float(last.get('target_realized_fpr'))} "
            f"thr={fmt_float(last.get('target_threshold'))}"
        )
    print_progress(load_json(run_dir / "progress.json"), active=status.get("state") == "running")
    print("\nGPU:")
    print(nvidia_smi())
    print("\nRecent live metrics:")
    metric_lines = tail_lines(run_dir / "live_metrics.jsonl", lines)
    if not metric_lines:
        print("(no completed epoch metrics yet; waiting for validation to finish)")
    for line in metric_lines:
        try:
            row = json.loads(line)
            print(
                f"epoch {row.get('epoch')}: "
                f"loss={row.get('train_loss'):.4f} "
                f"val={fmt_float(row.get('val_loss'))} "
                f"auroc={row.get('val_auroc'):.4f} "
                f"recall={row.get('target_recall'):.4f} "
                f"precision={fmt_float(row.get('target_precision'))} "
                f"fpr={fmt_float(row.get('target_realized_fpr'))} "
                f"best={row.get('best_recall_so_far'):.4f}"
            )
        except Exception:
            print(line)


def metric_from(history_row: dict, split: str, name: str) -> object:
    """Return one nested metric from a Phase 3 segmentation history row."""

    split_metrics = history_row.get(split, {})
    if not isinstance(split_metrics, dict):
        return None
    return split_metrics.get(name)


def print_unet_status(run_dir: Path, lines: int) -> None:
    """Print one status snapshot for Phase 3 segmentation/U-Net runs."""

    run_config = load_json(run_dir / "run_config.json")
    args = run_config.get("args", {}) if isinstance(run_config.get("args", {}), dict) else {}
    split = run_config.get("split", {}) if isinstance(run_config.get("split", {}), dict) else {}
    history = load_json_list(run_dir / "history.json")
    checkpoint_metric = args.get("checkpoint_metric", "hard_dice_pos")
    last = history[-1] if history else {}
    final_epoch = args.get("epochs", "-")
    completed_epoch = last.get("epoch", 0)
    best_row = None
    if history:
        best_row = max(history, key=lambda row: float(metric_from(row, "val", checkpoint_metric) or -1.0e99))

    batch_size = int(args.get("batch_size") or 0)
    grad_accum_steps = int(args.get("grad_accum_steps") or 1)
    effective_batch = batch_size * grad_accum_steps if batch_size > 0 else "-"

    print("=" * 88)
    print(f"run_dir: {run_dir}")
    print("state: phase3 segmentation run | pid: not tracked | alive: check GPU/process table")
    print(
        f"arch: {args.get('model_architecture')} | encoder: {args.get('encoder_name')} | "
        f"weights: {args.get('encoder_weights')} | gpu_ids: {args.get('gpu_ids', '-')}"
    )
    print(
        f"epoch: completed {completed_epoch} / {final_epoch} | "
        f"batch={batch_size} | accum={grad_accum_steps} | effective_batch={effective_batch}"
    )
    print(
        f"samples: train={split.get('train_samples', '-')} "
        f"calibration={split.get('calibration_samples', split.get('val_samples', '-'))} "
        f"test={split.get('test_samples', '-')}"
    )
    if best_row:
        print(
            f"best: epoch={best_row.get('epoch')} "
            f"{checkpoint_metric}={fmt_float(metric_from(best_row, 'val', checkpoint_metric))}"
        )
    if last:
        print(
            "last: "
            f"epoch={last.get('epoch')} "
            f"train_loss={fmt_float(metric_from(last, 'train', 'loss'))} "
            f"val_loss={fmt_float(metric_from(last, 'val', 'loss'))} "
            f"val_dice_pos={fmt_float(metric_from(last, 'val', 'hard_dice_pos'))} "
            f"val_iou_pos={fmt_float(metric_from(last, 'val', 'iou_pos'))} "
            f"val_img_f1={fmt_float(metric_from(last, 'val', 'image_f1'))} "
            f"val_img_recall={fmt_float(metric_from(last, 'val', 'image_recall'))} "
            f"lr={fmt_float(last.get('learning_rate'), 6)}"
        )
    print("\nGPU:")
    print(nvidia_smi())
    print("\nRecent epoch metrics:")
    if not history:
        print("(no completed epoch metrics yet; waiting for validation to finish)")
        return
    for row in history[-max(int(lines), 0):]:
        print(
            f"epoch {row.get('epoch')}: "
            f"loss={fmt_float(metric_from(row, 'train', 'loss'))} "
            f"val={fmt_float(metric_from(row, 'val', 'loss'))} "
            f"dice_pos={fmt_float(metric_from(row, 'val', 'hard_dice_pos'))} "
            f"iou_pos={fmt_float(metric_from(row, 'val', 'iou_pos'))} "
            f"img_f1={fmt_float(metric_from(row, 'val', 'image_f1'))} "
            f"img_recall={fmt_float(metric_from(row, 'val', 'image_recall'))} "
            f"lr={fmt_float(row.get('learning_rate'), 6)}"
        )


def main() -> None:
    """Monitor one run."""

    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if float(args.watch) <= 0.0:
        print_status(run_dir, int(args.lines))
        return
    while True:
        print("\033[2J\033[H", end="")
        print_status(run_dir, int(args.lines))
        time.sleep(float(args.watch))


if __name__ == "__main__":
    main()
