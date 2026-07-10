"""Helpers for optional periodic checkpoint retention.

Training always keeps best/last checkpoints. Periodic ``ep*.pth`` checkpoints are
disabled by default to avoid filling the log directory during long experiment
batches; set ``SAVE_PERIODIC_CHECKPOINTS=1`` to keep them.
"""

import glob
import os


def should_save_periodic_checkpoint():
    return os.getenv("SAVE_PERIODIC_CHECKPOINTS", "0") == "1"


def cleanup_periodic_checkpoints(save_dir):
    if should_save_periodic_checkpoint():
        return
    patterns = [
        os.path.join(save_dir, "ep*.pth"),
        os.path.join(save_dir, "epoch_*.pth"),
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            name = os.path.basename(path)
            if name in {"best.pth", "best_epoch_weights.pth", "best_map_weights.pth", "last.pth"}:
                continue
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
