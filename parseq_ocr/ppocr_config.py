"""Separate PP-OCRv6-derived price recipe; does not change PARSeq defaults."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/ppocr_price.yaml"):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    cfg = yaml.safe_load(path.read_text())
    for key in ["dataset_root", "output_dir", "pretrained_dir"]:
        p = Path(cfg[key]).expanduser()
        cfg[key] = p if p.is_absolute() else ROOT / p
    if cfg.get("height") != 128 or cfg.get("width") != 320:
        raise ValueError("Validated price preset requires128×320")
    if cfg.get("selection_metric") != "exact":
        raise ValueError("Only literal exact selection_metric: exact is supported")
    if cfg.get("epochs", 0) < 1 or cfg.get("batch_size", 0) < 1:
        raise ValueError("Positive epochs/batch required")
    if not 0 < cfg.get("holdout_fraction", 0.1) < 1:
        raise ValueError("Holdout fraction must lie0..1")
    for key in ("initial_checkpoint", "train_manifest", "validation_manifest", "precursor_receipt"):
        if cfg.get(key):
            value = Path(cfg[key]).expanduser()
            cfg[key] = value if value.is_absolute() else ROOT / value
    if cfg.get("recipe") not in (None, "legacy", "corrected_fixed_bn_v1", "selection_only_precursor_v1"):
        raise ValueError("Unknown PP recipe; use legacy or corrected_fixed_bn_v1")
    return cfg
