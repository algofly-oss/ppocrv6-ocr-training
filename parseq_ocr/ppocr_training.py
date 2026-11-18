"""Train-only selection then fresh full-fit PP-OCRv6-derived price recipe."""
from pathlib import Path
import contextlib, json, random, time
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from .ppocr_models import VOCAB, create_model, restore_bn, preprocess, decode
from .metrics import score_pairs, selection_score, EXACT_POLICY

from .ppocr_data import canonical, scan_labeled, split_price_holdout


class Crops(Dataset):
    def __init__(self, rows, height, augment=False):
        self.rows = rows
        self.height = height
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r["path"]) as im:
            image = im.convert("RGB")
        if self.augment:
            if random.random() < 0.4:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.4:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.25:
                image = image.rotate(
                    random.uniform(-2, 2),
                    Image.Resampling.BILINEAR,
                    fillcolor=(127, 127, 127),
                )
        return torch.from_numpy(preprocess(image, self.height)), r["label"], r["image"]


def loader(rows, cfg, augment=False):
    return DataLoader(
        Crops(rows, cfg["height"], augment),
        batch_size=cfg.get("batch_size", 64),
        shuffle=augment,
        num_workers=cfg.get("workers", 0),
    )


def autocast(cfg):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if cfg.get("device", "cuda") == "cuda"
        and cfg.get("precision", "bf16") == "bf16"
        else contextlib.nullcontext()
    )


@torch.inference_mode()
def evaluate_model(model, rows, cfg, output=None):
    model.eval()
    pred = []
    for x, labels, ids in loader(canonical(rows), cfg):
        with autocast(cfg):
            z = model(x.to(cfg.get("device", "cuda")))
        pred.extend(
            dict(image=i, gt=gt, prediction=t)
            for i, gt, t in zip(ids, labels, decode(z.argmax(-1).cpu().tolist()))
        )
    labels = [r["gt"] for r in pred]
    texts = [r["prediction"] for r in pred]
    metrics = {
        **score_pairs(labels, texts, "none"),
        "precision": cfg.get("precision", "bf16"),
    }
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in pred)
        )
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _fit(cfg, rows, held, epochs, folder):
    seed = cfg.get("seed", 20261114)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg.get("cpu_threads", 4))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda")
    model = create_model(cfg).to(device)
    if cfg.get("recover_bn", True):
        sample = random.Random(seed).sample(rows, min(cfg.get("calibration_samples", 128), len(rows)))
        calibration_cfg = {**cfg, "batch_size": cfg.get("calibration_batch_size", cfg.get("batch_size", 64))}
        restore_bn(model, (x for x, *_ in loader(sample, calibration_cfg)))
    parameters = sum(p.numel() for p in model.parameters())
    if cfg.get("auxiliary", True):
        from .ppocr_auxiliary import AuxiliaryReader

        model = AuxiliaryReader(model, Path(cfg["pretrained_dir"]) / "parseq_tiny").to(
            device
        )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 3e-5),
    )
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    best = (-1.0, -1.0)
    selected = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, labels, _ in loader(rows, cfg, True):
            target = torch.tensor(
                [VOCAB.index(c) for s in labels for c in s], device=device
            )
            lengths = torch.tensor([len(s) for s in labels], device=device)
            opt.zero_grad(set_to_none=True)
            with autocast(cfg):
                z = model(x.to(device))
            z = z.float().log_softmax(-1).transpose(0, 1)
            loss = ctc(
                z,
                target,
                torch.full((len(labels),), z.shape[0], device=device, dtype=torch.long),
                lengths,
            )
            if cfg.get("auxiliary", True):
                with autocast(cfg):
                    loss = loss + model.auxiliary_loss(labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite CTC/auxiliary loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            opt.step()
            losses.append(loss.item())
        row = {"epoch": epoch, "loss": sum(losses) / len(losses)}
        if held:
            row["holdout"] = evaluate_model(model, held, cfg)
            m = row["holdout"]
            score = selection_score(m)
        else:
            score = (epoch, epoch)
        if score > best:
            best = score
            selected = epoch
            base = model.base if cfg.get("auxiliary", True) else model
            torch.save(
                {
                    "state_dict": base.state_dict(),
                    "config": {
                        k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()
                    },
                    "epoch": epoch,
                    "inference_parameters": parameters,
                    "training_parameters": sum(p.numel() for p in model.parameters()),
                    "recover_bn": cfg.get("recover_bn", True),
                    "selection_metric": "exact",
                    "selection_policy": EXACT_POLICY,
                    "full_fit": not bool(held),
                },
                folder / "model.pt",
            )
        history.append(row)
        (folder / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
    return selected


def train(config, fit_rows=None, held_rows=None):
    """Explicit manifests may reproduce a prior train-only fold; no val selection."""
    cfg = dict(config)
    if cfg.get("selection_metric", "exact") != "exact":
        raise ValueError("Only literal exact checkpoint selection is supported")
    if cfg.get("epochs", 0) < 1 or cfg.get("batch_size", 0) < 1:
        raise ValueError(
            "Positive epochs and batch_size required; existing checkpoints are never reused"
        )
    if fit_rows is None or held_rows is None:
        if fit_rows is not None or held_rows is not None:
            raise ValueError("Supply both fit_rows and held_rows")
        rows = scan_labeled(Path(cfg["dataset_root"]) / "train")
        fit_rows, held_rows = split_price_holdout(
            rows, cfg["holdout_fraction"], cfg.get("seed", 20261114)
        )
    fit, held = canonical(fit_rows), canonical(held_rows)
    if not fit or not held:
        raise ValueError("Nonempty fit and internal holdout required")
    if {str(Path(r["path"]).resolve()) for r in fit} & {
        str(Path(r["path"]).resolve()) for r in held
    }:
        raise ValueError("Fit and holdout paths overlap")
    for r in fit + held:
        s = r["label"]
        if not s or any(c not in VOCAB[1:] for c in s) or len(s) > 12:
            raise ValueError(
                f'Unsupported/overlength label:{r["image"]}; never truncated'
            )
    from .data import audit_splits

    audit = audit_splits(fit, held)
    if audit["pixel_duplicates_across_splits"]:
        raise ValueError("Duplicate pixels across fit/holdout")
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection_split.json").write_text(
        json.dumps({"fit": fit, "holdout": held, "audit": audit}, indent=2)
    )
    selected = _fit(cfg, fit, held, cfg["epochs"], output / "selection")
    if cfg.get("refit", True):
        _fit(cfg, fit + held, [], selected, output / "full_fit")
    (output / "recipe.json").write_text(
        json.dumps(
            {
                "selected_epoch": selected,
                "fit_n": len(fit),
                "holdout_n": len(held),
                "full_fit_n": len(fit + held) if cfg.get("refit", True) else 0,
                "refit": cfg.get("refit", True),
                "selection_metric": "exact",
                    "selection_policy": EXACT_POLICY,
                "initialization": ("fresh pinned official medium model for selection and fresh full refit; respective FIT-only BN calibration" if cfg.get("refit", True) else "fresh pinned official medium model for FIT-only selection; no full refit; DEV never gradient/calibration"),
                "historical_note": "Preset uses exactinternalholdout selection; historicalwinnerselected15. Current dataset/precision/environment mayselectdifferentepoch.",
            },
            indent=2,
        )
    )
    return output / ("full_fit/model.pt" if cfg.get("refit", True) else "selection/model.pt")


def load_model(checkpoint, device="cpu"):
    c = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = c["config"]
    m = create_model(cfg)
    if c.get("recover_bn", True):
        restore_bn(m)
    m.load_state_dict(c["state_dict"], strict=True)
    cfg = dict(cfg)
    cfg["device"] = device
    return m.eval().to(device), cfg


class TorchRecognizer:
    def __init__(self, checkpoint, device="cpu"):
        self.model, self.config = load_model(checkpoint, device)
        self.config["device"] = device

    @torch.inference_mode()
    def infer(self, images):
        images = list(images)
        if not images:
            return []
        x = torch.stack(
            [torch.from_numpy(preprocess(im, self.config["height"])) for im in images]
        ).to(self.config["device"])
        return decode(self.model(x).argmax(-1).cpu().tolist())
