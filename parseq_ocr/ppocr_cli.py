"""Independent PP-derived price workflow: python -m parseq_ocr.ppocr_cli."""
import argparse, json
from pathlib import Path
from PIL import Image
from .ppocr_config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=[
            "prepare",
            "train",
            "evaluate",
            "infer",
            "export",
            "build",
            "benchmark",
        ],
    )
    p.add_argument("--config", default="configs/ppocr_price.yaml")
    p.add_argument("--recipe", choices=["legacy", "corrected_fixed_bn_v1", "selection_only_precursor_v1"], default=None,
                   help="Explicit optional training recipe; config default is unchanged")
    p.add_argument("--smoke-steps", type=int, choices=[2], default=None,
                   help="Corrected recipe only: two optimizer updates in a separate smoke folder")
    p.add_argument("--checkpoint")
    p.add_argument("--image")
    p.add_argument("--manifest")
    p.add_argument("--split", default="val")
    p.add_argument(
        "--runtime",
        "--backend",
        dest="runtime",
        choices=["torch", "tensorrt"],
        default="torch",
    )
    p.add_argument("--deployment")
    p.add_argument("--force", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--predecoded", action="store_true")
    a = p.parse_args()
    if not 1 <= a.batch_size <= 32:
        p.error("--batch-size must be1..32")
    cfg = load_config(a.config)
    if a.recipe is not None:
        cfg["recipe"] = a.recipe
    if a.smoke_steps is not None and (a.command != "train" or cfg.get("recipe") != "corrected_fixed_bn_v1"):
        p.error("--smoke-steps applies only to train with corrected_fixed_bn_v1")
    checkpoint = (
        Path(a.checkpoint) if a.checkpoint else cfg["output_dir"] / ("selection/model.pt" if cfg.get("recipe") == "selection_only_precursor_v1" else "full_fit/model.pt")
    )
    deployment = (
        Path(a.deployment) if a.deployment else cfg["output_dir"] / "deployment"
    )
    if a.command == "prepare":
        from .ppocr_models import prepare_pretrained

        print(prepare_pretrained(cfg["pretrained_dir"]))
    elif a.command == "train":
        if cfg.get("recipe") == "selection_only_precursor_v1":
            from .ppocr_precursor import train
            print(json.dumps(train(cfg), indent=2))
        elif cfg.get("recipe") == "corrected_fixed_bn_v1":
            from .ppocr_corrected_training import train
            print(json.dumps(train(cfg, smoke_steps=a.smoke_steps), indent=2))
        else:
            from .ppocr_training import train
            print(train(cfg))
    elif a.command == "export":
        from .ppocr_export import export_onnx

        print(export_onnx(checkpoint, deployment))
    elif a.command == "build":
        from .ppocr_export import build_engine

        print(build_engine(deployment, force=a.force))
    elif a.command == "benchmark":
        from .ppocr_benchmark import benchmark

        result = benchmark(
            cfg,
            a.runtime,
            cfg["dataset_root"] / a.split,
            a.repeats,
            a.predecoded,
            checkpoint,
            a.batch_size,
            deployment,
        )
        out = (
            cfg["output_dir"]
            / f"benchmark_{a.runtime}_b{a.batch_size}_{'predecoded' if a.predecoded else 'encoded'}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    else:
        if a.runtime == "torch":
            from .ppocr_training import TorchRecognizer

            model = TorchRecognizer(checkpoint, cfg.get("device", "cuda"))
        else:
            from .ppocr_runtime import TensorRTRecognizer

            model = TensorRTRecognizer(deployment)
        try:
            if a.command == "infer":
                if not a.image:
                    p.error("--image required")
                with Image.open(a.image) as im:
                    text = model.infer([im.convert("RGB")])[0]
                print(json.dumps({"text": text}))
            else:
                from .ppocr_data import scan_labeled
                from .ppocr_data import canonical
                from .metrics import score_pairs

                rows = canonical(
                    json.loads(Path(a.manifest).read_text())
                    if a.manifest
                    else scan_labeled(cfg["dataset_root"] / a.split)
                )
                pred = []
                if a.manifest:
                    for row in rows:
                        path = Path(row["path"])
                        row["path"] = str(
                            path
                            if path.is_absolute()
                            else Path(a.manifest).resolve().parent / path
                        )
                for at in range(0, len(rows), a.batch_size):
                    images = []
                    for row in rows[at : at + a.batch_size]:
                        with Image.open(row["path"]) as im:
                            images.append(im.convert("RGB"))
                    pred.extend(model.infer(images))
                labels = [r["label"] for r in rows]
                result = {
                    **score_pairs(labels, pred, "none"),
                    "runtime": a.runtime,
                    "batch_size": a.batch_size,
                    "precision": "nativeFP32"
                    if a.runtime == "torch"
                    else "TensorRT FP16allowed",
                }
                out = (
                    cfg["output_dir"]
                    / f"evaluation_{a.runtime}_{a.split}_b{a.batch_size}"
                )
                out.mkdir(parents=True, exist_ok=True)
                (out / "metrics.json").write_text(json.dumps(result, indent=2))
                (out / "predictions.jsonl").write_text(
                    "".join(
                        json.dumps(
                            {"image": r["image"], "gt": r["label"], "prediction": s}
                        )
                        + "\n"
                        for r, s in zip(rows, pred)
                    )
                )
                print(json.dumps(result, indent=2))
        finally:
            if hasattr(model, "close"):
                model.close()


if __name__ == "__main__":
    main()
