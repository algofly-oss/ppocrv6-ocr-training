"""Single-crop timing; TensorRT path does not import Torch or initialize its context."""
import os
import resource
import threading
import time
import numpy as np
from PIL import Image
from .ppocr_data import scan_labeled


def benchmark(
    config,
    backend,
    folder,
    repeats=3,
    predecoded=False,
    checkpoint=None,
    batch_size=1,
    deployment=None,
):
    if not 1 <= batch_size <= 32:
        raise ValueError("Validated batch range1..32")

    def make_recognizer(config, backend, checkpoint):
        if backend == "tensorrt":
            from .ppocr_runtime import TensorRTRecognizer

            return TensorRTRecognizer(deployment or config["output_dir"] / "deployment")
        from .ppocr_training import TorchRecognizer

        return TorchRecognizer(
            checkpoint or config["output_dir"] / "full_fit/model.pt",
            config.get("device", "cuda"),
        )

    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = scan_labeled(folder)
    if not rows:
        raise ValueError("No annotated images to benchmark")

    def read(row):
        with Image.open(row["path"]) as image:
            return image.convert("RGB")

    decoded = [read(row) for row in rows] if predecoded else None
    stop = threading.Event()
    memory = {"sampled_peak_bytes": None, "steady_bytes": None, "error": None}
    nv = None
    sampler = None
    try:
        import pynvml as nv

        nv.nvmlInit()
        # Runtime currently targets CUDA device zero; respect CUDA_VISIBLE_DEVICES.
        physical = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        handle = (
            nv.nvmlDeviceGetHandleByUUID(physical)
            if physical.startswith("GPU-")
            else nv.nvmlDeviceGetHandleByIndex(int(physical))
        )

        def sample():
            values = [
                int(p.usedGpuMemory)
                for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)
                if p.pid == os.getpid() and 0 <= int(p.usedGpuMemory) < (1 << 50)
            ]
            value = sum(values)
            memory["steady_bytes"] = value
            memory["sampled_peak_bytes"] = max(memory["sampled_peak_bytes"] or 0, value)

        def run():
            while not stop.is_set():
                try:
                    sample()
                except Exception as exc:
                    memory["error"] = str(exc)
                    return
                stop.wait(0.02)

        sampler = threading.Thread(target=run, daemon=True)
        sampler.start()
    except Exception as exc:
        memory["error"] = str(exc)
    recognizer = None
    try:
        begin = time.perf_counter()
        recognizer = make_recognizer(config, backend, checkpoint)
        load_seconds = time.perf_counter() - begin
        first = decoded[0] if decoded is not None else read(rows[0])
        for _ in range(5):
            recognizer.infer([first] * batch_size)
        # infer returns host text, so every recorded call includes completion.
        timings = []
        for _ in range(repeats):
            for index in range(0, len(rows), batch_size):
                batch = rows[index : index + batch_size]
                begin = time.perf_counter()
                images = (
                    decoded[index : index + batch_size]
                    if decoded is not None
                    else [read(row) for row in batch]
                )
                recognizer.infer(images)
                timings.extend(
                    [(time.perf_counter() - begin) * 1000 / len(batch)] * len(batch)
                )
        if sampler and not memory["error"]:
            sample()
        return {
            "backend": backend,
            "batch_size": batch_size,
            "latency_unit": "elapsedbatchtime/actualbatchimages; batch32 reports throughput amortization, notsinglecroplatency",
            "unique_images": len(rows),
            "timed_images": len(rows) * repeats,
            "timed_batches": ((len(rows) + batch_size - 1) // batch_size) * repeats,
            "warmup": 5,
            "mean_ms": float(np.mean(timings)),
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "images_per_second": 1000 / float(np.mean(timings)),
            "load_seconds": load_seconds,
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "gpu_memory": dict(memory),
            "memory_sample_interval_seconds": 0.02,
            "scope": (
                "Predecoded RGB to text; excludes image-file decoding"
                if predecoded
                else "Image-file decoding, preprocessing, inference and text decoding"
            ),
            "exclusions": "Network, detection, model loading and initial warmup",
        }
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=1)
        if nv:
            try:
                nv.nvmlShutdown()
            except Exception:
                pass
        if recognizer is not None and hasattr(recognizer, "close"):
            recognizer.close()
