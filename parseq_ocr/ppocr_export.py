"""Portable PP price CTC singlegraph ONNX and TensorRT export."""
from pathlib import Path
import hashlib, json


def fix_dynamic_shapes(path, model):
    """Restore original Shape ops constant-folded by onnx2torch tracing."""
    import onnx
    from onnx import helper

    graph = onnx.load(path)
    changed = []
    for fx in model.graph.nodes:
        if fx.op != "call_module":
            continue
        module = model.get_submodule(fx.target)
        if module.__class__.__name__ != "OnnxShape":
            continue
        name = "/m/" + fx.target + "/Constant"
        node = next(
            (n for n in graph.graph.node if n.name == name and n.op_type == "Constant"),
            None,
        )
        if node is None:
            continue
        source = fx.args[0]
        prefix = "/m/" + source.target + "/"
        producers = [n for n in graph.graph.node if n.name.startswith(prefix)]
        if len(producers) != 1:
            raise ValueError("Unexpected originalShape source graph")
        node.op_type = "Shape"
        del node.attribute[:]
        del node.input[:]
        node.input.append(producers[0].output[0])
        node.name = name.replace("/Constant", "/DynamicShape")
        if module._start:
            node.attribute.append(helper.make_attribute("start", module._start))
        if module._end is not None:
            node.attribute.append(helper.make_attribute("end", module._end))
        changed.append(fx.target)
    onnx.checker.check_model(graph)
    onnx.save(graph, path)
    return changed


def export_onnx(checkpoint, output):
    import torch, numpy as np, onnxruntime as ort
    from .ppocr_training import load_model
    from .ppocr_models import VOCAB

    torch.set_num_threads(4)
    model, cfg = load_model(checkpoint)
    model.eval()

    class Graph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, pixels):
            return self.m(pixels).argmax(-1).to(torch.int32)

    g = Graph().eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "model.onnx"
    x = torch.randn(2, 3, cfg["height"], 320)
    torch.onnx.export(
        g,
        (x,),
        str(path),
        input_names=["pixels"],
        output_names=["ids"],
        dynamic_axes={"pixels": {0: "batch"}, "ids": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    changes = fix_dynamic_shapes(path, model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    for n in [1, 2, 3, 8]:
        x = torch.randn(n, 3, cfg["height"], 320)
        with torch.inference_mode():
            expected = g(x).numpy()
        actual = session.run(None, {"pixels": x.numpy()})[0]
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"ONNX argmax parity failed atbatch{n}")
    (output / "parity.json").write_text(
        json.dumps(
            {
                "fp32_argmax_parity": True,
                "batches": [1, 2, 3, 8],
                "dynamic_shape_repairs": changes,
                "scope": "randomsyntheticinputs; notdatasetaccuracy orTensorRTparity",
            },
            indent=2,
        )
    )
    metadata = {
        "family": "ppocr_price",
        "height": cfg["height"],
        "width": 320,
        "vocabulary": VOCAB,
        "resize": "stretch",
        "input": "BGR OpenCVLINEAR normalized[-1,1]",
        "output": "CTCids; collapse repeats before removingblank0; strip outerwhitespace",
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "initialization": "Pinnedofficial PP-OCRv6_medium_rec; custom BN+ARtraining, ASCIIhead; ARdiscarded",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return output


def build_engine(deployment, max_batch=32, force=False):
    import tensorrt as trt

    folder = Path(deployment)
    metadata = json.loads((folder / "metadata.json").read_text())
    source = folder / "model.onnx"
    target = folder / "model.engine"
    if not 1 <= max_batch <= 32:
        raise ValueError("Validated batch range1..32")
    if target.exists() and not force:
        raise FileExistsError("Engine exists; force rebuild for currenttarget")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(source.read_bytes()):
        raise RuntimeError(
            "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt.PreviewFeature, "RUNTIME_ACTIVATION_RESIZE_10_10"):
        cfg.set_preview_feature(
            trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10, True
        )
    profile = builder.create_optimization_profile()
    h, w = metadata["height"], metadata["width"]
    profile.set_shape(
        "pixels", (1, 3, h, w), (min(8, max_batch), 3, h, w), (max_batch, 3, h, w)
    )
    cfg.add_optimization_profile(profile)
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    target.write_bytes(blob)
    (folder / "build.json").write_text(
        json.dumps(
            {
                "tensorrt": trt.__version__,
                "max_batch": max_batch,
                "precision": "FP16allowed",
                "activation_resize": True,
                "onnx_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "engine_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    return target
