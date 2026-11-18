"""PP-OCRv6-medium-derived price CTC family; pinned official initialization.

Apache-2.0 derivative; see _vendor/ppocr/NOTICE. No weights bundled. Conversion
uses installed paddle2onnx/onnx2torch; never loads downloaded Python pickles.
"""
from pathlib import Path
import hashlib, json, math, string, subprocess, sys, urllib.request
import numpy as np
from PIL import Image

VOCAB = [""] + list(string.ascii_letters + string.digits + string.punctuation + " ")
REVISION = "e5a92bcbc5cc1b494628e458d267778f0704fd7c"
FILES = {
    "inference.json": "0b2e25e990bd072f1bf77d59d67d508bce6c4bd44af6624e0fb27d6da2cd00e8",
    "inference.pdiparams": "1b01c79a914587933f615569e75de54f2e638ebb5d3f3b3c1b38c24ede8c7319",
    "inference.yml": "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129",
}


VARIANTS = {
    "medium": (REVISION, FILES),
    "tiny": ("0736086f72f666350ebcdc0c3a504eeac89cdfad", {
        "inference.json": "b5b14770c7dcf092781e92f4278a2ae5f95048f08b4b8a04140e88cb2745f147",
        "inference.yml": "66170210bad538e83fff3c4a3867e547d6bf20b50d64b20347c4b913f3034ea1",
        "inference.pdiparams": "bb2f8f54d1e25f28c71b6fa4fe23f5940e159cae27fbee96155c99f822156e57",
    }),
    "small": ("bd619643acac4b9650c040234da8d944476ee3f1", {
        "inference.json": "f0bf53c853937a917affdd74467472167727f8ab0f0f7bded01c4a16c27e46e6",
        "inference.yml": "ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1",
        "inference.pdiparams": "406e1e689c9a7fbb04178007a3fe10cf852afe7bf8bb3bc6dbb9d532b13bd907",
    }),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prepare_pretrained(folder, variant="medium"):
    """Fetch three hash-verified data files and perform local deterministic conversion."""
    import onnx, torch
    from onnx import helper
    from onnxsim import simplify

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    revision, files = VARIANTS[variant]
    for name, digest in files.items():
        target = folder / name
        if not target.exists():
            partial = target.with_suffix(".partial")
            urllib.request.urlretrieve(
                f"https://huggingface.co/PaddlePaddle/PP-OCRv6_{variant}_rec/resolve/{revision}/{name}",
                partial,
            )
            if sha(partial) != digest:
                raise ValueError(f"Official pretrained hash mismatch:{name}")
            partial.replace(target)
        if sha(target) != digest:
            raise ValueError(f"Official pretrained hash mismatch:{name}")
    target = folder / "model.onnx"
    if not target.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "paddle2onnx.command",
                "--model_dir",
                str(folder),
                "--model_filename",
                "inference.json",
                "--params_filename",
                "inference.pdiparams",
                "--save_file",
                str(target),
                "--opset_version",
                "17",
            ],
            check=True,
        )
    trained = folder / "trainable.onnx"
    if not trained.exists():
        m = onnx.load(target)
        keep = []
        for n in m.graph.node:
            if (
                n.op_type == "Constant"
                and len(n.attribute) == 1
                and n.attribute[0].name == "value"
            ):
                t = onnx.TensorProto()
                t.CopyFrom(n.attribute[0].t)
                t.name = n.output[0]
                m.graph.initializer.append(t)
            else:
                keep.append(n)
        del m.graph.node[:]
        m.graph.node.extend(keep)
        m.graph.input[0].type.tensor_type.shape.dim[3].ClearField("dim_param")
        m.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 320
        m, valid = simplify(m)
        assert valid
        m = onnx.shape_inference.infer_shapes(m)
        dims = {
            v.name: [x.dim_value for x in v.type.tensor_type.shape.dim]
            for v in list(m.graph.input) + list(m.graph.value_info)
        }
        for n in m.graph.node:
            a = {t.name: helper.get_attribute_value(t) for t in n.attribute}
            if a.get("auto_pad") == b"SAME_UPPER":
                ih, iw = dims[n.input[0]][-2:]
                kh, kw = a["kernel_shape"]
                sh, sw = a.get("strides", [1, 1])
                dh, dw = a.get("dilations", [1, 1])
                assert ih > 0 and iw > 0
                ph = max(0, ((ih + sh - 1) // sh - 1) * sh + (kh - 1) * dh + 1 - ih)
                pw = max(0, ((iw + sw - 1) // sw - 1) * sw + (kw - 1) * dw + 1 - iw)
                keep = [t for t in n.attribute if t.name not in ("auto_pad", "pads")]
                del n.attribute[:]
                n.attribute.extend(keep)
                n.attribute.append(
                    helper.make_attribute(
                        "pads", [ph // 2, pw // 2, ph - ph // 2, pw - pw // 2]
                    )
                )
        onnx.checker.check_model(m)
        onnx.save(m, trained)
    (folder / "conversion.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "upstream_revision": revision,
                "official_files": files,
                "onnx_sha256": sha(target),
                "trainable_onnx_sha256": sha(trained),
            },
            indent=2,
        )
    )
    return folder


def create_model(config, prepare=True):
    import torch, yaml
    from onnx2torch import convert

    folder = (
        prepare_pretrained(config["pretrained_dir"], config.get("variant", "medium"))
        if prepare
        else Path(config["pretrained_dir"])
    )
    m = convert(str(folder / "trainable.onnx")).eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.LayerNorm):
            mod.normalized_shape = tuple(mod.weight.shape)
    old = (
        [""]
        + yaml.safe_load((folder / "inference.yml").read_text())["PostProcess"][
            "character_dict"
        ]
        + [" "]
    )
    idx = [old.index(c) for c in VOCAB]
    for name, buf in list(m.initializers.named_buffers()):
        if not buf.is_floating_point() or buf.ndim == 0:
            continue
        if buf.shape[-1] == len(old):
            buf = buf[..., idx]
        delattr(m.initializers, name)
        m.initializers.register_parameter(name, torch.nn.Parameter(buf.clone()))
    soft = list(m.graph.nodes)[-2]
    if soft.op != "call_module" or not isinstance(
        m.get_submodule(soft.target), torch.nn.Softmax
    ):
        raise ValueError("Pinned graph final Softmax differs")
    soft.replace_all_uses_with(soft.args[0])
    m.graph.erase_node(soft)
    m.recompile()
    height = int(config.get("height", 128))
    if height % 16 or height < 16:
        raise ValueError("Height must be positive multiple16")
    pool = m.get_submodule("AveragePool/0")
    pool.kernel_size = (height // 16, 2)
    pool.stride = (height // 16, 2)
    return m


def preprocess(image, height=128):
    import cv2

    pixels = np.asarray(image.convert("RGB"))[:, :, ::-1]
    return (
        cv2.resize(pixels, (320, height), interpolation=cv2.INTER_LINEAR)
        .transpose(2, 0, 1)
        .astype(np.float32)
        / 127.5
        - 1
    )


def decode(ids):
    out = []
    for seq in ids:
        previous = -1
        text = []
        for i in seq:
            i = int(i)
            if i and i != previous:
                text.append(VOCAB[i])
            previous = i
        out.append("".join(text))
    return out


def restore_bn(model, calibration_batches=None):
    """Identity-preserving train-time BN reconstruction; calibrate FIT pixels only."""
    import torch

    convs = {
        name: m for name, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)
    }
    stats = {}
    hooks = []
    if calibration_batches is not None:

        def hook(name):
            def collect(mod, x, y):
                v = y.detach().float()
                a, b, n = (
                    v.sum((0, 2, 3)),
                    v.square().sum((0, 2, 3)),
                    v.numel() // v.shape[1],
                )
                if name in stats:
                    stats[name] = [
                        stats[name][0] + a,
                        stats[name][1] + b,
                        stats[name][2] + n,
                    ]
                else:
                    stats[name] = [a, b, n]

            return collect

        for name, m in convs.items():
            hooks.append(m.register_forward_hook(hook(name)))
        model.eval()
        with torch.no_grad():
            for x in calibration_batches:
                model(x.to(next(model.parameters()).device))
        for h in hooks:
            h.remove()
    for name, conv in convs.items():
        bn = torch.nn.BatchNorm2d(conv.out_channels).to(conv.weight.device)
        if name in stats:
            a, b, n = stats[name]
            mean = a / n
            var = (b / n - mean.square()).clamp_min(1e-5)
        else:
            mean = torch.zeros(conv.out_channels, device=conv.weight.device)
            var = torch.ones_like(mean)
        with torch.no_grad():
            bn.running_mean.copy_(mean)
            bn.running_var.copy_(var)
            bn.weight.copy_((var + bn.eps).sqrt())
            bn.bias.copy_(mean)
        bn.eval()
        model.set_submodule(name, torch.nn.Sequential(conv, bn))
    return model
