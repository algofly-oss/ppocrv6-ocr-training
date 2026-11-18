"""Pinned upstream models and verified official pretrained weights."""
from pathlib import Path
import hashlib
import sys
import urllib.request
import torch

VENDOR = Path(__file__).resolve().parent / "_vendor" / "parseq"
sys.path.insert(0, str(VENDOR))
from strhub.models.utils import create_model

WEIGHTS = {
    "parseq": ("parseq-bb5792a6.pt", "bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00"),
    "parseq-tiny": ("parseq_tiny-e7a21b54.pt", "e7a21b543c98e67414a584c93b1dbb71c26e9463ae4aaa391d54e833660a4711"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pretrained(name, folder):
    filename, expected = WEIGHTS[name]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    if not target.exists():
        partial = target.with_suffix(".partial")
        url = f"https://github.com/baudm/parseq/releases/download/v1.0.0/{filename}"
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as output:
            while chunk := source.read(1024*1024):
                output.write(chunk)
        if expected and sha256(partial) != expected:
            raise ValueError("Pretrained checksum mismatch; partial download was not installed")
        partial.replace(target)
    if expected and sha256(target) != expected:
        raise ValueError(f"Pretrained checksum mismatch: {target}")
    return target


def resize_encoder(model, image_size):
    """Interpolate positional embeddings when transferring between input geometries."""
    h, w = image_size
    encoder = model.model.encoder
    old_grid = encoder.patch_embed.grid_size
    grid = (h//4, w//8)
    if tuple(grid) != tuple(old_grid):
        position = encoder.pos_embed.reshape(1, *old_grid, -1).permute(0, 3, 1, 2)
        position = torch.nn.functional.interpolate(position, size=grid, mode="bicubic", align_corners=False)
        encoder.pos_embed = torch.nn.Parameter(position.permute(0, 2, 3, 1).reshape(1, grid[0]*grid[1], -1))
    encoder.patch_embed.img_size = (h, w)
    encoder.patch_embed.grid_size = grid
    encoder.patch_embed.num_patches = grid[0]*grid[1]


def create_system(config, checkpoint=None, pretrained=True):
    model = create_model(config["model"], False)
    if checkpoint is not None:
        resize_encoder(model, config["image_size"])
        model.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    elif config.get("initial_checkpoint"):
        resize_encoder(model, config["initial_image_size"])
        model.model.load_state_dict(torch.load(config["initial_checkpoint"], map_location="cpu", weights_only=True), strict=True)
        resize_encoder(model, config["image_size"])
    else:
        if pretrained:
            path = download_pretrained(config["model"], config["pretrained_dir"])
            model.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        resize_encoder(model, config["image_size"])
    model.model.max_label_length = config.get("max_label_length", 25)
    model.model.decode_ar = config["autoregressive"]
    model.model.refine_iters = config["refinements"]
    return model
