"""Convert the official PGM and rendered OCRA CLEVR-ART formats to JSONL.

This module never downloads data, renders 3D scenes, or passes symbolic labels to
the model. See docs/DATASETS.md for acquisition and protocol details.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ADAPTER_VERSION = "1"
PGM_REGIMES = (
    "neutral", "interpolation", "extrapolation", "attr.rel.pairs", "attr.rels",
    "attrs.pairs", "attrs.shape.color", "attrs.line.type",
)
PGM_PROMPT = (
    "Complete the 3 by 3 visual reasoning matrix. The first eight panels are "
    "the context, in row-major order; the bottom-right panel is missing. "
    "Choose the panel from A through H that completes the rules across the "
    "matrix. Respond with only the answer letter."
)
CLEVR_PROMPTS = {
    "rmts": (
        "Each labeled scene shows the same source pair in the top row and a "
        "candidate pair in the bottom row. Choose the scene whose bottom pair "
        "has the same same-or-different relation as the top pair. Object identity "
        "is determined by color, shape, size, and material; ignore position, "
        "lighting, and orientation. Respond with only A or B."
    ),
    "idrules": (
        "Each labeled scene contains two rows of three objects. The top row "
        "demonstrates an identity pattern, such as first equals third, last two "
        "equal, or all three equal. The bottom-right object is a candidate "
        "completion. Choose the scene whose bottom row follows the same "
        "identity pattern as its top row. Object identity is determined by "
        "color, shape, size, and material; ignore position, lighting, and "
        "orientation. Respond with only A, B, C, or D."
    ),
}
OFFICIAL_CLEVR_SHA256 = {
    "RMTS_ood_test.npz": "a7fe285f30e702551a20225414b84dc3c315b03010481f1403f9564c942bb78a",
    "RMTS_ood_train.npz": "95821b6a36b0ad7283e80a9e4018d4b3c6b3e28b8fe28f2f2a56273035821921",
    "identity_rules_ood_test.npz": "cc501460a35402e7922803e755c82e5026927d85b77c5e2ab203e255f69cc387",
    "identity_rules_ood_train.npz": "1370aede8207d0c3552dcef68361ff4fd6cda106c11c6e606ab52e099c2da2f0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_subset(items: Iterable[tuple[str, Path]], limit: int | None, seed: int):
    """Stable hash ranking gives reproducible, nested subsets without RNG drift."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive (omit it to include all available examples)")
    ranked = sorted(items, key=lambda item: (
        hashlib.sha256(f"{seed}\0{item[0]}".encode()).hexdigest(), item[0]
    ))
    if not ranked:
        raise ValueError("No examples match the requested task, regime and split")
    ids = [item[0] for item in ranked]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate example IDs: point --root at one dataset copy")
    return ranked[:limit], ranked


def _font():
    # Pillow's bundled font avoids a dependency on machine-specific system fonts.
    try:
        return ImageFont.load_default(size=20)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        base = Image.new("RGBA", image.size, "white")
        return Image.alpha_composite(base, image).convert("RGB")
    return image.convert("RGB")


def render_pgm(panels: np.ndarray) -> Image.Image:
    """No target argument: the missing cell is always blank and options stay fixed."""
    if panels.shape != (16, 160, 160) or panels.dtype != np.uint8:
        raise ValueError("PGM renderer requires 16 uint8 panels of size 160x160")
    gap, label, panel = 12, 30, 160
    width = 4 * panel + 5 * gap
    matrix_left = (width - (3 * panel + 2 * gap)) // 2
    matrix_top = 42
    choices_top = matrix_top + 3 * (panel + gap) + 42
    sheet = Image.new("RGB", (width, choices_top + 2 * (label + panel + gap) + gap), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 10), "MATRIX", fill="black", font=_font())
    for i in range(9):
        x = matrix_left + (i % 3) * (panel + gap)
        y = matrix_top + (i // 3) * (panel + gap)
        if i < 8:
            sheet.paste(Image.fromarray(panels[i]).convert("RGB"), (x, y))
        else:
            draw.rectangle((x, y, x + panel - 1, y + panel - 1), fill="#eeeeee", outline="black")
            draw.text((x + 72, y + 68), "?", fill="black", font=_font())
    draw.text((gap, choices_top - 34), "ANSWER CHOICES", fill="black", font=_font())
    for i in range(8):
        x = gap + (i % 4) * (panel + gap)
        y = choices_top + (i // 4) * (label + panel + gap)
        draw.text((x, y), chr(65 + i), fill="black", font=_font())
        sheet.paste(Image.fromarray(panels[8 + i]).convert("RGB"), (x, y + label))
    return sheet


def render_clevr(candidates: list[Image.Image]) -> Image.Image:
    """Preserve each complete source/candidate scene at its original resolution."""
    if len(candidates) not in (2, 4):
        raise ValueError("CLEVR-ART requires exactly two or four candidate scenes")
    if len({image.size for image in candidates}) != 1:
        raise ValueError("CLEVR-ART candidate scenes must have identical dimensions")
    w, h = candidates[0].size
    if w < 1 or h < 1:
        raise ValueError("Empty candidate image")
    gap, label, title = 16, 32, 38
    rows = len(candidates) // 2
    sheet = Image.new("RGB", (2 * w + 3 * gap, title + rows * (h + label + gap) + gap), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 8), "CANDIDATE SCENES", fill="black", font=_font())
    for i, candidate in enumerate(candidates):
        x = gap + (i % 2) * (w + gap)
        y = title + (i // 2) * (h + label + gap)
        draw.text((x, y), chr(65 + i), fill="black", font=_font())
        sheet.paste(_rgb(candidate), (x, y + label))
    return sheet


def _int_target(value: np.ndarray, count: int, location: str) -> int:
    target = np.asarray(value)
    if target.size != 1 or target.dtype.kind not in "iu":
        raise ValueError(f"{location}: expected one integer target")
    number = int(target.reshape(-1)[0])
    if not 0 <= number < count:
        raise ValueError(f"{location}: target {number} outside 0..{count - 1}")
    return number


def _write_manifest(output: Path, selected, pool, loader: Callable, provenance: dict) -> Path:
    output = Path(output)
    if output.suffix != ".jsonl":
        raise ValueError("--output must be a .jsonl file")
    assets = output.with_name(output.stem + "_images")
    sidecar = output.with_name(output.name + ".provenance.json")
    for path in (output, assets, sidecar):
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing output: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as tmp:
        tmp = Path(tmp)
        image_dir = tmp / assets.name
        image_dir.mkdir()
        labels: Counter = Counter()
        with (tmp / output.name).open("w", encoding="utf-8") as manifest:
            for identifier, path in selected:
                row, image = loader(identifier, path)
                filename = hashlib.sha256(identifier.encode()).hexdigest()[:24] + ".png"
                target = image_dir / filename
                image.save(target, format="PNG")
                row["images"] = [f"{assets.name}/{filename}"]
                row["metadata"].update({
                    "adapter_version": ADAPTER_VERSION,
                    "contact_sheet_sha256": sha256_file(target),
                    "contact_sheet_size": list(image.size),
                    "selection_seed": provenance["seed"],
                    "available_pool_size": len(pool),
                })
                manifest.write(json.dumps(row, sort_keys=True) + "\n")
                labels[row["answer"]] += 1
        provenance.update({
            "adapter_version": ADAPTER_VERSION,
            "selected_count": len(selected), "available_pool_size": len(pool),
            "pool_ids_sha256": hashlib.sha256("\n".join(sorted(x[0] for x in pool)).encode()).hexdigest(),
            "selection_method": "sha256(seed + NUL + canonical_id), ascending",
            "selected_ids": [x[0] for x in selected],
            "answer_counts": dict(sorted(labels.items())),
            "manifest_sha256": sha256_file(tmp / output.name),
            "numpy_version": np.__version__,
            "pillow_version": Image.__version__,
        })
        (tmp / sidecar.name).write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        image_dir.rename(assets)
        (tmp / sidecar.name).rename(sidecar)
        (tmp / output.name).rename(output)
    return output


def prepare_pgm(root: Path, output: Path, regime: str, split: str,
                limit: int | None = None, seed: int = 0) -> Path:
    root = Path(root)
    if regime not in PGM_REGIMES or split not in ("train", "val", "test"):
        raise ValueError("Unsupported PGM regime or split")
    if not root.is_dir():
        raise FileNotFoundError(f"PGM root does not exist: {root}")
    pattern = re.compile(rf"PGM_{re.escape(regime)}_{split}_(\d+)\.npz\Z")
    items = []
    for path in root.rglob(f"PGM_{regime}_{split}_*.npz"):
        match = pattern.fullmatch(path.name)
        if match:
            items.append((f"pgm/{regime}/{split}/{int(match[1])}", path))
    selected, pool = select_subset(items, limit, seed)

    def load(identifier, path):
        with np.load(path, allow_pickle=False) as data:
            if not {"image", "target"}.issubset(data.files):
                raise ValueError(f"{path}: missing image or target array")
            raw = data["image"]
            if raw.shape not in ((16, 160, 160), (160, 160, 16)):
                raise ValueError(f"{path}: invalid image shape {raw.shape}")
            if raw.dtype.kind not in "iu" or raw.min() < 0 or raw.max() > 255:
                raise ValueError(f"{path}: image must contain integer pixels in 0..255")
            # Official correction explicitly requires reshape, NOT transpose.
            panels = raw.reshape(16, 160, 160).astype(np.uint8)
            target = _int_target(data["target"], 8, str(path))
        row = {
            "id": identifier, "dataset": "pgm", "task": "matrix-completion",
            "regime": regime, "split": split, "prompt": PGM_PROMPT,
            "choices": list("ABCDEFGH"), "answer": chr(65 + target),
            "metadata": {"source_path": str(path.relative_to(root)),
                         "source_sha256": sha256_file(path),
                         "source_image_shape": list(raw.shape),
                         "presentation": "matrix-3x3-options-2x4-v1"},
        }
        return row, render_pgm(panels)

    return _write_manifest(Path(output), selected, pool, load, {
        "dataset": "pgm", "regime": regime, "split": split, "seed": seed,
        "requested_limit": limit, "source_root": str(root.resolve()),
        "upstream": "https://github.com/google-deepmind/abstract-reasoning-matrices",
        "data_terms": "Non-commercial research only (upstream data terms)",
    })


def _validate_clevr_arrays(x: np.ndarray, y: np.ndarray, task: str, split: str):
    objects = 6 if task == "rmts" else 9
    if x.ndim != 3 or x.shape[1:] != (objects, 4) or x.dtype.kind not in "iu":
        raise ValueError(f"CLEVR-ART x must have shape (N, {objects}, 4) and integer features")
    if y.shape != (len(x),) or y.dtype.kind not in "iu" or len(x) == 0:
        raise ValueError("CLEVR-ART y must have shape (N,) with integer targets")
    ranges = ((0, 3), (0, 0), (0, 1), (0, 0)) if split == "train" else ((4, 7), (1, 2), (2, 2), (1, 1))
    for i, (low, high) in enumerate(ranges):
        if (x[:, :, i] < low).any() or (x[:, :, i] > high).any():
            raise ValueError(f"CLEVR-ART {split}_ood has out-of-domain feature {i}; check that NPZ and split match")
    if (y < 0).any() or (y >= (2 if task == "rmts" else 4)).any():
        raise ValueError("CLEVR-ART target index is out of range")


def _validate_clevr_answer(x: np.ndarray, target: int, task: str):
    if task == "rmts":
        source_same = np.array_equal(x[0], x[1])
        valid = [np.array_equal(x[2 + 2 * i], x[3 + 2 * i]) == source_same for i in range(2)]
    else:
        def pattern(row):
            return tuple(np.array_equal(row[a], row[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
        source = pattern(x[:3])
        valid = [pattern(np.stack((x[3], x[4], x[5 + i]))) == source for i in range(4)]
    if sum(valid) != 1 or not valid[target]:
        raise ValueError("CLEVR-ART symbolic rule and target disagree, or the answer is ambiguous")


def prepare_clevr_art(root: Path, output: Path, task: str, regime: str, split: str,
                      limit: int | None = None, seed: int = 0) -> Path:
    root = Path(root)
    if task not in ("rmts", "idrules") or regime != "ood" or split not in ("train", "test"):
        raise ValueError("Supported CLEVR-ART format: task rmts|idrules, regime ood, split train|test")
    label_name = f"{'RMTS' if task == 'rmts' else 'identity_rules'}_ood_{split}.npz"
    label_path = root / label_name
    scene_root = root / "output" / f"{task}_images" / f"{split}_ood"
    if not label_path.is_file() or not scene_root.is_dir():
        raise FileNotFoundError(
            f"Expected {label_path} and {scene_root}/prob_N/CLEVR_0.png. "
            "Official NPZ files contain symbols, not rendered images. See docs/DATASETS.md."
        )
    with np.load(label_path, allow_pickle=False) as data:
        if not {"x", "y"}.issubset(data.files):
            raise ValueError(f"{label_path}: expected x and y arrays")
        x, y = data["x"], data["y"]
    _validate_clevr_arrays(x, y, task, split)
    digest = sha256_file(label_path)
    candidates = 2 if task == "rmts" else 4
    items = []
    for directory in scene_root.iterdir():
        match = re.fullmatch(r"prob_(\d+)", directory.name)
        if directory.is_dir() and match:
            index = int(match[1])
            if index >= len(y):
                raise ValueError(f"{directory}: problem index exceeds the label array")
            # Fail on incomplete renders, rather than silently biasing the pool.
            for choice in range(candidates):
                if not (directory / f"CLEVR_{choice}.png").is_file():
                    raise FileNotFoundError(f"Incomplete render: {directory}/CLEVR_{choice}.png")
            items.append((f"clevr-art/{task}/{regime}/{split}/{index}", directory))
    selected, pool = select_subset(items, limit, seed)

    def load(identifier, directory):
        index = int(identifier.rsplit("/", 1)[1])
        target = int(y[index])
        _validate_clevr_answer(x[index], target, task)
        paths = [directory / f"CLEVR_{i}.png" for i in range(candidates)]
        scenes = []
        for path in paths:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ValueError(f"{path}: source must be PNG")
                scenes.append(_rgb(image))
        row = {
            "id": identifier, "dataset": "clevr-art", "task": task,
            "regime": regime, "split": split, "prompt": CLEVR_PROMPTS[task],
            "choices": list("ABCD"[:candidates]), "answer": chr(65 + target),
            "metadata": {
                "source_paths": [str(path.relative_to(root)) for path in paths],
                "source_sha256": [sha256_file(path) for path in paths],
                "label_source": label_name, "label_source_sha256": digest,
                "official_label_file": digest == OFFICIAL_CLEVR_SHA256[label_name],
                "source_image_size": list(scenes[0].size),
                "presentation": "complete-candidate-scenes-2col-v1",
            },
        }
        return row, render_clevr(scenes)

    return _write_manifest(Path(output), selected, pool, load, {
        "dataset": "clevr-art", "task": task, "regime": regime, "split": split,
        "seed": seed, "requested_limit": limit, "source_root": str(root.resolve()),
        "upstream": "https://github.com/Shanka123/OCRA/tree/f15fa625d7e5b4458d82c24108b26419037d940d",
        "symbolic_examples": len(y), "label_source_sha256": digest,
        "official_label_file": digest == OFFICIAL_CLEVR_SHA256[label_name],
        "partial_render_pool": len(pool) != len(y),
    })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="dataset", required=True)
    for dataset in ("pgm", "clevr-art"):
        child = commands.add_parser(dataset)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True, help="New JSONL path; existing outputs are never replaced")
        child.add_argument("--regime", required=True, choices=PGM_REGIMES if dataset == "pgm" else ("ood",))
        child.add_argument("--split", required=True, choices=("train", "val", "test") if dataset == "pgm" else ("train", "test"))
        child.add_argument("--limit", type=int, help="Number to select; default is all available examples")
        child.add_argument("--seed", type=int, default=0)
        if dataset == "clevr-art":
            child.add_argument("--task", choices=("rmts", "idrules"), required=True)
    args = vars(parser.parse_args(argv))
    dataset = args.pop("dataset")
    try:
        output = (prepare_pgm if dataset == "pgm" else prepare_clevr_art)(**args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Data preparation failed: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
