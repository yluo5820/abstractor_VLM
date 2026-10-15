"""Load the authors' architecture without importing their training launchers."""
import hashlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_COMMIT = "2219512022157f8f8e5e357d8d29a44f6fbe6691"
CANONICAL_COMMIT = "100ca0bc87bca519e85f3d134084c1078caf27ef"
SPECS = {
    "pgm": {
        "source": "pgm/slot_abstractor_pgm.py",
        "source_sha256": "f70676665fe6bfc4cbe75437db43a8d859f41302759142b7c490b8fa22698d8c",
        "encoder_sha256": "bec5621708498348837626f06081c52681ca133719b6afe73f730997e47b52cd",
        "size": 80, "slots": 16, "dim": 32,
    },
    "clevr": {
        "source": "clevr-art/slot_abstractor_clevr.py",
        "source_sha256": "bf843b5bb0bc09cc5ebb7ca2abb49ff9dadf0d423a26bf9e09a4ddec98405855",
        "encoder_sha256": "fc4d0e88f5c6425e7ed38fa79b0cf6287b19de73a7691496066b4bcb2225099d",
        "size": 128, "slots": 7, "dim": 64,
    },
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_module(root, family):
    spec = SPECS[family]
    path = Path(root) / spec["source"]
    if digest(path) != spec["source_sha256"]:
        raise ValueError(f"Upstream architecture hash mismatch: {path}")
    source = path.read_text()
    # The logger is unused. Position grids were hard-coded to CUDA and not buffers.
    # Neither compatibility change alters trained parameters or network equations.
    source = source.replace("from util import log", "import logging\nlog = logging.getLogger(__name__)")
    source = source.replace("torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).cuda()",
                            "torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))")
    module = ModuleType(f"pinned_slot_abstractor_{family}")
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def unwrapped_state(state):
    """Strip DDP's prefix only when all keys actually have it."""
    prefixed = [key.startswith("module.") for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Mixed DDP and ordinary checkpoint keys")
    return {key[7:] if all(prefixed) else key: value for key, value in state.items()}


class NativeModel(nn.Module):
    def __init__(self, upstream, encoder_checkpoint, task, device="cpu"):
        super().__init__()
        if task not in ("pgm", "rmts", "idrules"):
            raise ValueError(task)
        self.task = task
        self.family = "pgm" if task == "pgm" else "clevr"
        self.spec = SPECS[self.family]
        module = checked_module(upstream, self.family)
        if digest(encoder_checkpoint) != self.spec["encoder_sha256"]:
            raise ValueError("Pretrained encoder hash mismatch")
        self.encoder = module.SlotAttentionAutoEncoder(
            (self.spec["size"],) * 2, self.spec["slots"], 3, self.spec["dim"])
        for child in self.encoder.modules():
            if isinstance(child, module.SoftPositionEmbed):
                grid = child.grid
                del child.grid
                child.register_buffer("grid", grid, persistent=False)
        weights = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(unwrapped_state(weights["slot_model_state_dict"]), strict=True)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.head = module.scoring_model(SimpleNamespace(apply_context_norm=True),
                                        self.spec["dim"], 24, 8, 512, self.spec["slots"])
        self.to(device)

    @property
    def device(self):
        return next(self.head.parameters()).device

    @torch.no_grad()
    def encode(self, images):
        """Original forward prefix; omit the unused reconstruction decoder only."""
        x, feat, pos = self.encoder.encoder_cnn(images)
        # Upstream constructs this identity-affine LayerNorm on every call.
        x = nn.LayerNorm(x.shape[1:]).to(self.device)(x)
        x = self.encoder.fc2(F.relu(self.encoder.fc1(x)))
        _, features, positions, _ = self.encoder.slot_attention(x, feat, pos, self.device)
        return features, positions

    def features(self, panels):
        features, positions = [], []
        # Match the authors' panel-major call order, including random slot draws.
        for index in range(panels.shape[1]):
            f, p = self.encode(panels[:, index].to(self.device))
            features.append(f)
            positions.append(p)
        return torch.stack(features, 1), torch.stack(positions, 1)

    def score(self, features, positions):
        if self.task == "pgm":
            return self.head(features[:, :8], positions[:, :8],
                             features[:, 8:], positions[:, 8:], self.device)
        return self.head(features, positions, self.device)

    def forward(self, panels):
        return self.score(*self.features(panels))


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_compute():
    import os
    import socket
    if "SLURM_JOB_ID" not in os.environ or "login" in socket.gethostname().lower():
        raise RuntimeError("Run this command in a Slurm GPU allocation, not a login shell")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
