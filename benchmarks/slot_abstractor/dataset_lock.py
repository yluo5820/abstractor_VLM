"""Exact test inputs already used by the three-model zero-shot campaign."""
from pathlib import Path

from .model import digest

PGM_REVISION = "195a03a7e3bc6d6a349b55986afba5319929c3bf"
# Same HuggingFaceM4/PGM LFS hashes as vlm_bench/pgm_mirror.py.
PGM_SHARDS = {
    "test-00000-of-00009.parquet": "1270f82193c76e46872b79dfd73fc3b303d61c9777a9a8b4353fb261e0f257b6",
    "test-00001-of-00009.parquet": "b324364f95c4423c8b238cc57d3bfcecf9ba9fd121eeb56d303864ad1c57303c",
    "test-00002-of-00009.parquet": "b483f216ccb16e27f73290b544602770638a576ecc82e0b5d44e1986e52c0cf5",
    "test-00003-of-00009.parquet": "539433e82c5c0a57c921397455ea02dc3de00f0cbd73f5afdc67782009629664",
    "test-00004-of-00009.parquet": "75ba00cf11ad55850c4932821940e7da2db2ef97e3a612d180df2622a90961d7",
    "test-00005-of-00009.parquet": "b54b4e1f309c9459dbe951cc5948bf8b9495d65786ffff6f08fe1ff99bbb9046",
    "test-00006-of-00009.parquet": "a8d2174cb4c5dbf0a43e841c4db54cbdaf68724f24015441b9052983c1f73b09",
    "test-00007-of-00009.parquet": "9be6a3ab7b30ff4f3591d5067034c96609be9a1d71cc685fc811b909d699a48b",
    "test-00008-of-00009.parquet": "6ea46da021635cab7e603a5169dd3eb28e361844eaf2070915ca2b190f7c684c",
}
CLEVR_MANIFESTS = {
    "rmts": "f2dc09cad2fa644db952a0df96e40a1e59f91654194771f1e9239f655f4f5496",
    "idrules": "320142264ae614c3660a1960cd2b0bef891e38a99faa49b52dcfc0a2c58339fc",
}


def validate_sources(task, paths):
    paths = list(map(Path, paths))
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("Duplicate input files")
    result = {}
    for path in paths:
        expected = PGM_SHARDS.get(path.name) if task == "pgm" else CLEVR_MANIFESTS[task]
        actual = digest(path)
        if not expected or expected != actual:
            raise ValueError(f"Input differs from the pinned test campaign: {path}")
        result[str(path.resolve())] = actual
    return result
