import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from vlm_bench import pgm_mirror


class FakeTable:
    def __init__(self, rows, materialized):
        self.rows = rows
        self.materialized = materialized

    def column(self, name):
        return SimpleNamespace(to_pylist=lambda: [row[name] for row in self.rows])

    def take(self, indices):
        return FakeTable([self.rows[index] for index in indices], self.materialized)

    def slice(self, offset, length):
        return FakeTable(self.rows[offset:offset + length], self.materialized)

    def to_pylist(self):
        self.materialized.extend(row["id"] for row in self.rows)
        return self.rows


class MirrorPreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.shard = self.root / "test-00000-of-00009.parquet"
        self.shard.write_bytes(b"fake parquet fixture")
        self.digest = hashlib.sha256(self.shard.read_bytes()).hexdigest()
        images = []
        for i in range(16):
            payload = BytesIO()
            Image.new("L", (160, 160), i * 10).save(payload, format="PNG")
            images.append({"bytes": payload.getvalue(), "path": None})
        self.raw_rows = [{"id": 1000 + i, "target": i % 8,
                          "panels": images[:8], "choices": images[8:]}
                         for i in range(24)]
        self.materialized = []

        def read_group(index, columns):
            self.assertEqual(columns, ["panels", "choices", "target", "id"])
            return FakeTable(self.raw_rows[index * 8:(index + 1) * 8], self.materialized)

        def read(columns):
            self.assertEqual(columns, ["id"])
            return FakeTable(self.raw_rows, self.materialized)

        self.parquet = SimpleNamespace(
            num_row_groups=3,
            metadata=SimpleNamespace(row_group=lambda index: SimpleNamespace(num_rows=8)),
            read=MagicMock(side_effect=read), read_row_group=MagicMock(side_effect=read_group),
        )
        self.arrow = ModuleType("pyarrow")
        self.arrow.__version__ = "test-version"
        self.arrow.parquet = ModuleType("pyarrow.parquet")
        self.arrow.parquet.ParquetFile = MagicMock(return_value=self.parquet)
        self.module_patch = patch.dict("sys.modules", {
            "pyarrow": self.arrow, "pyarrow.parquet": self.arrow.parquet,
        })
        self.hash_patch = patch.dict(pgm_mirror.SHARD_SHA256, {self.shard.name: self.digest})
        self.module_patch.start()
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)
        self.addCleanup(self.module_patch.stop)
        self.addCleanup(self.directory.cleanup)

    def test_shared_row_groups_read_once_and_only_selected_rows_materialized(self):
        output = self.root / "pilot.jsonl"
        pgm_mirror.prepare_mirror([self.shard], output, limit=16, seed=0)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        groups = [call.args[0] for call in self.parquet.read_row_group.call_args_list]
        self.assertEqual(sorted(groups), [0, 1, 2])
        self.assertEqual(len(self.materialized), 16)
        self.assertEqual(set(self.materialized), {row["metadata"]["mirror_sample_id"] for row in rows})
        self.assertEqual(len(rows), 16)
        for row in rows:
            original_id = row["metadata"]["mirror_sample_id"]
            self.assertEqual(row["id"], f"pgm/neutral/test/{original_id}")
            self.assertEqual(row["answer"], chr(65 + (original_id - 1000) % 8))
            self.assertEqual(row["metadata"]["mirror_row_index"], original_id - 1000)
            self.assertNotIn("relation_structure", row["prompt"])
        sidecar = json.loads(output.with_name(output.name + ".provenance.json").read_text())
        self.assertEqual(sidecar["selected_ids"], [row["id"] for row in rows])
        self.assertEqual(sidecar["selected_count"], 16)
        self.assertEqual(sidecar["available_pool_size"], 24)
        self.assertEqual(sidecar["mirror_revision"], pgm_mirror.REVISION)
        self.assertEqual(sidecar["shards"], [{"file": self.shard.name, "sha256": self.digest,
                                             "rows": 24, "bytes": self.shard.stat().st_size}])

    def test_checksum_failure_prevents_parquet_read_or_published_output(self):
        self.shard.write_bytes(b"modified fixture")
        output = self.root / "bad.jsonl"
        with self.assertRaisesRegex(ValueError, "checksum"):
            pgm_mirror.prepare_mirror([self.shard], output)
        self.arrow.parquet.ParquetFile.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
