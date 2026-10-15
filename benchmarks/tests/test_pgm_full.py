import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from vlm_bench import pgm_full, pgm_mirror

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


@unittest.skipIf(pa is None, "Real parquet integration tests require pyarrow")
class FullMirrorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shards = []
        hashes = {}
        for shard_index in range(9):
            path = self.root / f"test-{shard_index:05d}-of-00009.parquet"
            rows = []
            for local in range(3):
                identifier = shard_index * 3 + local
                images = []
                for panel in range(16):
                    stream = BytesIO()
                    Image.new("L", (160, 160), panel * 10 + identifier % 5).save(stream, format="PNG")
                    images.append({"bytes": stream.getvalue(), "path": None})
                rows.append({"id": identifier, "target": identifier % 8,
                             "panels": images[:8], "choices": images[8:],
                             "rule_structure": "metadata must never reach the prompt"})
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
            self.shards.append(path)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        p = patch.dict(pgm_mirror.SHARD_SHA256, hashes, clear=True)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(pgm_mirror, "ORIGINAL_TEST_COUNT", 27)
        p.start()
        self.addCleanup(p.stop)
        self.output = self.root / "full" / "manifest.jsonl"

    def prepare(self, **kwargs):
        return pgm_full.prepare_full(self.shards, self.output, batch_size=1, **kwargs)

    def stage(self):
        return self.output.with_name("." + self.output.name + ".preparing")

    def rows(self):
        return [json.loads(line) for line in self.output.read_text().splitlines()]

    def test_full_stream_matches_existing_manifest_and_png_bytes(self):
        old = self.root / "old" / "manifest.jsonl"
        pgm_mirror.prepare_mirror(self.shards, old, limit=27, seed=7)
        real_parquet = pq.ParquetFile
        batch_sizes, reads = [], []

        class TrackedParquet:
            def __init__(self, path):
                self.raw = real_parquet(path)
                self.metadata = self.raw.metadata
                self.num_row_groups = self.raw.num_row_groups

            def read(self, columns):
                reads.append(columns)
                return self.raw.read(columns=columns)

            def iter_batches(self, **kwargs):
                self_outer.assertEqual(kwargs["columns"], pgm_full.COLUMNS)
                self_outer.assertFalse(kwargs["use_threads"])
                for batch in self.raw.iter_batches(**kwargs):
                    batch_sizes.append(batch.num_rows)
                    yield batch

        self_outer = self
        with patch.object(pq, "ParquetFile", TrackedParquet):
            self.prepare(seed=7)
        self.assertEqual(self.output.read_bytes(), old.read_bytes())
        self.assertEqual(len(batch_sizes), 27)
        self.assertEqual(set(batch_sizes), {1})
        self.assertEqual(reads, [["id"]] * 9)
        for row in self.rows():
            self.assertEqual((self.output.parent / row["images"][0]).read_bytes(),
                             (old.parent / row["images"][0]).read_bytes())
        sidecar = json.loads(self.output.with_name(self.output.name + ".provenance.json").read_text())
        self.assertEqual(sidecar["selected_count"], 27)
        self.assertEqual(sidecar["available_shard_count"], 9)
        self.assertEqual(sidecar["manifest_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertEqual(sidecar["selected_ids"], [r["id"] for r in self.rows()])

    def test_resume_after_committed_batches_preserves_completed_images(self):
        def stop(done, total):
            if done >= 4:
                raise RuntimeError("simulated allocation end")
        with self.assertRaisesRegex(RuntimeError, "allocation end"):
            self.prepare(progress=stop)
        self.assertFalse(self.output.exists())
        pngs = list((self.stage() / "manifest_images").glob("*.png"))
        self.assertEqual(len(pngs), 4)
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in pngs}
        calls = []
        original = pgm_full._make_row
        def tracked(*args):
            calls.append(args[0])
            return original(*args)
        with patch.object(pgm_full, "_make_row", side_effect=tracked):
            self.prepare(resume=True)
        self.assertEqual(len(calls), 23)
        for name, (payload, mtime) in before.items():
            path = self.output.parent / "manifest_images" / name
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
        original_manifest = self.output.read_bytes()
        with patch.object(pgm_full, "_make_row", side_effect=AssertionError("must skip completed rows")):
            self.prepare(resume=True)
        self.assertEqual(self.output.read_bytes(), original_manifest)

    def test_resume_adopts_only_matching_png_left_before_journal_commit(self):
        original = pgm_full._publish_bytes
        count = [0]
        def crash_after_png(path, payload):
            original(path, payload)
            if path.suffix == ".png":
                count[0] += 1
                if count[0] == 3:
                    raise RuntimeError("crash before commit")
        with patch.object(pgm_full, "_publish_bytes", side_effect=crash_after_png):
            with self.assertRaisesRegex(RuntimeError, "before commit"):
                self.prepare()
        with sqlite3.connect(self.stage() / "checkpoint.sqlite3") as database:
            self.assertEqual(database.execute("SELECT COUNT(*) FROM completed").fetchone()[0], 2)
        files = {p.name: p.stat().st_mtime_ns for p in (self.stage() / "manifest_images").glob("*.png")}
        self.assertEqual(len(files), 3)
        self.prepare(resume=True)
        self.assertEqual(len(self.rows()), 27)
        for name, mtime in files.items():
            self.assertEqual((self.output.parent / "manifest_images" / name).stat().st_mtime_ns, mtime)

    def test_resume_rejects_modified_completed_image_or_changed_plan(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "plan differs"):
            self.prepare(seed=1, resume=True)
        target = self.output.parent / self.rows()[0]["images"][0]
        target.write_bytes(b"foreign replacement")
        with self.assertRaisesRegex(ValueError, "missing or modified"):
            self.prepare(resume=True)
        self.assertEqual(target.read_bytes(), b"foreign replacement")

    def test_resume_recovers_interrupted_final_publication(self):
        original = pgm_full._link_verified
        def interrupt(source, target):
            if target == self.output.with_name(self.output.name + ".provenance.json"):
                raise RuntimeError("publication interrupted")
            return original(source, target)
        with patch.object(pgm_full, "_link_verified", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "publication interrupted"):
                self.prepare()
        self.assertTrue((self.output.parent / "manifest_images").is_dir())
        self.assertFalse(self.output.exists())
        self.prepare(resume=True)
        self.assertEqual(len(self.rows()), 27)

    def test_refuses_unrelated_output_incomplete_pool_and_changed_source(self):
        self.output.parent.mkdir()
        self.output.write_text("unrelated content")
        with self.assertRaisesRegex(FileExistsError, "existing output"):
            self.prepare(resume=True)
        self.assertEqual(self.output.read_text(), "unrelated content")
        self.output = self.root / "new.jsonl"
        with self.assertRaisesRegex(ValueError, "nine pinned"):
            pgm_full.prepare_full(self.shards[:-1], self.output)
        self.shards[0].write_bytes(b"wrong source")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.prepare()
        self.assertFalse(self.stage().exists())


if __name__ == "__main__":
    unittest.main()
