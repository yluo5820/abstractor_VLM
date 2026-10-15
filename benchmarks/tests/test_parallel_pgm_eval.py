import json
from pathlib import Path
import tempfile
import unittest

from cluster.evaluate_pgm_parallel import digest, merge, validate_partition


class ParallelPGMEvaluationTests(unittest.TestCase):
    def test_partition_rejects_missing_duplicate_and_changed_questions(self):
        rows = [{'id': str(i), 'answer': 'A'} for i in range(12)]
        shards = [rows[i::4] for i in range(4)]
        validate_partition(rows, shards)
        for bad in [shards[:-1], shards+[shards[0]],
                    [[dict(r, answer='B') for r in shards[0]]] + shards[1:]]:
            with self.assertRaises(ValueError):
                validate_partition(rows, bad)

    def test_merge_preserves_original_order_and_marks_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [dict(id=str(i), dataset='pgm', task='matrix-completion', regime='neutral',
                         split='test', choices=['A','B'], answer='A') for i in range(12)]
            full = root/'full.jsonl'
            full.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            plan = {'manifest': str(full), 'manifest_sha256': digest(full), 'images_sha256': 'full',
                    'models': {'lfm25': {'revision': 'pinned'}}, 'shards': []}
            for rank in range(4):
                manifest = root/f'shard-{rank}.jsonl'
                manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows[rank::4]))
                spec = {'manifest': str(manifest), 'manifest_sha256': digest(manifest),
                        'images_sha256': str(rank)}
                plan['shards'].append(spec)
                out = root/'lfm25/shards'/str(rank)
                out.mkdir(parents=True)
                (out/'config.json').write_text(json.dumps(spec))
                preds = [dict(r, prediction='A', raw_output='A', correct=True, latency_seconds=0.1)
                         for r in rows[rank::4]]
                (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in preds))
            result = merge(plan, 'lfm25', root)
            self.assertTrue(result['complete'])
            self.assertEqual(result['evaluated'], 12)
            self.assertEqual(result['groups'][0]['accuracy'], 1.0)
            merged = [json.loads(x) for x in (root/'lfm25/test/predictions.jsonl').read_text().splitlines()]
            self.assertEqual([x['id'] for x in merged], [x['id'] for x in rows])
            last = root/'lfm25/shards/3/predictions.jsonl'
            last.write_text('\n'.join(last.read_text().splitlines()[:-1])+'\n')
            result = merge(plan, 'lfm25', root)
            self.assertFalse(result['complete'])
            self.assertEqual(result['evaluated'], 11)
            # Saved correctness must agree with the original labels and strict parser.
            records = [json.loads(x) for x in last.read_text().splitlines()]
            records[0]['correct'] = False
            last.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(ValueError):
                merge(plan, 'lfm25', root)


if __name__ == '__main__':
    unittest.main()
