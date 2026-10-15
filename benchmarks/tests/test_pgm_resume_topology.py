import unittest

from vlm_bench.pgm_epochs import next_indices, resume_rng_rank


class ResumeTopologyTest(unittest.TestCase):
    def test_reduction_requires_explicit_flag(self):
        with self.assertRaises(ValueError):
            resume_rng_rank(4, 1, 0)
        self.assertEqual(resume_rng_rank(4, 1, 0, True), 0)
        self.assertEqual(resume_rng_rank(4, 2, 1, True), 1)
        self.assertEqual(resume_rng_rank(4, 4, 3), 3)
        with self.assertRaises(ValueError):
            resume_rng_rank(1, 4, 0, True)

    def test_repartition_preserves_exact_global_batch_and_update_weight(self):
        batch = next_indices(240000, 240000, 2)
        contributions = {item: (i + 1) / 16 for i, item in enumerate(batch)}
        for world in (1, 2, 4):
            shards = [batch[rank::world] for rank in range(world)]
            flat = [item for shard in shards for item in shard]
            self.assertEqual(len(flat), 16)
            self.assertEqual(set(flat), set(batch))
            self.assertEqual(sum(sum(contributions[item] for item in shard) for shard in shards),
                             sum(contributions.values()))


if __name__ == '__main__':
    unittest.main()
