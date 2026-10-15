import unittest
from vlm_bench.pgm_epochs import ORDER_POLICY, coverage, next_indices, validate_resume_schedule


class PGMEpochTests(unittest.TestCase):
    def test_first_epoch_unchanged_and_second_is_same_pool_once(self):
        count = 240
        passes = []
        for epoch in range(2):
            passes.append([i for seen in range(epoch*count, (epoch+1)*count, 16)
                           for i in next_indices(seen, count, 2)])
        self.assertEqual(passes[0], list(range(count)))
        self.assertEqual(sorted(passes[1]), list(range(count)))
        self.assertNotEqual(passes[0], passes[1])
        self.assertEqual(next_indices(2*count, count, 2), [])

    def test_resume_across_boundary_and_mid_second_epoch(self):
        full = [i for seen in range(0, 480, 16) for i in next_indices(seen, 240, 2)]
        for offset in [224, 240, 256, 400]:
            prefix = [i for seen in range(0, offset, 16) for i in next_indices(seen, 240, 2)]
            resumed = [i for seen in range(offset, 480, 16) for i in next_indices(seen, 240, 2)]
            self.assertEqual(prefix + resumed, full)

    def test_unique_coverage_does_not_increase_with_repeats(self):
        self.assertEqual(coverage(400000)['unique_examples'], 240000)
        self.assertEqual(coverage(400000)['fraction_official_train'], .2)
        self.assertEqual(coverage(480000)['effective_epochs'], 2)

    def test_legacy_endpoint_can_resume_but_incompatible_state_cannot(self):
        config = {'training_order':'nested hash order, no repeats','seed':20261007,'train_pool':240000}
        validate_resume_schedule(config, 240000, 15000, 240000, 2)
        for seen, step in [(240000, 14999), (480000, 30000), (240016, 15001)]:
            with self.assertRaises(ValueError):
                validate_resume_schedule(config, seen, step, 240000, 2)
        config['training_order'] = ORDER_POLICY
        validate_resume_schedule(config, 320000, 20000, 240000, 2)


if __name__ == '__main__':
    unittest.main()
