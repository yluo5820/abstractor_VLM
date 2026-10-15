import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from vlm_bench.finetune import (answer_labels, epoch_order, save_checkpoint, suffix_loss,
                                epoch_budget_exhausted, verify_pilot_source)
from vlm_bench.finetune_data import partition_ids, verify_row


class FineTuneTests(unittest.TestCase):
    def test_time_budget_does_not_stop_at_three_epochs(self):
        for epoch in (0, 3, 6, 100):
            self.assertFalse(epoch_budget_exhausted(epoch, 0))
        self.assertFalse(epoch_budget_exhausted(2, 3))
        self.assertTrue(epoch_budget_exhausted(3, 3))
        with self.assertRaises(ValueError):
            epoch_budget_exhausted(0, -1)

    def test_pilot_compatibility_is_exactly_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'review.json'
            path.write_text(json.dumps({'pilot_source_sha256': 'old', 'training_source_sha256': 'new',
                                        'scope': 'stopping-policy-only', 'core_functions_unchanged': True}))
            verify_pilot_source('old', 'new', path)
            with self.assertRaises(ValueError):
                verify_pilot_source('old', 'different', path)
            with self.assertRaises(ValueError):
                verify_pilot_source('different', 'new', path)

    def test_fixed_disjoint_partition_does_not_depend_on_render_availability(self):
        train, validation = partition_ids()
        self.assertEqual((len(train), len(validation)), (9000, 1000))
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(len(set(train + validation)), 10000)
        self.assertTrue(all('/train/' in i for i in train + validation))
        self.assertEqual((train, validation), partition_ids())

    def test_no_question_or_image_tokens_in_loss(self):
        full = torch.tensor([[1, 2, 3, 4, 5, 6]])
        labels, start = answer_labels(full[:, :4], full)
        self.assertEqual(labels.tolist(), [[-100, -100, -100, -100, 5, 6]])
        self.assertEqual(start, 4)
        with self.assertRaises(ValueError):
            answer_labels(torch.tensor([[1, 8, 3, 4]]), full)

    def test_suffix_loss_and_gradients_match_full_masked_causal_loss(self):
        torch.manual_seed(5)
        logits = torch.randn(1, 7, 11, requires_grad=True)
        inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6, 7]])}
        labels, start = answer_labels(inputs['input_ids'][:, :4], inputs['input_ids'])
        class Model:
            def __call__(self, logits_to_keep, **kwargs):
                return SimpleNamespace(logits=logits[:, -logits_to_keep:])
        actual = suffix_loss(Model(), inputs, labels, start)
        expected = F.cross_entropy(logits[:, :-1].reshape(-1, 11), labels[:, 1:].reshape(-1), ignore_index=-100)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                       torch.autograd.grad(expected, logits)[0]))

    def test_resume_order_is_exact_and_each_epoch_covers_all_examples(self):
        first = epoch_order(9000, 20261007, 0)
        second = epoch_order(9000, 20261007, 1)
        self.assertEqual(sorted(first), list(range(9000)))
        self.assertNotEqual(first, second)
        self.assertEqual(first[320:], epoch_order(9000, 20261007, 0)[320:])

    def test_test_rows_cannot_enter_training(self):
        with self.assertRaises(ValueError):
            verify_row({'dataset': 'clevr-art', 'task': 'idrules', 'split': 'test',
                        'id': 'clevr-art/idrules/ood/test/0'}, Path('/tmp'))

    def test_failed_checkpoint_does_not_publish_latest(self):
        class BrokenModel:
            def save_pretrained(self, *args, **kwargs):
                raise RuntimeError('simulated write failure')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                save_checkpoint(root, BrokenModel(), None, {'step': 4, 'seen': 64})
            self.assertFalse((root / 'latest.json').exists())


if __name__ == '__main__':
    unittest.main()
