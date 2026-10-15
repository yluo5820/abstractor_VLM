import unittest
from types import SimpleNamespace

import torch

from vlm_bench.relational import RelationalProbe, VARIANTS, Controller, prefix_end, ordered_pool


class TestRelational(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)

    def test_pool_deterministic_and_short_text(self):
        x = torch.arange(12.).reshape(1, 6, 2).requires_grad_()
        self.assertTrue(torch.equal(ordered_pool(x, 3), x.reshape(1, 3, 2, 2).mean(2)))
        short = ordered_pool(x[:, :2], 8)
        self.assertEqual(short.shape, (1, 8, 2))
        short.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_noop_and_deep_gradient_flow(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                m = RelationalProbe(variant, 12, 16, width=8, heads=2, depth=2, symbols=4)
                x = torch.randn(1, 9, 12 if variant == 'early' else 16)
                t = torch.randn(1, 3, 16)
                def forward():
                    return m.read_memory(x, m.abstract(x)) if variant == 'memory' else m(x, t)
                self.assertTrue(torch.equal(x, forward()))
                opt = torch.optim.SGD(m.parameters(), lr=.1)
                target = torch.randn_like(x)
                for _ in range(2):
                    opt.zero_grad()
                    (forward()-target).square().mean().backward()
                    opt.step()
                self.assertGreater(m.layers[0].relation.q.weight.grad.abs().sum().item(), 0)
                self.assertGreater(m.symbols.grad.abs().sum().item(), 0)
                if variant == 'dual':
                    self.assertGreater(m.text_layers[0].relation.q.weight.grad.abs().sum().item(), 0)

    def test_no_cross_example_mixing(self):
        m = RelationalProbe('late', text_dim=16, width=8, heads=2, depth=2, symbols=4)
        torch.nn.init.normal_(m.output.weight)
        x = torch.randn(2, 9, 16)
        expected = m(x)[0]
        x[1] += 100
        self.assertTrue(torch.equal(expected, m(x)[0]))

    def test_header_detection(self):
        self.assertEqual(prefix_end(torch.tensor([8, 4, 5, 6, 7]), [4, 5]), 3)
        for ids in ([8, 4], [4, 5, 4, 5]):
            with self.assertRaises(ValueError):
                prefix_end(torch.tensor(ids), [4, 5])

    def test_joint_excludes_answers_and_decode_is_unchanged(self):
        c = object.__new__(Controller)
        c.probe = RelationalProbe('joint', text_dim=16, width=8, heads=2, depth=2, symbols=4)
        torch.nn.init.normal_(c.probe.output.weight)
        c.context = {'prefill': True, 'end': 7}
        x = torch.randn(1, 10, 16)
        y = c.language(None, None, x)
        x2 = x.clone(); x2[:, 7:] += 100
        y2 = c.language(None, None, x2)
        self.assertTrue(torch.equal(y[:, :7], y2[:, :7]))
        self.assertTrue(torch.equal(y[:, 7:], x[:, 7:]))
        c.context['prefill'] = False
        single = x[:, :1]
        self.assertIs(c.language(None, None, single), single)

    def test_vision_padding_not_read_or_modified(self):
        c = object.__new__(Controller)
        c.probe = RelationalProbe('early', vision_dim=12, width=8, heads=2, depth=2, symbols=4)
        torch.nn.init.normal_(c.probe.output.weight)
        x = torch.randn(2, 5, 12)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        y = c.vision(None, None, {'pixel_attention_mask': mask}, SimpleNamespace(last_hidden_state=x)).last_hidden_state
        changed = x.clone(); changed[~mask] += 1000
        z = c.vision(None, None, {'pixel_attention_mask': mask}, SimpleNamespace(last_hidden_state=changed)).last_hidden_state
        self.assertTrue(torch.equal(y[mask], z[mask]))
        self.assertTrue(torch.equal(y[~mask], x[~mask]))


if __name__ == '__main__':
    unittest.main()
