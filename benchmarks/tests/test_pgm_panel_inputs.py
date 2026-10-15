import unittest

import numpy as np
from PIL import Image

from vlm_bench.data import render_pgm
from vlm_bench.pgm_panel_inputs import messages, split_composite


class PanelInputsTest(unittest.TestCase):
    def test_lossless_native_panel_recovery(self):
        # Random pixels detect one-pixel crop errors, border contamination,
        # resizing, and any permutation of context or answer panels.
        panels = np.random.default_rng(42).integers(0, 256, (16, 160, 160), dtype=np.uint8)
        recovered = split_composite(render_pgm(panels))
        self.assertEqual(len(recovered), 16)
        for source, panel in zip(panels, recovered):
            self.assertEqual(panel.size, (160, 160))
            np.testing.assert_array_equal(np.asarray(panel), np.repeat(source[:, :, None], 3, axis=2))

    def test_labels_preserve_matrix_and_option_order(self):
        image = render_pgm(np.zeros((16, 160, 160), dtype=np.uint8))
        content = messages(image, "Original instruction", "separate")[0]["content"]
        self.assertEqual(sum(x["type"] == "image" for x in content), 16)
        labels = [x["text"] for x in content if x["type"] == "text"]
        self.assertEqual(labels[:8], [f"Context panel: row {i // 3 + 1}, column {i % 3 + 1}." for i in range(8)])
        self.assertEqual(labels[8:16], [f"Answer option {a}." for a in "ABCDEFGH"])
        self.assertEqual(labels[-1], "Original instruction")
        self.assertEqual(len(messages(image, "Original instruction", "composite")[0]["content"]), 2)

    def test_rejects_wrong_layout(self):
        with self.assertRaises(ValueError):
            split_composite(Image.new("RGB", (700, 1015)))


if __name__ == "__main__":
    unittest.main()
