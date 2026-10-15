import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cluster'))
import evaluate_pgm_parallel as common
from evaluate_relational_frozen import check_probe


class FrozenEvaluationTests(unittest.TestCase):
    def test_module_checkpoint_integrity_and_no_adapter_requirement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'checkpoint').mkdir()
            weights = root/'checkpoint/probe.safetensors'
            weights.write_bytes(b'module checkpoint')
            (root/'config.json').write_text('{}')
            checkpoint = {'directory': 'checkpoint', 'seen': 240000}
            (root/'best.json').write_text(json.dumps(checkpoint))
            spec = dict(train=str(root), checkpoint=checkpoint,
                        training_config_sha256=common.digest(root/'config.json'),
                        probe_sha256=common.digest(weights))
            self.assertEqual(check_probe(spec), weights)
            weights.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                check_probe(spec)

    def test_launcher_calls_explicit_validator_before_workers(self):
        child = Mock(returncode=0)
        child.poll.return_value = 0
        validator = Mock()
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3'), \
             patch.object(common, 'check_adapter', side_effect=AssertionError('Must not require LoRA')), \
             patch.object(common.signal, 'signal'), \
             patch.object(common.subprocess, 'Popen', return_value=child) as spawn, \
             patch.object(common, 'merge', return_value={'complete':True,'evaluated':200000}):
            common.launch({'models': {'lfm25': {'module':True}}}, 'lfm25', Path(temp), 60,
                          worker_script='frozen.py', checkpoint_validator=validator)
            validator.assert_called_once_with({'module':True})
            self.assertEqual(spawn.call_count, 4)


if __name__ == '__main__': unittest.main()
