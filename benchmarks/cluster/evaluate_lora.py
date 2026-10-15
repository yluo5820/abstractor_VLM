"""Evaluate a validation-selected adapter through the unchanged benchmark runner."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    campaign = Path(a.campaign)
    sys.path.insert(0, str(campaign / 'source'))
    from vlm_bench import models, run
    from peft import PeftModel
    selected = json.loads((campaign / 'train/best.json').read_text())
    adapter = campaign / 'train' / selected['directory'] / 'adapter'
    config = json.loads((campaign / 'train/config.json').read_text())
    provenance = {
        'protocol': 'validation-selected-lora-ood-test-v1',
        'checkpoint': selected,
        'adapter_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in sorted(adapter.iterdir()) if f.is_file()},
        'training_config_sha256': run.sha256(campaign / 'train/config.json'),
        'evaluator_sha256': run.sha256(Path(__file__)),
        'baseline_accuracy_percent': 24.89,
    }
    rows = run.load_manifest(Path(a.manifest), 0)
    if len(rows) != 10000 or any(r['dataset'] != 'clevr-art' or r['task'] != 'idrules'
                              or r['split'] != 'test' for r in rows):
        raise ValueError('Expected the original 10,000-question CLEVR identity test set')
    expected = '320142264ae614c3660a1960cd2b0bef891e38a99faa49b52dcfc0a2c58339fc'
    if run.sha256(Path(a.manifest)) != expected:
        raise ValueError('Original test manifest checksum changed')
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'adapter-provenance.json'
    if receipt.exists() and json.loads(receipt.read_text()) != provenance:
        raise ValueError('Adapter or evaluator changed on resume')
    run.atomic_json(receipt, provenance)
    original = models.load_backend
    def load(*args, **kwargs):
        backend = original(*args, **kwargs)
        backend.model = PeftModel.from_pretrained(backend.model, str(adapter), is_trainable=False)
        backend.model.eval()
        backend.metadata['adaptation'] = provenance
        return backend
    models.load_backend = load
    run.run(argparse.Namespace(manifest=a.manifest, output=a.output, model='lfm25',
        revision=models.DEFAULT_REVISIONS['lfm25'], max_samples=0, max_new_tokens=16,
        seed=0, dtype='bfloat16', cpu_threads=2, interop_threads=1,
        summary_every=25, dry_run=False))


if __name__ == '__main__':
    main()
