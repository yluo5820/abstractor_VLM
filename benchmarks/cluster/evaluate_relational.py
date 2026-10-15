"""Full 200k fixed-endpoint evaluation, reusing the audited PGM shard merger."""
import argparse
import json
from pathlib import Path

import evaluate_pgm_parallel as common


def pin(campaign):
    from vlm_bench.run import atomic_json
    train = campaign/'train'
    done = json.loads((train/'COMPLETE.json').read_text())
    assert done['seen'] == 240000 and done['status'] == 'one-pass'
    assert done['validation']['complete'] and done['validation']['evaluated'] == 1000
    checkpoint = done['selected_checkpoint']
    assert checkpoint == json.loads((train/'best.json').read_text())
    config = json.loads((train/'config.json').read_text())
    from vlm_bench.relational_train import source_digest
    assert config['source_sha256'] == source_digest()
    folder = train/checkpoint['directory']; adapter=folder/'adapter'
    for name, digest in json.loads((folder/'checksums.json').read_text()).items():
        assert common.digest(folder/name) == digest
    plan = json.loads((campaign.parent/'eval-template.json').read_text())
    plan['models'] = {'lfm25': dict(train=str(train), checkpoint=checkpoint, revision=config['revision'],
        training_config_sha256=common.digest(train/'config.json'),
        adapter_sha256={p.name:common.digest(p) for p in adapter.iterdir() if p.is_file()},
        probe_sha256=common.digest(folder/'probe.safetensors'), variant=config['variant'],
        probe=config['probe'], selection_policy='fixed-240k-endpoint',
        selected_validation_sha256=common.digest(train/'selected-validation.json'),
        training_source_sha256=config['source_sha256'])}
    path = campaign/'plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        atomic_json(path, plan)
    return plan


def worker(plan, rank, campaign):
    from peft import PeftModel
    from safetensors.torch import load_file
    from vlm_bench import run, models
    from vlm_bench.relational import install
    from vlm_bench.relational_train import source_digest
    run.require_allocation()
    shard, spec = plan['shards'][rank], plan['models']['lfm25']
    assert common.digest(shard['manifest']) == shard['manifest_sha256']
    assert source_digest() == spec['training_source_sha256']
    adapter = common.check_adapter(spec)
    weights = adapter.parent/'probe.safetensors'
    assert common.digest(weights) == spec['probe_sha256']
    output = campaign/'lfm25/shards'/str(rank); output.mkdir(parents=True, exist_ok=True)
    provenance = dict(protocol='relational-lora-fixed-240k-pgm-neutral-test-v1', **spec,
                      shard=rank, original_manifest_sha256=plan['manifest_sha256'],
                      evaluator_sha256=common.digest(__file__))
    receipt=output/'adapter-provenance.json'
    if receipt.exists(): assert json.loads(receipt.read_text()) == provenance
    run.atomic_json(receipt, provenance)
    original=models.load_backend
    def load(*args, **kwargs):
        assert json.loads((output/'config.json').read_text())['images_sha256'] == shard['images_sha256']
        backend=original(*args, **kwargs)
        backend.model=PeftModel.from_pretrained(backend.model, str(adapter), is_trainable=False)
        controller=install(backend.model.get_base_model(), backend.processor, spec['variant'], load_file(str(weights)))
        assert controller.probe.spec == spec['probe']
        backend.model.requires_grad_(False); backend.model.eval()
        backend.metadata['adaptation']=provenance
        return backend
    models.load_backend=load
    result=run.run(argparse.Namespace(manifest=shard['manifest'], output=str(output), model='lfm25',
        revision=spec['revision'], max_samples=0, max_new_tokens=16, seed=0, dtype='bfloat16',
        cpu_threads=2, interop_threads=1, summary_every=100, dry_run=False))
    assert result['complete'] and result['evaluated']==shard['count'], 'Incomplete test shard'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True, help='Variant root')
    p.add_argument('--model', choices=['lfm25'], default='lfm25')
    p.add_argument('--mode', choices=['pin','worker','launch','merge'], default='launch')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--seconds', type=int, default=14220)
    a=p.parse_args()
    if a.mode == 'pin': pin(a.campaign); return
    plan=json.loads((a.campaign/'plan.json').read_text())
    if a.mode == 'worker': worker(plan, a.rank, a.campaign)
    elif a.mode == 'merge': print(json.dumps(common.merge(plan,'lfm25',a.campaign)))
    else: common.launch(plan,'lfm25',a.campaign,a.seconds,worker_script=str(Path(__file__).resolve()))


if __name__ == '__main__': main()
