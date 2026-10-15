"""Resumable, fixed-coverage LFM relational probes; production is pilot-gated."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import uuid

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

from .adaptation import encode, suffix_loss, make_train_mode, validate
from .adaptation_data import image_for
from .data import PGM_PROMPT, sha256_file
from .models import load_backend, DEFAULT_REVISIONS
from .parallel_adaptation import read_pool, synchronize_gradients
from .relational import install, VARIANTS, prefix_end
from .run import atomic_json, require_allocation, wilson


def source_digest():
    h = hashlib.sha256()
    for name in ('relational.py', 'relational_train.py', 'parallel_adaptation.py',
                 'adaptation.py', 'adaptation_data.py', 'models.py', 'data.py', 'run.py'):
        h.update(name.encode()); h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--mode', choices=['pilot', 'train'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='Variant root, shared across bounded segments')
    p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    require_allocation()
    rank, world, local = [int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    assert world == 4, 'This campaign is pinned to four GPUs and global batch 16'
    torch.cuda.set_device(local)
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261007)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
    root = a.output.resolve()
    output = root / a.mode
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = (root / '.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dist.barrier()
    if a.mode == 'train' and (output / 'COMPLETE.json').exists():
        done = json.loads((output / 'COMPLETE.json').read_text())
        assert done['seen'] == 240000 and done['source_sha256'] == source_digest()
        dist.destroy_process_group()
        return
    if a.mode == 'train':
        gate = json.loads((root / 'pilot-result.json').read_text())
        assert gate['status'] == 'passed' and gate['source_sha256'] == source_digest()
        assert gate['variant'] == a.variant
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda number, frame: stopped.append(number))
    started = time.time()
    rows, ready = read_pool(a.data_root)
    backend = load_backend('lfm25', revision=DEFAULT_REVISIONS['lfm25'], device=f'cuda:{local}',
                           dtype='bfloat16', max_new_tokens=16)

    def inputs(row):
        assert sha256_file(a.data_root / row['images'][0]) == row['metadata']['contact_sheet_sha256']
        return encode(backend.processor, image_for(row, a.data_root), PGM_PROMPT,
                      row['answer'], f'cuda:{local}', torch.bfloat16)

    audit = inputs(rows['train'][rank])
    audit_kwargs, labels, start = audit
    count = audit_kwargs['input_ids'].shape[1] - start
    with torch.no_grad():
        base_logits = backend.model(**audit_kwargs, use_cache=False, logits_to_keep=count+1).logits.detach().clone()
    targets = [n for n, m in backend.model.named_modules()
               if n.startswith('model.language_model.') and isinstance(m, torch.nn.Linear)]
    assert targets
    model = get_peft_model(backend.model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                                  bias='none', target_modules=targets))
    backend.model = model
    controller = install(model.get_base_model(), backend.processor, a.variant)
    probe = controller.probe
    assert prefix_end(audit_kwargs['input_ids'][0], controller.marker) == start
    named = [(n, v) for n, v in model.named_parameters() if v.requires_grad]
    assert all('lora_' in n or 'relation_probe.' in n for n, v in named)
    trainable = [v for n, v in named]
    for v in trainable:
        v.data = v.data.float()
        dist.broadcast(v.data, src=0)
    model.eval()
    with torch.no_grad():
        logits = model(**audit_kwargs, use_cache=False, logits_to_keep=count+1).logits
    noop_error = float((logits - base_logits).abs().max())
    assert noop_error == 0, ('Changed pretrained function at initialization', noop_error)
    del logits, base_logits
    config = {'model': 'lfm25', 'variant': a.variant, 'probe': probe.spec,
        'revision': DEFAULT_REVISIONS['lfm25'], 'source_sha256': source_digest(),
        'policy': 'new-relational-module-plus-language-lora; original-weights-frozen',
        'world_size': world, 'global_batch': 16, 'micro_batch': 1, 'rank': 16, 'alpha': 32,
        'dropout': .05, 'lr': 1e-4, 'warmup_steps': 20, 'weight_decay': .01, 'clip_grad_norm': 1.,
        'train_pool': 240000, 'official_train_count': 1200000, 'max_epochs': 1,
        'selection_policy': 'fixed-240k-endpoint', 'seed': 20261007,
        'training_order': 'unchanged nested hash order; no repeats; resume next unseen example',
        'base_dtype': 'bfloat16', 'trainable_dtype': 'float32', 'deterministic_algorithms': True,
        'synchronization': 'SUM of globally normalized FP32 trainable gradients',
        'trainable_parameters': sum(v.numel() for v in trainable),
        'relational_parameters': sum(v.numel() for v in probe.parameters()),
        'lora_parameters': sum(v.numel() for n, v in named if 'lora_' in n), 'data': ready}
    if (output / 'config.json').exists():
        assert json.loads((output / 'config.json').read_text()) == config
    elif rank == 0:
        atomic_json(output / 'config.json', config)
    fresh = {n: v.detach().cpu().clone() for n, v in named}
    def new_optimizer():
        return torch.optim.AdamW(trainable, lr=1e-4, weight_decay=.01)
    optimizer = new_optimizer()
    torch.manual_seed(20261007 + rank)

    def stop(deadline):
        flag = torch.tensor(int(bool(stopped) or time.time() >= deadline), device=f'cuda:{local}')
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def update(seen, step, active=None):
        make_train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = 1e-4 * min(1., (step+1)/20)
        total = torch.zeros((), device=f'cuda:{local}')
        for index in range(seen+rank, seen+16, world):
            loss = suffix_loss(model, *inputs(rows['train'][index]))
            assert torch.isfinite(loss)
            (loss / 16).backward()
            total += loss.detach().float()/16
        active = trainable if active is None else active
        synchronize_gradients(active)
        assert all(v.grad is None for v in model.parameters() if not v.requires_grad)
        norm = float(torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True))
        optimizer.step()
        dist.all_reduce(total)
        return float(total), norm

    def checkpoint(state):
        rng = {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(local)}
        gathered = [None]*world if rank == 0 else None
        dist.gather_object(rng, gathered, dst=0)
        pointer = None
        if rank == 0:
            name = f"checkpoint-{state['step']:06d}-{uuid.uuid4().hex[:8]}"
            staged = output / ('.'+name); staged.mkdir()
            model.save_pretrained(staged/'adapter', safe_serialization=True, save_embedding_layers=False)
            save_file({n: v.detach().cpu().contiguous() for n, v in probe.state_dict().items()}, str(staged/'probe.safetensors'))
            torch.save({'optimizer': optimizer.state_dict(), 'state': state, 'rank_rng': gathered,
                        'world_size': world, 'config_sha256': sha256_file(output/'config.json')}, staged/'training.pt')
            atomic_json(staged/'state.json', state)
            hashes = {str(q.relative_to(staged)): sha256_file(q) for q in staged.rglob('*') if q.is_file()}
            atomic_json(staged/'checksums.json', hashes)
            for q in staged.rglob('*'):
                if q.is_file():
                    with q.open('rb') as f: os.fsync(f.fileno())
            staged.rename(output/name)
            pointer = {'directory': name, 'step': state['step'], 'seen': state['seen']}
            atomic_json(output/'latest.json', pointer)
            # Retain two committed checkpoints in this new run only. Old campaigns
            # and any other files are never candidates for this retention policy.
            old = sorted(output.glob('checkpoint-*'), key=lambda q: q.stat().st_mtime)
            for q in old[:-2]:
                if q.is_dir() and (q/'checksums.json').exists(): shutil.rmtree(q)
        result = [pointer]; dist.broadcast_object_list(result, src=0)
        return result[0]

    def restore(pointer):
        folder = output / pointer['directory']
        assert folder.resolve().is_relative_to(output)
        for name, digest in json.loads((folder/'checksums.json').read_text()).items():
            assert sha256_file(folder/name) == digest
        result = set_peft_model_state_dict(model, load_file(str(folder/'adapter/adapter_model.safetensors')))
        assert not result.unexpected_keys and not any('lora_' in n for n in result.missing_keys)
        probe.load_state_dict(load_file(str(folder/'probe.safetensors')), strict=True)
        payload = torch.load(folder/'training.pt', map_location='cpu', weights_only=True)
        assert payload['world_size'] == world and payload['config_sha256'] == sha256_file(output/'config.json')
        optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['rank_rng'][rank]['torch'])
        torch.cuda.set_rng_state(payload['rank_rng'][rank]['cuda'], local)
        return payload['state']

    def replay(pointer, seen, step):
        first, _ = update(seen, step)
        expected = [v.detach().clone() for v in trainable]
        restore(pointer)
        second, _ = update(seen, step)
        error = max(float((v-e).abs().max()) for v, e in zip(trainable, expected))
        assert error <= 1e-6 and abs(first-second) <= 1e-6, (error, first, second)
        restore(pointer)
        return error

    def validation(selected, name, deadline):
        result = validate(backend, selected[rank::world], a.data_root, PGM_PROMPT,
                          output, f'{name}-rank{rank}', time.monotonic()+max(0, deadline-time.time()), stopped)
        all_results = [None]*world; dist.all_gather_object(all_results, result)
        preds = [x for r in all_results for x in r['predictions']]
        n, correct = len(preds), sum(x['correct'] for x in preds)
        merged = dict(phase=name, planned=len(selected), evaluated=n, complete=n==len(selected), correct=correct,
                      accuracy=correct/n if n else None, loss=sum(x['loss'] for x in preds)/n if n else None,
                      invalid=sum(x['prediction'] is None for x in preds), wilson_95=wilson(correct,n), predictions=preds)
        if rank == 0: atomic_json(output/(name+'.json'), merged)
        return merged

    state = dict(step=0, seen=0, training_seconds=0.)
    if a.mode == 'pilot':
        tick = time.time()
        for step in range(8):
            assert not stop(a.deadline-180), 'Pilot deadline exhausted'
            update(step*16, step)
        seconds = time.time()-tick
        assert float(probe.layers[0].relation.q.weight.grad.abs().sum()) > 0
        if a.variant == 'dual':
            assert float(probe.text_layers[0].relation.q.weight.grad.abs().sum()) > 0
        flat = torch.cat([v.detach().reshape(-1) for v in trainable]); ref = flat.clone()
        dist.broadcast(ref, src=0)
        replica_error = float((flat-ref).abs().max()); assert replica_error <= 1e-6
        del flat, ref
        pointer = checkpoint(dict(step=8, seen=128, training_seconds=seconds))
        replay_error = replay(pointer, 128, 8)
        model.eval()
        # Changing all supplied answer IDs must leave the first answer logits unchanged.
        changed = {**audit_kwargs, 'input_ids': audit_kwargs['input_ids'].clone()}
        changed['input_ids'][:, start:] = backend.processor.tokenizer.encode('Z', add_special_tokens=False)[0]
        with torch.no_grad():
            left = model(**audit_kwargs, use_cache=False, logits_to_keep=count+1).logits[:, 0].clone()
            right = model(**changed, use_cache=False, logits_to_keep=count+1).logits[:, 0]
        leak_error = float((left-right).abs().max()); assert leak_error == 0
        # Exercise actual nonzero modules with cached and uncached generation.
        prefill = {k: (v[:, :start] if k in ('input_ids', 'attention_mask') else v) for k,v in audit_kwargs.items()}
        with torch.no_grad():
            cached = model.generate(**prefill, max_new_tokens=4, do_sample=False, use_cache=True)
            uncached = model.generate(**prefill, max_new_tokens=4, do_sample=False, use_cache=False)
        assert torch.equal(cached, uncached), 'Cached and recomputed generation diverged'
        # Measure the module-only policy, without retaining these pilot updates.
        for n, v in named: v.data.copy_(fresh[n].to(v.device))
        optimizer = new_optimizer()
        for n, v in named:
            v.grad = None
            if 'lora_' in n: v.requires_grad_(False)
        module_parameters = list(probe.parameters())
        tick = time.time()
        for step in range(2): update(step*16, step, module_parameters)
        frozen_seconds = time.time()-tick
        for n, v in named: v.requires_grad_(True)
        # Inference sample is validation only, never a benchmark-test selection.
        for n, v in named: v.data.copy_(fresh[n].to(v.device))
        model.eval(); tick=time.time()
        for row in rows['validation'][rank*8:(rank+1)*8]:
            backend.predict(image_for(row, a.data_root), PGM_PROMPT)
        infer_seconds = time.time()-tick
        times = torch.tensor([seconds, frozen_seconds, infer_seconds], device=f'cuda:{local}')
        dist.all_reduce(times, op=dist.ReduceOp.MAX)
        if rank == 0:
            sec, frozen, infer = times.tolist()
            atomic_json(root/'pilot-result.json', dict(status='passed', variant=a.variant, source_sha256=source_digest(),
                initial_logits_max_abs_error=noop_error, answer_leakage_max_abs_error=leak_error,
                cached_uncached_generation_equal=True, deep_relation_gradient_nonzero=True,
                replica_max_abs_error=replica_error, checkpoint_replay_max_abs_error=replay_error,
                world_size=world, global_batch=16, examples_per_second=128/sec,
                projected_training_hours=240000/(128/sec)/3600,
                module_only_examples_per_second=32/frozen, inference_examples_per_second=32/infer,
                projected_test_hours=200000/(32/infer)/3600, **{'parameters': config},
                note='Pilot excludes startup/checkpoints/validation from throughput; small sample is uncertain'))
        dist.destroy_process_group(); return

    del fresh
    if (output/'latest.json').exists():
        pointer = json.loads((output/'latest.json').read_text())
        state = restore(pointer)
        assert state['seen'] == 16*state['step'] <= 240000
        if state['seen'] < 240000:
            error = replay(pointer, state['seen'], state['step'])
            if rank == 0: atomic_json(output/f'resume-{os.environ["SLURM_JOB_ID"]}.json',
                                      dict(checkpoint=pointer, next_update_replay_error=error))
    else:
        validation(rows['validation'][:128], 'validation-000000', a.deadline-1200)
        torch.manual_seed(20261007+rank)
        pointer = checkpoint(state)
    # Reserve endpoint validation and checkpoint time in every bounded segment.
    while state['seen'] < 240000 and not stop(a.deadline-1200):
        tick=time.time(); loss, norm = update(state['seen'], state['step'])
        state['step'] += 1; state['seen'] += 16; state['training_seconds'] += time.time()-tick
        if rank == 0:
            entry = dict(**state, loss=loss, gradient_norm=norm, unique_examples=state['seen'],
                         fraction_official_train=state['seen']/1200000, job_id=os.environ['SLURM_JOB_ID'])
            atomic_json(output/'progress.json', entry)
            with (output/'learning-curve.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
            if state['step'] % 25 == 0: print(json.dumps(entry), flush=True)
        if state['step'] % 2000 == 0:
            pointer = checkpoint(state)
            if not stop(a.deadline-1380):
                validation(rows['validation'][:128], f"validation-{state['step']:06d}", a.deadline-1200)
                restore(pointer)  # Validation must not change the training RNG stream.
    pointer = checkpoint(state)
    full = None
    if state['seen'] == 240000:
        full = validation(rows['validation'], 'selected-validation', a.deadline-90)
        if rank == 0:
            atomic_json(output/'best.json', pointer)
    if rank == 0:
        summary = dict(**state, status='one-pass' if state['seen']==240000 else 'checkpointed-awaiting-next-segment',
            unique_examples=state['seen'], fraction_official_train=state['seen']/1200000,
            effective_epochs=state['seen']/240000, selected_checkpoint=pointer,
            validation={k:v for k,v in full.items() if k!='predictions'} if full else None,
            test_evaluated=False, source_sha256=source_digest(), job_id=os.environ['SLURM_JOB_ID'])
        atomic_json(output/'summary.json', summary)
        if full and full['complete']:
            atomic_json(output/'COMPLETE.json', summary)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
