"""Synchronous multi-GPU PGM LoRA, fixed global batch 16 and one bounded allocation.

Each replica processes disjoint examples; trainable gradients are summed before
clipping and the optimizer update. Frozen weights never enter synchronization.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import time
import uuid

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

from .adaptation import encode, suffix_loss, make_train_mode, validate
from .adaptation_data import image_for
from .data import PGM_PROMPT, sha256_file
from .models import load_backend, DEFAULT_REVISIONS
from .run import require_allocation, atomic_json, wilson
from .pgm_epochs import ORDER_POLICY, next_indices, coverage, validate_resume_schedule, resume_rng_rank


def rank_indices(start, count, rank, world):
    """Strided shards cover a global update once, including an uneven final batch."""
    return list(range(start + rank, start + count, world))


def synchronize_gradients(trainable):
    """Sum gradients already normalized by the global number of examples."""
    grads = [v.grad if v.grad is not None else torch.zeros_like(v) for v in trainable]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for parameter in trainable:
        n = parameter.numel()
        parameter.grad = flat[offset:offset+n].view_as(parameter).clone()
        offset += n


def read_pool(root):
    ready = json.loads((root / 'READY.json').read_text())
    parts = {}
    for part, count in [('train', 240000), ('validation', 1000)]:
        path = root / (part + '.jsonl')
        assert sha256_file(path) == ready['manifests'][path.name]
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == count and len({r['id'] for r in rows}) == count
        assert all(r['dataset'] == 'pgm' and r['split'] == part and
                   r['id'].startswith(f'pgm/neutral/{part}/') and r['prompt'] == PGM_PROMPT for r in rows)
        parts[part] = rows
    assert not {r['id'] for r in parts['train']} & {r['id'] for r in parts['validation']}
    return parts, ready


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['lfm25', 'smolvlm2', 'spatialevo'], required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True, help='Unix deadline, including startup and pilot')
    p.add_argument('--resume-checkpoint', type=Path)
    p.add_argument('--select-final', action='store_true', help='Evaluate the fixed-coverage endpoint, not validation-selected best')
    p.add_argument('--max-epochs', type=int, default=1, help='Total epochs, including the resumed checkpoint')
    p.add_argument('--validation-count', type=int, default=128, choices=[128, 1000])
    p.add_argument('--keep-checkpoints', type=int, default=0, help='0 retains all; >=2 bounds newly created resumable checkpoints')
    p.add_argument('--expected-validation', type=Path, help='Require the resumed checkpoint to reproduce this saved full validation')
    p.add_argument('--allow-world-size-reduction', action='store_true',
                   help='Explicitly migrate a checkpoint to fewer GPUs, preserving global batch and optimizer')
    a = p.parse_args()
    if a.max_epochs < 1 or a.keep_checkpoints == 1 or a.keep_checkpoints < 0:
        p.error('max-epochs must be positive; keep-checkpoints is 0 or >=2')
    require_allocation()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world not in (1, 2, 4, 8) or 16 % world:
        raise ValueError('GPU count must divide global batch 16')
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261007)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=20))
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda number, frame: stopped.append(number))
    started = time.time()
    # Keep a generous final-validation/checkpoint reserve, inside the Slurm limit.
    train_deadline = a.deadline - 1200
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = None
    if rank == 0:
        lock = (output / '.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'latest.json').exists():
            raise ValueError('Fresh campaign only: refusing to overwrite an existing checkpoint')
    rows, ready = read_pool(a.data_root)
    target_exposures = len(rows['train']) * a.max_epochs
    backend = load_backend(a.model, revision=DEFAULT_REVISIONS[a.model], device=f'cuda:{local_rank}',
                           dtype='bfloat16', max_new_tokens=16)
    targets = [name for name, module in backend.model.named_modules()
        if any(name.startswith(prefix) for prefix in ('model.language_model.', 'model.text_model.', 'language_model.'))
        and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError('No language LoRA targets found')

    def inputs(row):
        path = (a.data_root / row['images'][0]).resolve()
        assert sha256_file(path) == row['metadata']['contact_sheet_sha256']
        return encode(backend.processor, image_for(row, a.data_root), PGM_PROMPT,
                      row['answer'], f'cuda:{local_rank}', torch.bfloat16)

    audit_input = inputs(rows['train'][rank])
    with torch.no_grad():
        base_loss = float(suffix_loss(backend.model, *audit_input))
    model = get_peft_model(backend.model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                  bias='none', target_modules=targets))
    backend.model = model
    trainable = [v for v in model.parameters() if v.requires_grad]
    if any(v.requires_grad and 'lora_' not in n for n, v in model.named_parameters()):
        raise ValueError('A base weight was unfrozen')
    for parameter in trainable:
        dist.broadcast(parameter.data, src=0)
    model.eval()
    with torch.no_grad():
        initial_difference = abs(float(suffix_loss(model, *audit_input)) - base_loss)
    if initial_difference > 1e-6:
        raise ValueError('Fresh adapter changed base output')
    fresh = {n: v.detach().cpu().clone() for n, v in model.named_parameters() if v.requires_grad}

    def optimizer_new():
        return torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)

    optimizer = optimizer_new()
    torch.manual_seed(20261007 + rank)

    def collective_stop(deadline):
        value = torch.tensor(int(bool(stopped) or time.time() >= deadline), device=f'cuda:{local_rank}')
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return bool(value.item())

    def update(indices, step):
        make_train_mode(model)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group['lr'] = 1e-4 * min(1.0, (step + 1) / 20)
        loss_total = torch.zeros((), device=f'cuda:{local_rank}')
        for index in indices[rank::world]:
            loss = suffix_loss(model, *inputs(rows['train'][index]))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            (loss / len(indices)).backward()
            loss_total += loss.detach().float() / len(indices)
        # SUM because each rank contributes its share divided by the global batch.
        # A flat buffer avoids hundreds of tiny collective launches per update.
        synchronize_gradients(trainable)
        if any(v.grad is not None for v in model.parameters() if not v.requires_grad):
            raise ValueError('Frozen base received a gradient')
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True))
        optimizer.step()
        dist.all_reduce(loss_total)
        return float(loss_total), norm

    def checkpoint(state, folder=output):
        rng = {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(local_rank)}
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(rng, gathered, dst=0)
        pointer = None
        if rank == 0:
            folder.mkdir(parents=True, exist_ok=True)
            name = f"checkpoint-{state['step']:06d}-{uuid.uuid4().hex[:8]}"
            staged = folder / ('.' + name)
            staged.mkdir()
            model.save_pretrained(staged / 'adapter', safe_serialization=True, save_embedding_layers=False)
            torch.save({'optimizer': optimizer.state_dict(), 'state': state,
                        'rank_rng': gathered, 'world_size': world}, staged / 'training.pt')
            atomic_json(staged / 'state.json', state)
            for path in staged.rglob('*'):
                if path.is_file():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
            staged.rename(folder / name)
            pointer = {'directory': name, 'step': state['step'], 'seen': state['seen']}
            atomic_json(folder / 'latest.json', pointer)
        objects = [pointer]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]

    def restore(folder, pointer):
        directory = folder / pointer['directory']
        result = set_peft_model_state_dict(model, load_file(str(directory / 'adapter/adapter_model.safetensors')))
        assert not result.unexpected_keys and not any('lora_' in k for k in result.missing_keys)
        payload = torch.load(directory / 'training.pt', map_location='cpu', weights_only=True)
        rng_rank = resume_rng_rank(payload['world_size'], world, rank, a.allow_world_size_reduction)
        assert len(payload['rank_rng']) == payload['world_size']
        optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['rank_rng'][rng_rank]['torch'])
        torch.cuda.set_rng_state(payload['rank_rng'][rng_rank]['cuda'], local_rank)
        return payload['state']

    def prune_checkpoints():
        # Only this new campaign's own checkpoints; never modify the resume source.
        if rank == 0 and a.keep_checkpoints:
            protected = {json.loads((output/name).read_text())['directory']
                         for name in ('latest.json', 'best.json') if (output/name).exists()}
            candidates = sorted((p for p in output.glob('checkpoint-*') if p.is_dir()),
                                key=lambda p: p.stat().st_mtime_ns, reverse=True)
            keep = protected | {p.name for p in candidates[:a.keep_checkpoints]}
            for directory in candidates:
                if directory.name not in keep:
                    assert directory.parent == output and not directory.is_symlink()
                    shutil.rmtree(directory)

    # Gate production on real synchronized updates and deterministic next-update replay.
    tick = time.time()
    for step in range(8):
        if collective_stop(train_deadline):
            raise RuntimeError('Allocation exhausted during correctness pilot')
        update(list(range(step * 16, (step + 1) * 16)), step)
    pilot_seconds = time.time() - tick
    flat = torch.cat([v.detach().reshape(-1) for v in trainable])
    reference = flat.clone()
    dist.broadcast(reference, src=0)
    replica_error = float((reference - flat).abs().max())
    assert replica_error <= 1e-6
    pointer = checkpoint({'step': 8, 'seen': 128}, output.parent / 'parallel-pilot')
    expected_loss, _ = update(list(range(128, 144)), 8)
    expected = [v.detach().clone() for v in trainable]
    restore(output.parent / 'parallel-pilot', pointer)
    actual_loss, _ = update(list(range(128, 144)), 8)
    replay_error = max(float((x - v).abs().max()) for x, v in zip(expected, trainable))
    assert replay_error <= 1e-6 and abs(expected_loss - actual_loss) <= 1e-6
    if rank == 0:
        atomic_json(output.parent / 'pilot-result.json', {'status': 'passed', 'world_size': world,
            'examples_per_second': 128 / pilot_seconds, 'replica_max_abs_error': replica_error,
            'resume_next_update_max_abs_error': replay_error, 'initial_loss_difference': initial_difference,
            'global_batch': 16, 'revision': DEFAULT_REVISIONS[a.model]})
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data.copy_(fresh[name].to(parameter.device))
    del fresh, expected, flat, reference
    optimizer = optimizer_new()
    torch.manual_seed(20261007 + rank)

    resume_state = None
    resume_provenance = None
    if a.resume_checkpoint:
        directory = a.resume_checkpoint.resolve()
        old_config = json.loads((directory.parent / 'config.json').read_text())
        resume_rng_rank(old_config['world_size'], world, rank, a.allow_world_size_reduction)
        for key, expected in [('model', a.model), ('revision', DEFAULT_REVISIONS[a.model]),
                              ('global_batch', 16), ('rank', 16),
                              ('alpha', 32), ('dropout', 0.05), ('lr', 1e-4),
                              ('warmup_steps', 20), ('weight_decay', 0.01)]:
            assert old_config[key] == expected, (key, old_config[key], expected)
        assert old_config['data']['manifests'] == ready['manifests']
        pointer = {'directory': directory.name}
        resume_state = restore(directory.parent, pointer)
        assert resume_state == json.loads((directory / 'state.json').read_text())
        validate_resume_schedule(old_config, resume_state['seen'], resume_state['step'],
                                 len(rows['train']), a.max_epochs)
        # Test the very next resumed update, then rewind weights, Adam moments and RNG.
        indices = next_indices(resume_state['seen'], len(rows['train']), a.max_epochs)
        first_loss, _ = update(indices, resume_state['step'])
        expected = [v.detach().clone() for v in trainable]
        restore(directory.parent, pointer)
        second_loss, _ = update(indices, resume_state['step'])
        error = max(float((x-v).abs().max()) for x,v in zip(expected, trainable))
        assert error <= 1e-6 and abs(first_loss-second_loss) <= 1e-6
        restore(directory.parent, pointer)
        del expected
        resume_provenance = {'directory': str(directory), 'state': resume_state.copy(),
            'saved_world_size': old_config['world_size'], 'current_world_size': world,
            'rng_mapping': 'retained-rank-index',
            'training_sha256': sha256_file(directory/'training.pt'),
            'adapter_sha256': sha256_file(directory/'adapter/adapter_model.safetensors'),
            'next_update_replay_max_error': error, 'next_update_loss_difference': abs(first_loss-second_loss)}
        if rank == 0:
            atomic_json(output/'resume-audit.json', resume_provenance)

    def validation(selected, name, deadline):
        result = validate(backend, selected[rank::world], a.data_root, PGM_PROMPT,
                          output, f'{name}-rank{rank}', time.monotonic() + max(0, deadline-time.time()), stopped)
        results = [None] * world
        dist.all_gather_object(results, result)
        preds = [x for r in results for x in r['predictions']]
        n, correct = len(preds), sum(x['correct'] for x in preds)
        merged = {'phase': name, 'planned': len(selected), 'evaluated': n,
                  'complete': n == len(selected), 'correct': correct,
                  'accuracy': correct/n if n else None,
                  'loss': sum(x['loss'] for x in preds)/n if n else None,
                  'invalid': sum(x['prediction'] is None for x in preds),
                  'wilson_95': wilson(correct, n), 'predictions': preds}
        if rank == 0:
            atomic_json(output / (name+'.json'), merged)
        return merged

    config = {'model': a.model, 'task': 'pgm', 'revision': DEFAULT_REVISIONS[a.model],
        'world_size': world, 'global_batch': 16, 'micro_batch': 1, 'rank': 16, 'alpha': 32,
        'dropout': 0.05, 'lr': 1e-4, 'warmup_steps': 20, 'weight_decay': 0.01,
        'train_pool': 240000, 'official_train_count': 1200000, 'max_epochs': a.max_epochs,
        'stopping_policy': 'single-allocation-time-or-epoch-budget', 'fresh_base': not bool(a.resume_checkpoint),
        'resume': resume_provenance, 'selection_policy': 'fixed-coverage-final' if a.select_final else 'validation-best',
        'seed': 20261007, 'training_order': ORDER_POLICY,
        'validation_count': a.validation_count, 'validation_every_updates': 1000,
        'keep_checkpoints': a.keep_checkpoints,
        'synchronization': 'SUM of global-batch-normalized FP32 LoRA gradients',
        'deterministic_algorithms': True, 'deadline': a.deadline,
        'allow_world_size_reduction': a.allow_world_size_reduction,
        'topology_migration_policy': 'Preserve optimizer and global batch; retained ranks reuse saved RNG. Different topology changes dropout trajectory.',
        'source_sha256': sha256_file(Path(__file__)), 'data': ready, 'backend': backend.metadata}
    if rank == 0:
        atomic_json(output / 'config.json', config)
    baseline = validation(rows['validation'][:a.validation_count], f"validation-{resume_state['step'] if resume_state else 0:06d}", train_deadline)
    if not baseline['complete']:
        raise RuntimeError('Baseline validation did not complete')
    if a.expected_validation:
        expected_validation = json.loads(a.expected_validation.read_text())
        old = {r['id']: r for r in expected_validation['predictions']}
        new = {r['id']: r for r in baseline['predictions']}
        assert expected_validation['complete'] and set(old) == set(new)
        assert all(old[k]['answer'] == new[k]['answer'] and old[k]['prediction'] == new[k]['prediction'] for k in old)
        loss_difference = abs(expected_validation['loss'] - baseline['loss'])
        assert loss_difference <= 1e-6, loss_difference
        if rank == 0:
            atomic_json(output/'baseline-reproduction.json', {'status':'passed', 'questions':len(new),
                'loss_difference':loss_difference, 'accuracy':baseline['accuracy'],
                'source_sha256':sha256_file(a.expected_validation)})
    state = {'step': 0, 'seen': 0, 'training_seconds': 0.0, 'best_accuracy': baseline['accuracy'],
             'best_loss': baseline['loss']}
    if resume_state is not None:
        state = {**resume_state, 'best_accuracy': baseline['accuracy'], 'best_loss': baseline['loss']}
        # Validation must not perturb the saved training RNG stream.
        restore(a.resume_checkpoint.resolve().parent, {'directory': a.resume_checkpoint.name})
    best = checkpoint(state)
    if rank == 0:
        atomic_json(output/'best.json', best)
    start_seen = state['seen']
    while state['seen'] < target_exposures and not collective_stop(train_deadline):
        indices = next_indices(state['seen'], len(rows['train']), a.max_epochs)
        tick = time.time()
        loss, norm = update(indices, state['step'])
        state['step'] += 1
        state['seen'] += len(indices)
        state['training_seconds'] += time.time() - tick
        if rank == 0:
            entry = {**state, 'loss': loss, 'gradient_norm': norm, **coverage(state['seen']),
                     'additional_examples_this_job': state['seen']-start_seen, 'elapsed_seconds': time.time()-started}
            atomic_json(output/'progress.json', entry)
            with (output/'learning-curve.jsonl').open('a') as out:
                out.write(json.dumps(entry)+'\n')
            if state['step'] % 25 == 0:
                print(json.dumps(entry), flush=True)
        if state['step'] % 1000 == 0:
            pointer = checkpoint(state)
            if not collective_stop(train_deadline-180):
                v = validation(rows['validation'][:a.validation_count], f"validation-{state['step']:06d}", train_deadline)
                if v['complete'] and (v['accuracy'], -v['loss']) > (state['best_accuracy'], -state['best_loss']):
                    state.update(best_accuracy=v['accuracy'], best_loss=v['loss'])
                    best = checkpoint(state)
                    if rank == 0:
                        atomic_json(output/'best.json', best)
            prune_checkpoints()
    pointer = checkpoint(state)
    if not collective_stop(a.deadline-900):
        v = validation(rows['validation'][:a.validation_count], f"validation-{state['step']:06d}", a.deadline-900)
        if v['complete'] and (v['accuracy'], -v['loss']) > (state['best_accuracy'], -state['best_loss']):
            state.update(best_accuracy=v['accuracy'], best_loss=v['loss'])
            best = checkpoint(state)
            if rank == 0:
                atomic_json(output/'best.json', best)
    if a.select_final:
        if state['seen'] != target_exposures:
            raise RuntimeError('Fixed-coverage endpoint not reached; dependent test must not run')
        best = pointer
        if rank == 0:
            atomic_json(output/'best.json', best)
    prune_checkpoints()
    # Explicitly measure the latest endpoint as well as the validation-selected best.
    # This distinguishes a saved checkpoint from the one whose accuracy is reported.
    last_full = validation(rows['validation'], 'latest-validation', a.deadline-600) if a.max_epochs > 1 else None
    set_peft_model_state_dict(model, load_file(str(output/best['directory']/'adapter/adapter_model.safetensors')))
    full = validation(rows['validation'], 'selected-validation', a.deadline-90)
    if rank == 0:
        atomic_json(output/'summary.json', {**state, 'status': 'epoch-budget-complete' if state['seen']==target_exposures else 'allocation-budget-stop',
            **coverage(state['seen']), 'additional_examples_this_job': state['seen']-start_seen,
            'target_exposures': target_exposures, 'selected_checkpoint': best,
            'resumable_checkpoint': pointer, 'validation': {k:v for k,v in full.items() if k!='predictions'},
            'latest_validation': {k:v for k,v in last_full.items() if k!='predictions'} if last_full else None,
            'world_size': world, 'test_evaluated': False, 'elapsed_seconds': time.time()-started,
            'job_id': os.environ['SLURM_JOB_ID']})
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
