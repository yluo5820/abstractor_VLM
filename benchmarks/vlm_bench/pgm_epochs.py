"""Deterministic repeated passes over the unchanged nested PGM training pool."""
from functools import lru_cache
import random

ORDER_POLICY = 'nested-hash-first-pass; seeded-permutation-on-repeat-v1'


def resume_rng_rank(saved_world, world, rank, allow_reduction=False):
    """Explicit topology migration; retained ranks reuse their saved RNG stream.

    Global batches and optimizer state are unchanged, but example-to-rank
    assignment changes, so dropout does not reproduce the old topology's path.
    Same-topology checkpoint replay remains exact and is tested by the trainer.
    """
    if saved_world not in (1, 2, 4, 8) or world not in (1, 2, 4, 8) or not 0 <= rank < world:
        raise ValueError('Invalid saved/current GPU topology')
    if saved_world != world and not (allow_reduction and world < saved_world):
        raise ValueError('Changing GPU count requires explicit reduction authorization')
    return rank


@lru_cache(maxsize=2)
def epoch_order(count, epoch, seed=20261007):
    if count <= 0 or epoch < 0:
        raise ValueError('Invalid training pool or epoch')
    order = list(range(count))
    if epoch:
        random.Random(seed + epoch).shuffle(order)
    return tuple(order)


def next_indices(seen, count, max_epochs, batch=16, seed=20261007):
    if seen < 0 or max_epochs < 1 or batch < 1 or count < 1:
        raise ValueError('Invalid schedule')
    if seen >= count * max_epochs:
        return []
    epoch, offset = divmod(seen, count)
    return list(epoch_order(count, epoch, seed)[offset:min(offset + batch, count)])


def coverage(seen, count=240000, official_count=1200000):
    return {'unique_examples': min(seen, count), 'effective_epochs': seen/count,
            'fraction_official_train': min(seen, count)/official_count,
            'epoch': seen//count, 'epoch_offset': seen % count,
            'example_exposures': seen}


def validate_resume_schedule(config, seen, step, count, max_epochs, seed=20261007):
    if not (0 <= seen == step * 16 < count * max_epochs):
        raise ValueError('Resume checkpoint is outside the requested epoch budget')
    policy = config['training_order']
    if policy not in ('nested hash order, no repeats', ORDER_POLICY):
        raise ValueError('Unknown training order; refusing to change resume semantics')
    if seen > count and policy != ORDER_POLICY:
        raise ValueError('Repeated-epoch checkpoint is missing its ordering policy')
    if config['seed'] != seed or config['train_pool'] != count:
        raise ValueError('Resume checkpoint belongs to a different training pool or seed')
