"""CPU distributed test: synchronized gradients/updates equal a full-batch reference."""
import os
import torch
import torch.distributed as dist
from vlm_bench.parallel_adaptation import synchronize_gradients, rank_indices

rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
dist.init_process_group('gloo')
for count in [16, 11]:
    torch.manual_seed(51)
    model = torch.nn.Linear(5, 3)
    reference = torch.nn.Linear(5, 3)
    reference.load_state_dict(model.state_dict())
    x, y = torch.randn(count, 5), torch.randn(count, 3)
    ((reference(x)-y).square().sum()/count).backward()
    for i in rank_indices(0, count, rank, world):
        ((model(x[i])-y[i]).square().sum()/count).backward()
    synchronize_gradients(list(model.parameters()))
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-6)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    torch.optim.AdamW(reference.parameters(), lr=1e-4).step()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-6)
for n in [1, 11, 16, 31]:
    for w in [1, 2, 4, 8]:
        assigned = [i for r in range(w) for i in rank_indices(0, n, r, w)]
        assert sorted(assigned) == list(range(n)) and len(set(assigned)) == n
if rank == 0:
    print('PASS: global-batch gradients, optimizer update and disjoint coverage', flush=True)
dist.destroy_process_group()
