"""Bounded LFM LoRA experiment. Never select a checkpoint on benchmark test data."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import uuid

import torch
from torch.nn import functional as F

from .data import sha256_file
from .adaptation_data import image_for, prepare
from .models import DEFAULT_REVISIONS, load_backend
from .run import atomic_json, parse_answer, require_allocation, wilson


def answer_labels(prefix, full):
    """Require the expanded image+instruction prefix to be identical at train/eval."""
    n = prefix.shape[-1]
    if prefix.shape[0] != 1 or full.shape[0] != 1 or full.shape[-1] <= n or not torch.equal(prefix, full[:, :n]):
        raise ValueError("Chat template changed the prompt prefix; cannot safely mask the loss")
    labels = full.clone()
    labels[:, :n] = -100
    return labels, n


def encode(processor, image, prompt, answer, device, dtype):
    user = {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    common = dict(tokenize=True, return_dict=True, return_tensors="pt", truncation=False)
    prefix = processor.apply_chat_template([user], add_generation_prompt=True, **common)
    full = processor.apply_chat_template([user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
                                         add_generation_prompt=False, **common)
    labels, start = answer_labels(prefix["input_ids"], full["input_ids"])
    for name in ("pixel_values", "spatial_shapes", "pixel_attention_mask", "image_grid_thw"):
        if name in prefix and not torch.equal(prefix[name], full[name]):
            raise ValueError(f"Training/evaluation image processing differs: {name}")
    suffix = processor.tokenizer.decode(full["input_ids"][0, start:], skip_special_tokens=True).strip()
    if suffix != answer:
        raise ValueError(f"Unexpected supervised continuation: {suffix!r}")
    inputs = {name: tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
              for name, tensor in full.items()}
    return inputs, labels.to(device), start


def suffix_loss(model, inputs, labels, start):
    # Calculate only logits that predict the answer and chat-ending tokens.
    # This is exactly masked causal cross entropy, without projecting prompt positions.
    count = inputs["input_ids"].shape[-1] - start
    import inspect
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if "logits_to_keep" in inspect.signature(base.forward).parameters:
        logits = model(**inputs, use_cache=False, logits_to_keep=count + 1).logits
    else:
        logits = model(**inputs, use_cache=False).logits[:, -(count + 1):]
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, start:].reshape(-1))


def epoch_order(count, seed, epoch):
    indices = list(range(count))
    random.Random(seed + epoch).shuffle(indices)
    return indices


def source_digest():
    value = hashlib.sha256()
    for name in ("adaptation.py", "adaptation_data.py", "models.py", "data.py", "run.py"):
        value.update(name.encode())
        value.update(Path(__file__).with_name(name).read_bytes())
    return value.hexdigest()


def epoch_budget_exhausted(epoch, max_epochs):
    """Zero means wall-time bounded, with no independent epoch cutoff."""
    if max_epochs < 0:
        raise ValueError("max_epochs cannot be negative")
    return bool(max_epochs and epoch >= max_epochs)


def verify_pilot_source(pilot_digest, current_digest, compatibility_path):
    if pilot_digest == current_digest:
        return
    # A narrow, hash-pinned review can reuse a GPU pilot after a scheduling-only
    # change. Never accept arbitrary older pilots or claim the new source ran it.
    review = json.loads(Path(compatibility_path).read_text())
    if (review.get("pilot_source_sha256") != pilot_digest
            or review.get("training_source_sha256") != current_digest
            or review.get("scope") != "stopping-policy-only"
            or review.get("core_functions_unchanged") is not True):
        raise ValueError("Pilot/source compatibility review does not cover this revision")


def save_checkpoint(output, model, optimizer, state):
    """Publish adapter+optimizer+RNG as one immutable directory before its pointer."""
    name = f"checkpoint-{state['step']:06d}-{uuid.uuid4().hex[:8]}"
    staged = output / ("." + name)
    staged.mkdir()
    model.save_pretrained(staged / "adapter", safe_serialization=True, save_embedding_layers=False)
    payload = {"optimizer": optimizer.state_dict(), "state": state,
               "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(payload, staged / "training.pt")
    atomic_json(staged / "state.json", state)
    for path in staged.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    staged.rename(output / name)
    pointer = {"directory": name, "step": state["step"], "seen": state["seen"]}
    atomic_json(output / "latest.json", pointer)
    return pointer


def restore_checkpoint(output, model, optimizer):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    pointer = json.loads((output / "latest.json").read_text())
    directory = (output / pointer["directory"]).resolve()
    if not directory.is_relative_to(output.resolve()):
        raise ValueError("Checkpoint path escaped output")
    weights = load_file(str(directory / "adapter/adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, weights)
    if result.unexpected_keys or any("lora_" in k for k in result.missing_keys):
        raise ValueError("Adapter checkpoint does not match the training model")
    payload = torch.load(directory / "training.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["state"]


def make_train_mode(model):
    model.train()
    # Frozen vision features remain deterministic, including any encoder dropout.
    for name, module in model.named_modules():
        if name.endswith(("vision_tower", "multi_modal_projector", "vision_model", "visual", "connector")):
            module.eval()


def validate(backend, rows, root, prompt, output, name, deadline, stop):
    backend.model.eval()
    predictions, losses = [], []
    started = time.monotonic()
    for row in rows:
        if stop or time.monotonic() >= deadline:
            break
        image = image_for(row, root)
        inputs, labels, start = encode(backend.processor, image, prompt, row["answer"], "cuda", torch.bfloat16)
        with torch.no_grad():
            loss = float(suffix_loss(backend.model, inputs, labels, start))
        if not math.isfinite(loss):
            raise ValueError("Non-finite validation loss")
        raw = backend.predict(image, prompt)
        prediction = parse_answer(raw, row["choices"])
        losses.append(loss)
        predictions.append({"id": row["id"], "answer": row["answer"], "prediction": prediction,
                            "raw_output": raw, "correct": prediction == row["answer"], "loss": loss})
    correct = sum(row["correct"] for row in predictions)
    n = len(predictions)
    result = {"phase": name, "planned": len(rows), "evaluated": n, "complete": n == len(rows),
              "correct": correct, "accuracy": correct / n if n else None,
              "wilson_95": wilson(correct, n), "loss": sum(losses) / n if n else None,
              "invalid": sum(row["prediction"] is None for row in predictions),
              "seconds": time.monotonic() - started, "predictions": predictions}
    atomic_json(output / (name + ".json"), result)
    return result


def run(args):
    require_allocation()
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    deadline = started + args.seconds
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda number, frame: stop.append(number))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.manual_seed(20261007)
    torch.cuda.manual_seed_all(20261007)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.pilot:
            gate = json.loads(args.pilot_result.read_text())
            if gate.get("status") != "passed" or gate.get("revision") != DEFAULT_REVISIONS[args.model]:
                raise ValueError("Training requires a successful pilot of this model revision")
            verify_pilot_source(gate.get("source_sha256"), source_digest(),
                                Path(__file__).with_name("finetune_pilot_compatibility.json"))
        data = prepare(args.data_root, output, args.pilot, args.task)
        backend = load_backend(args.model, revision=DEFAULT_REVISIONS[args.model], dtype="bfloat16", max_new_tokens=16)
        from peft import LoraConfig, get_peft_model
        targets = [name for name, module in backend.model.named_modules()
                   if any(name.startswith(prefix) for prefix in ("model.language_model.", "model.text_model.", "language_model.")) and isinstance(module, torch.nn.Linear)]
        if not targets or any("vision" in name for name in targets):
            raise ValueError("Failed to identify LFM language-only LoRA layers")
        audit_inputs = None
        if args.pilot:
            audit_row = data["train"][0]
            audit_inputs = encode(backend.processor, image_for(audit_row, args.data_root), data["prompt"],
                                  audit_row["answer"], "cuda", torch.bfloat16)
            with torch.no_grad():
                base_loss = float(suffix_loss(backend.model, *audit_inputs))
        model = get_peft_model(backend.model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                        bias="none", target_modules=targets))
        backend.model = model
        if args.pilot:
            model.eval()
            with torch.no_grad():
                initial_loss = float(suffix_loss(model, *audit_inputs))
            if abs(initial_loss - base_loss) > 1e-6:
                raise ValueError("Fresh LoRA adapter changed the frozen base output")
        trainable = [p for p in model.parameters() if p.requires_grad]
        if any("lora_" not in name for name, p in model.named_parameters() if p.requires_grad):
            raise ValueError("A pretrained weight was accidentally unfrozen")
        optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
        config = {"schema": 1, "model": args.model, "task": args.task, "revision": DEFAULT_REVISIONS[args.model],
                  "source_sha256": source_digest(), "data_sha256": sha256_file(output / "split.json"),
                  "pilot": args.pilot, "rank": 16, "alpha": 32, "dropout": 0.05, "targets": targets,
                  "trainable_parameters": sum(p.numel() for p in trainable), "lr": 1e-4, "warmup_steps": 20,
                  "weight_decay": 0.01, "micro_batch": 1, "accumulation": 16, "max_epochs": args.max_epochs,
                  "stopping_policy": "walltime" if not args.max_epochs else "walltime-or-epochs",
                  "deterministic_algorithms": True, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                  "dtype": "bfloat16-base-float32-adapters", "seed": 20261007,
                  "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate", "Pillow")},
                  "backend": backend.metadata}
        if (output / "config.json").exists() and json.loads((output / "config.json").read_text()) != config:
            raise ValueError("Model/data/source/hyperparameters changed on resume")
        atomic_json(output / "config.json", config)
        state = {"step": 0, "seen": 0, "epoch": 0, "offset": 0, "training_seconds": 0.0,
                 "best_accuracy": -1.0, "best_loss": float("inf"), "last_monitor_step": -1}
        if (output / "latest.json").exists():
            state = restore_checkpoint(output, model, optimizer)
        monitor = data["validation"][:data["monitor_count"]]
        if state["step"] == 0 and state["last_monitor_step"] < 0:
            result = validate(backend, monitor, args.data_root, data["prompt"], output, "validation-000000", deadline - 90, stop)
            if not result["complete"]:
                raise RuntimeError("Allocation expired before baseline validation completed")
            state.update(best_accuracy=result["accuracy"], best_loss=result["loss"], last_monitor_step=0)
            pointer = save_checkpoint(output, model, optimizer, state)
            atomic_json(output / "best.json", pointer)
        steps_this_job, training_seconds_this_job, examples_this_job = 0, 0.0, 0
        max_steps = 8 if args.pilot else math.inf
        # Leave time for an end-of-run validation and checkpoint. Scheduler still enforces the hard cap.
        training_deadline = deadline - (150 if args.pilot else 900)
        while state["step"] < max_steps and (args.pilot or not epoch_budget_exhausted(state["epoch"], args.max_epochs)):
            if stop or time.monotonic() >= training_deadline:
                break
            make_train_mode(model)
            order = epoch_order(len(data["train"]), 20261007, state["epoch"])
            selected = order[state["offset"]:state["offset"] + 16]
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = 1e-4 * min(1.0, (state["step"] + 1) / 20)
            torch.cuda.synchronize()
            tick, mean_loss = time.monotonic(), 0.0
            for index in selected:
                row = data["train"][index]
                inputs, labels, answer_start = encode(backend.processor, image_for(row, args.data_root),
                                                     data["prompt"], row["answer"], "cuda", torch.bfloat16)
                loss = suffix_loss(model, inputs, labels, answer_start)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                (loss / len(selected)).backward()
                mean_loss += float(loss.detach()) / len(selected)
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError("A frozen pretrained weight received a gradient")
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            if not any(p.grad is not None and p.grad.count_nonzero().item() for p in trainable):
                raise ValueError("LoRA gradients are all zero")
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - tick
            state["step"] += 1
            state["seen"] += len(selected)
            state["offset"] += len(selected)
            state["training_seconds"] += elapsed
            if state["offset"] == len(order):
                state["epoch"], state["offset"] = state["epoch"] + 1, 0
            steps_this_job += 1
            if steps_this_job > 1:
                training_seconds_this_job += elapsed
                examples_this_job += len(selected)
            entry = dict(state, train_loss=mean_loss, grad_norm=float(norm),
                         lr=optimizer.param_groups[0]["lr"], step_seconds=elapsed,
                         effective_epochs=state["seen"] / len(data["train"]),
                         peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30)
            with (output / "learning-curve.jsonl").open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
            atomic_json(output / "progress.json", entry)
            print(json.dumps(entry), flush=True)
            if state["step"] % 25 == 0 or args.pilot:
                save_checkpoint(output, model, optimizer, state)
            if not args.pilot and state["step"] % 200 == 0 and time.monotonic() < training_deadline - 180:
                result = validate(backend, monitor, args.data_root, data["prompt"], output,
                                  f"validation-{state['step']:06d}", training_deadline, stop)
                if result["complete"]:
                    state["last_monitor_step"] = state["step"]
                    better = (result["accuracy"], -result["loss"]) > (state["best_accuracy"], -state["best_loss"])
                    if better:
                        state.update(best_accuracy=result["accuracy"], best_loss=result["loss"])
                    pointer = save_checkpoint(output, model, optimizer, state)
                    if better:
                        atomic_json(output / "best.json", pointer)
        pointer = save_checkpoint(output, model, optimizer, state)
        if args.pilot:
            if steps_this_job < 8:
                raise RuntimeError("Pilot did not finish its eight optimizer updates")
            # An actual save/load audit verifies portable adapter and optimizer continuation.
            before = [p.detach().clone() for p in trainable]
            saved_rng = torch.cuda.get_rng_state_all()
            restored = restore_checkpoint(output, model, optimizer)
            if restored != state or any(not torch.equal(a, b) for a, b in zip(before, trainable)):
                raise ValueError("Checkpoint round trip changed adapter parameters/state")
            if any(not torch.equal(a, b) for a, b in zip(saved_rng, torch.cuda.get_rng_state_all())):
                raise ValueError("Checkpoint round trip changed RNG")
            # Replay one identical update on both sides of a checkpoint reload.
            # Includes optimizer moments and stochastic LoRA dropout, not just weights.
            def next_update():
                make_train_mode(model)
                optimizer.zero_grad(set_to_none=True)
                loss = suffix_loss(model, *audit_inputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                return float(loss.detach()), [p.detach().clone() for p in trainable]
            expected_loss, expected_parameters = next_update()
            restore_checkpoint(output, model, optimizer)
            actual_loss, actual_parameters = next_update()
            resume_error = max(float((a-b).abs().max()) for a, b in zip(expected_parameters, actual_parameters))
            if abs(expected_loss - actual_loss) > 1e-6 or resume_error > 1e-6:
                raise ValueError(f"Resume changed the next optimizer update: {resume_error}")
            restore_checkpoint(output, model, optimizer)
            rate = examples_this_job / training_seconds_this_job
            result = {"status": "passed", "source_sha256": source_digest(), "revision": config["revision"],
                      "optimizer_updates": state["step"], "examples_per_second": rate,
                      "projected_seconds_per_9000_example_epoch": 9000 / rate,
                      "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "trainable_parameters": config["trainable_parameters"], "elapsed_seconds": time.monotonic() - started,
                      "checkpoint_roundtrip": True, "resume_next_update_max_abs_error": resume_error,
                      "initial_adapter_loss_difference": initial_loss - base_loss, "training_only": True,
                      "note": "Repeated 64-example correctness/throughput pilot; no benchmark accuracy claim. Weights discarded for production."}
            atomic_json(output / "pilot-result.json", result)
            print(json.dumps(result), flush=True)
            return
        if state["last_monitor_step"] != state["step"] and not stop and time.monotonic() < deadline - 150:
            result = validate(backend, monitor, args.data_root, data["prompt"], output,
                              f"validation-{state['step']:06d}", deadline - 90, stop)
            if result["complete"]:
                better = (result["accuracy"], -result["loss"]) > (state["best_accuracy"], -state["best_loss"])
                state["last_monitor_step"] = state["step"]
                if better:
                    state.update(best_accuracy=result["accuracy"], best_loss=result["loss"])
                pointer = save_checkpoint(output, model, optimizer, state)
                if better:
                    atomic_json(output / "best.json", pointer)
        # Evaluate the selected adapter on all reserved validation examples if time permits.
        # A partial result is explicitly marked and cannot silently become a full score.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        best = json.loads((output / "best.json").read_text())
        set_peft_model_state_dict(model, load_file(str(output / best["directory"] / "adapter/adapter_model.safetensors")))
        full = validate(backend, data["validation"], args.data_root, data["prompt"], output,
                        "selected-validation", deadline - 90, stop)
        summary = {"status": "epoch_limit" if epoch_budget_exhausted(state["epoch"], args.max_epochs) else "budget_or_signal_stop", **state,
                   "effective_epochs": state["seen"] / len(data["train"]), "selected_checkpoint": best,
                   "validation": {k: v for k, v in full.items() if k != "predictions"},
                   "test_evaluated": False, "elapsed_seconds": time.monotonic() - started,
                   "job_id": os.environ["SLURM_JOB_ID"], "resumable_checkpoint": pointer}
        atomic_json(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("lfm25", "smolvlm2", "spatialevo"), required=True)
    parser.add_argument("--task", choices=("idrules", "rmts", "pgm"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-result", type=Path)
    parser.add_argument("--seconds", type=int, default=6900)
    parser.add_argument("--max-epochs", type=int, default=0,
                        help="Optional epoch cap; default 0 uses the allocated time budget")
    args = parser.parse_args()
    if args.max_epochs < 0:
        parser.error("--max-epochs cannot be negative")
    if not args.pilot and args.pilot_result is None:
        parser.error("--pilot-result is required for production training")
    run(args)


if __name__ == "__main__":
    main()
