"""Fine-tune a .cact model on tool-calling data and export a new .cact.

What trains:
  * every backbone weight, through the 4-bit quantizer (straight-through), at `lr`,
  * the Hadamard MLP blocks at their own, higher rate `hadamard_lr` (they are small FP16
    tensors, so they are not limited by 4-bit rounding),
  * the runtime LoRA on the attention projections at `lora_lr` (continued when the parent
    model already has one),
with an optional span attention loss (see spans.py). Training rows get varied tool lists and
tool-absent copies first (see tool_lists.py); validation rows stay as they are.

Data parallel over all local devices; gradients are accumulated to `batch_size` rows.
Only the final model is written.
"""
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from spindle import archive
from spindle.model import quantize, runtime_lora
from spindle.model.architecture import SpindleModel, attention_mask
from spindle.tokenizer import Tokenizer
from . import data, spans, tool_lists
from .config import TrainConfig


def _schedule(rate, steps, config):
    warm = max(1, math.ceil(steps * config.warmup_fraction))
    if steps <= warm:
        return optax.linear_schedule(0., rate, warm)
    return optax.warmup_cosine_decay_schedule(0., rate, warm, steps, end_value=rate * config.final_lr_fraction)


def make_optimizer(trainable, steps, config):
    def label(path, _):
        names = [str(getattr(p, 'key', p)) for p in path]
        if names[0] == 'lora':
            return 'lora'
        if 'hadamard_mlp' in names:
            return 'hadamard'
        return 'decay' if names[-1] in ('kernel', 'embedding') or names[-1].startswith('mhc_phi') else 'no_decay'

    def adamw(rate, decay):
        return optax.adamw(_schedule(rate, steps, config), b1=0.9, b2=0.999, eps=1e-8, weight_decay=decay)

    transforms = {'decay': adamw(config.lr, config.weight_decay), 'no_decay': adamw(config.lr, 0.),
                  'hadamard': adamw(config.hadamard_lr, 0.), 'lora': adamw(config.lora_lr, 0.)}
    labels = jax.tree_util.tree_map_with_path(label, trainable)
    return optax.chain(optax.clip_by_global_norm(config.clip), optax.multi_transform(transforms, labels))


def make_loss(model_config, codebook, pad_id, span_weight):
    model = SpindleModel(model_config)
    codebook = jnp.asarray(codebook)

    def loss_sums(trainable, batch, with_spans=True):
        """(weighted loss sum, weight sum); the caller divides once over the whole batch."""
        params = runtime_lora.apply(quantize.cq4_ste(trainable['model'], codebook), trainable['lora'])
        tokens = batch['tokens']
        mask = attention_mask(tokens, model_config.kv_window, pad_id)
        span_total = 0.
        if with_spans and span_weight:
            span = (batch['span_query'], batch['span_start'], batch['span_end'])
            logits, state = model.apply({'params': params}, tokens, mask, span=span, mutable=['intermediates'])
            span_total = sum(jnp.sum(v) for v in jax.tree.leaves(state['intermediates'])) / len(spans.SPAN_LAYERS)
        else:
            logits = model.apply({'params': params}, tokens, mask)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], tokens[:, 1:])
        weights = batch['weights'][:, 1:]
        return jnp.sum(losses * weights) + span_weight * span_total, jnp.sum(weights)

    return loss_sums


def make_step(loss_sums, optimizer, devices):
    def step(trainable, state, batches):
        def accumulate(carry, batch):
            total, count, grads = carry
            (value, weight), grad = jax.value_and_grad(loss_sums, has_aux=True)(trainable, batch)
            return (total + value, count + weight, jax.tree.map(jnp.add, grads, grad)), None

        zero = (jnp.float32(0), jnp.float32(0), jax.tree.map(jnp.zeros_like, trainable))
        (total, count, grads), _ = jax.lax.scan(accumulate, zero, batches)
        if devices > 1:
            total, count, grads = jax.lax.psum((total, count, grads), 'devices')
        grads = jax.tree.map(lambda g: g / jnp.maximum(count, 1.), grads)
        updates, state = optimizer.update(grads, state, trainable)
        return optax.apply_updates(trainable, updates), state, total / jnp.maximum(count, 1.), optax.global_norm(grads)

    return jax.pmap(step, axis_name='devices') if devices > 1 else jax.jit(step)


def _shard(batch, devices, micro):
    """[rows, ...] -> [devices, accumulation, micro, ...] (no device axis for one device)."""
    out = {}
    for name, value in batch.items():
        value = value.reshape(devices, -1, micro, *value.shape[1:])
        out[name] = value[0] if devices == 1 else value
    return out


def _validate(evaluate, trainable, dataset, size):
    total = count = 0.
    for start in range(0, len(dataset), size):
        value, weight = evaluate(trainable, dataset.batch(np.arange(start, min(start + size, len(dataset))), size))
        total, count = total + float(value), count + float(weight)
    return total / max(count, 1.)


def finetune(parent, train_path, output, *, validation_path=None, config=TrainConfig()):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    devices = jax.local_device_count()
    print(f'Devices: {devices} x {jax.devices()[0].platform}', flush=True)
    if config.batch_size % (devices * config.micro_batch):
        raise ValueError('batch_size must be a multiple of devices * micro_batch')

    print(f'Loading {parent}', flush=True)
    params, model_config, codebook, parent_lora = archive.load(parent)
    tokenizer = Tokenizer.from_archive(parent)
    print(f'  {model_config.num_layers} layers, width {model_config.d_model}, '
          f'{"Monarch" if model_config.monarch else "Kronecker"} Hadamard factors, '
          f'KV window {model_config.kv_window}', flush=True)

    print('Preparing data', flush=True)
    rows = data.read_rows(train_path)
    if validation_path:
        validation_rows = data.read_rows(validation_path)
    else:
        order = np.random.default_rng(config.seed).permutation(len(rows))
        held = set(order[:min(config.validation_rows, len(rows) // 10)].tolist())
        validation_rows = [r for i, r in enumerate(rows) if i in held]
        rows = [r for i, r in enumerate(rows) if i not in held]
    if config.tool_variety or config.absent_tools:
        def fits(row):
            return len(data.encode(row, tokenizer)[0]) <= config.max_len
        rows = tool_lists.augment(rows, config.tool_variety, config.absent_tools, config.seed, fits)
    train_set = data.Dataset(rows, tokenizer, max_len=config.max_len, argument_weight=config.argument_weight,
                             no_call_weight=config.no_call_weight, with_spans=config.span_weight > 0, name='train')
    validation_set = (data.Dataset(validation_rows, tokenizer, max_len=config.max_len, name='validation')
                      if validation_rows else None)

    key = jax.random.PRNGKey(config.seed)
    if parent_lora is not None:
        if any(v['A'].shape[-1] != config.lora_rank for v in parent_lora.values()):
            raise ValueError('lora_rank differs from the parent model runtime LoRA')
        print('Continuing the parent model runtime LoRA', flush=True)
        lora = parent_lora
    else:
        lora = runtime_lora.init(params, key, config.lora_rank)
    trainable = jax.tree.map(jnp.asarray, {'model': params, 'lora': lora})
    print(f'Trainable parameters: {sum(x.size for x in jax.tree.leaves(trainable)):,} '
          f'(runtime LoRA {sum(x.size for x in jax.tree.leaves(trainable["lora"])):,})', flush=True)

    steps_per_epoch = math.ceil(len(train_set) / config.batch_size)
    steps = steps_per_epoch * config.epochs
    if config.max_steps:
        steps = min(steps, config.max_steps)
    optimizer = make_optimizer(trainable, steps, config)
    state = optimizer.init(trainable)
    model_config.remat = True
    loss_sums = make_loss(model_config, codebook, tokenizer.pad_id, config.span_weight)
    update = make_step(loss_sums, optimizer, devices)
    evaluate = jax.jit(lambda t, b: loss_sums(t, b, with_spans=False))
    eval_size = devices * config.micro_batch * 4

    log = (output / 'training-log.jsonl').open('w')

    def record(entry):
        log.write(json.dumps(entry) + '\n')
        log.flush()
        print(json.dumps(entry), flush=True)

    if validation_set:
        print('Initial validation', flush=True)
        record({'step': 0, 'validation_loss': _validate(evaluate, trainable, validation_set, eval_size)})

    if devices > 1:
        trainable, state = jax.device_put_replicated((trainable, state), jax.local_devices())
    host = (lambda tree: jax.tree.map(lambda x: x[0], tree)) if devices > 1 else (lambda tree: tree)

    print(f'Training {steps:,} updates ({config.epochs} epoch(s) of {steps_per_epoch:,}), '
          f'{config.batch_size} rows each', flush=True)
    step, losses, train_started = 0, [], time.monotonic()
    for epoch in range(config.epochs):
        order = np.random.default_rng(config.seed + epoch).permutation(len(train_set))
        for start in range(0, len(order), config.batch_size):
            if step >= steps:
                break
            batch = _shard(train_set.batch(order[start:start + config.batch_size], config.batch_size),
                           devices, config.micro_batch)
            if step == 0:
                print('Compiling the training step (first update is slow)', flush=True)
            trainable, state, loss, grad_norm = update(trainable, state, batch)
            loss = float(np.mean(loss))
            if not math.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at update {step + 1}')
            losses.append(loss)
            step += 1
            if step % config.log_every == 0 or step == steps:
                elapsed = time.monotonic() - train_started
                record({'step': step, 'epoch': epoch + 1, 'loss': sum(losses) / len(losses),
                        'grad_norm': float(np.mean(grad_norm)), 'elapsed_min': round(elapsed / 60, 1),
                        'eta_min': round(elapsed / step * (steps - step) / 60, 1)})
                losses = []
            if validation_set and config.eval_every and step % config.eval_every == 0 and step < steps:
                record({'step': step, 'validation_loss': _validate(evaluate, host(trainable), validation_set, eval_size)})

    trainable = jax.device_get(host(trainable))
    if validation_set:
        record({'step': step, 'validation_loss': _validate(evaluate, trainable, validation_set, eval_size)})
    log.close()

    print('Exporting', flush=True)
    model_path = archive.save(parent, jax.tree.map(np.asarray, trainable['model']), model_config,
                              output / 'model.cact', jax.tree.map(np.asarray, trainable['lora']))
    (output / 'run.json').write_text(json.dumps(
        {'parent': str(parent), 'train': str(train_path), 'validation': str(validation_path or ''),
         'train_rows': len(train_set), 'updates': step, 'devices': devices,
         'minutes': round((time.monotonic() - started) / 60, 1), 'config': asdict(config)}, indent=2) + '\n')
    print(f'Saved {model_path} ({model_path.stat().st_size / 1e6:.1f} MB) in '
          f'{(time.monotonic() - started) / 60:.1f} min', flush=True)
    return model_path
