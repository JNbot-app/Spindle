"""Fine-tune a .cact model on tool-calling data and export a new .cact.

What trains:
  * every backbone weight, through the CQ quantizer (straight-through) at `weight_bits` (4, or 2
    for a smaller model, or 8 for 2-bit E8 lattice codes; default the parent's width; a 2-bit model
    keeps the embedding and the mHC mixing matrices at 4 bits, as Needle 3 does), at `lr`,
  * the Hadamard MLP blocks at their own, higher rate `hadamard_lr` (they are small FP16
    tensors, so they are not limited by 4-bit rounding),
  * the runtime LoRA on the attention projections at `lora_lr` (continued when the parent
    model already has one),
  * the Q/K/V convolution taps at `conv_lr` (default `lr`); `conv_taps` can widen them first,
    with the extra taps starting at zero,
with an optional span attention loss (see spans.py) and an optional decision term, where a frozen
reasoning `teacher` teaches the call / no-call choice (see decision.py). With decode skip (`skip_layers`, or the
parent's own list), answer tokens skip those layers in training as the engine does at decode, so the model
learns to work without them; the list is saved in the exported archive. Training rows get varied tool lists and
tool-absent copies first (see tool_lists.py); validation rows stay as they are.

Training runs in rounds of `updates_per_round` updates. Each round tokenizes only its own part
of the training file, trains, validates (per `task` tag when the rows have one) and overwrites
`model.cact` and `progress.json`. A run that stopped can go on from there with `resume`. The
learning-rate schedule spans the whole run, so rounds do not restart it.

Data parallel over all local devices; gradients are accumulated to `batch_size` rows, in groups
of rows of similar length so short rows are not padded to the longest one.
"""
import json
import math
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from spindle import archive, lattice
from spindle.model import quantize, runtime_lora
from spindle.model.architecture import SpindleModel, attention_mask
from spindle.tokenizer import Tokenizer
from . import data, decision, spans, tool_lists
from .loss_trend import LossTrend
from .config import TrainConfig

RESUME_WARMUP = 50  # updates of warm-up after a resume (the optimizer state is not saved)


def _schedule(rate, steps, config, start=0):
    warm = max(1, math.ceil(steps * config.warmup_fraction))
    if steps <= warm:
        base = optax.linear_schedule(0., rate, warm)
    else:
        base = optax.warmup_cosine_decay_schedule(0., rate, warm, steps, end_value=rate * config.final_lr_fraction)
    if not start:
        return base
    return lambda count: base(count + start) * jnp.minimum(1., (count + 1) / RESUME_WARMUP)


def make_optimizer(trainable, steps, config, start=0):
    def label(path, _):
        names = [str(getattr(p, 'key', p)) for p in path]
        if names[0] == 'lora':
            return 'lora'
        if 'hadamard_mlp' in names:
            return 'hadamard'
        if names[-1] in ('q_taps', 'k_taps', 'v_taps'):
            return 'conv'
        return 'decay' if names[-1] in ('kernel', 'embedding') or names[-1].startswith('mhc_phi') else 'no_decay'

    def adamw(rate, decay):
        return optax.adamw(_schedule(rate, steps, config, start), b1=0.9, b2=0.999, eps=1e-8, weight_decay=decay)

    transforms = {'decay': adamw(config.lr, config.weight_decay), 'no_decay': adamw(config.lr, 0.),
                  'hadamard': adamw(config.hadamard_lr, 0.), 'lora': adamw(config.lora_lr, 0.),
                  'conv': adamw(config.conv_lr or config.lr, 0.)}
    labels = jax.tree_util.tree_map_with_path(label, trainable)
    return optax.chain(optax.clip_by_global_norm(config.clip), optax.multi_transform(transforms, labels))


def make_loss(model_config, codebook, pad_id, span_weight, distill_weight=0., decisions=None, wide_codebook=None,
              e8=False):
    """(loss_sums, rounding): rounding(model params) gives the CQ offsets that loss_sums adds to them."""
    model = SpindleModel(model_config)
    codebook = jnp.asarray(codebook)

    def rounding(model_params):
        return quantize.cq_offsets(model_params, codebook, wide_codebook, e8)

    def decision_sum(logits, batch, teacher):
        """Cross-entropy of the student's `[]` / `[{"` choice against the teacher's chance of declining."""
        rows = jnp.arange(logits.shape[0])
        t_tokens = batch['teacher_tokens']
        t_logits = model.apply({'params': teacher}, t_tokens, attention_mask(t_tokens, model_config.kv_window, pad_id))
        p = jax.nn.softmax(jax.lax.stop_gradient(t_logits[rows, batch['teacher_pos']]).astype(jnp.float32))
        ids = batch['call_ids']
        call = jnp.sum(jnp.take_along_axis(p, jnp.maximum(ids, 0), axis=1) * (ids >= 0), axis=-1)
        decline = jnp.sum(p[:, jnp.asarray(decisions.declines)], axis=-1)
        target = decline / jnp.maximum(call + decline, 1e-9)
        choice = logits[rows, batch['decide_pos']][:, jnp.asarray([decisions.no_call, decisions.call])]
        choice = jax.nn.log_softmax(choice.astype(jnp.float32))
        return jnp.sum(-(target * choice[:, 0] + (1 - target) * choice[:, 1]) * batch['decide'])

    def loss_sums(trainable, batch, offsets, with_spans=True, teacher=None):
        """(weighted loss sum, weight sum); the caller divides once over the whole batch."""
        params = runtime_lora.apply(jax.tree.map(jnp.add, trainable['model'], offsets), trainable['lora'])
        tokens = batch['tokens']
        mask = attention_mask(tokens, model_config.kv_window, pad_id)
        answer = (batch['weights'] > 0).astype(jnp.float32)      # decode skip applies to answer tokens
        span_total = 0.
        if with_spans and span_weight:
            span = (batch['span_query'], batch['span_start'], batch['span_end'])
            logits, state = model.apply({'params': params}, tokens, mask, span=span, answer=answer,
                                        mutable=['intermediates'])
            span_total = sum(jnp.sum(v) for v in jax.tree.leaves(state['intermediates'])) / len(spans.SPAN_LAYERS)
        else:
            logits = model.apply({'params': params}, tokens, mask, answer=answer)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], tokens[:, 1:])
        weights = batch['weights'][:, 1:]
        total = jnp.sum(losses * weights) + span_weight * span_total
        if teacher is not None and distill_weight:
            total = total + distill_weight * decision_sum(logits, batch, teacher)
        return total, jnp.sum(weights)

    return loss_sums, rounding


def make_steps(loss_sums, rounding, optimizer, devices):
    """(round_weights, accumulate, apply): make the rounding offsets of the weights once per update;
    add the gradients of one group of rows; then make one update."""
    def round_weights(trainable):
        return rounding(trainable['model'])

    def accumulate(trainable, carry, batches, teacher, offsets):
        def one(carry, batch):
            total, count, grads = carry
            (value, weight), grad = jax.value_and_grad(loss_sums, has_aux=True)(trainable, batch, offsets,
                                                                                teacher=teacher)
            return (total + value, count + weight, jax.tree.map(jnp.add, grads, grad)), None

        return jax.lax.scan(one, carry, batches)[0]

    def apply(trainable, state, carry):
        if devices > 1:
            carry = jax.lax.psum(carry, 'devices')
        total, count, grads = carry
        grads = jax.tree.map(lambda g: g / jnp.maximum(count, 1.), grads)
        updates, state = optimizer.update(grads, state, trainable)
        return optax.apply_updates(trainable, updates), state, total / jnp.maximum(count, 1.), optax.global_norm(grads)

    if devices > 1:
        return jax.pmap(round_weights), jax.pmap(accumulate, axis_name='devices'), jax.pmap(apply, axis_name='devices')
    return jax.jit(round_weights), jax.jit(accumulate), jax.jit(apply)


def _shard(batch, devices, micro):
    """[rows, ...] -> [devices, accumulation, micro, ...] (no device axis for one device)."""
    out = {}
    for name, value in batch.items():
        value = value.reshape(devices, -1, micro, *value.shape[1:])
        out[name] = value[0] if devices == 1 else value
    return out


def _validate(evaluate, round_weights, trainable, datasets, size):
    """{task: loss} over the validation sets."""
    offsets = round_weights(trainable)
    result = {}
    for task, dataset in datasets.items():
        total = count = 0.
        length = dataset.padded_length(range(len(dataset)))
        for start in range(0, len(dataset), size):
            value, weight = evaluate(trainable, dataset.batch(range(start, min(start + size, len(dataset))), size, length),
                                     offsets)
            total, count = total + float(value), count + float(weight)
        result[task] = round(total / max(count, 1.), 4)
    return result


def _by_task(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row.get('task', 'all'), []).append(row)
    return groups


def _round_rows(rows, round_index, count):
    """The `count` rows of one round, read in file order (wrapping around at the end)."""
    start = round_index * count % len(rows)
    return [rows[(start + i) % len(rows)] for i in range(min(count, len(rows)))]


def finetune(parent, train_path, output, *, validation_path=None, config=TrainConfig()):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    devices = jax.local_device_count()
    print(f'Devices: {devices} x {jax.devices()[0].platform}', flush=True)
    per_call = devices * config.micro_batch
    if config.group_rows % per_call or config.batch_size % config.group_rows:
        raise ValueError('group_rows must be a multiple of devices * micro_batch, and batch_size of group_rows')

    progress = None
    if config.resume:
        progress = json.loads((Path(config.resume) / 'progress.json').read_text())
        parent = Path(config.resume) / 'model.cact'
        print(f'Resuming after round {progress["round"]} ({progress["updates"]:,} updates) from {parent}', flush=True)
    with archive.Archive(parent) as a:
        taps = a.header[19]
    bits = config.weight_bits or archive.weight_bits(parent)
    if bits not in (2, 4, lattice.E8):
        raise ValueError(f'weight_bits must be 4, 2 or 8 (E8 lattice), not {bits}')
    if config.conv_taps and config.conv_taps != taps:
        widened = Path(tempfile.mkdtemp()) / 'parent.cact'  # not in `output`: only the trained model is kept
        print(f'Widening the Q/K/V convolution from {taps} to {config.conv_taps} taps (new taps start at zero)',
              flush=True)
        parent = archive.widen_taps(parent, widened, config.conv_taps)
    print(f'Loading {parent}', flush=True)
    params, model_config, codebook, parent_lora = archive.load(parent, bits)
    tokenizer = Tokenizer.from_archive(parent)
    print(f'  {model_config.num_layers} layers, width {model_config.d_model}, {"E8 lattice" if bits == lattice.E8 else f"{bits}-bit"} weights '
          f'(parent {archive.weight_bits(parent)}-bit), '
          f'{"Monarch" if model_config.monarch else "Kronecker"} Hadamard factors, '
          f'KV window {model_config.kv_window}', flush=True)
    if config.skip_layers:
        model_config.skip_layers = () if config.skip_layers == 'none' else archive.parse_layers(config.skip_layers)
        if model_config.num_layers - 1 in model_config.skip_layers:
            raise ValueError('the last layer is the read-out layer and cannot be skipped')
    if model_config.skip_layers:
        print(f'  decode skip: answer tokens skip layers {list(model_config.skip_layers)}', flush=True)

    print('Reading data', flush=True)
    rows = data.read_rows(train_path)
    if validation_path:
        validation_rows = data.read_rows(validation_path)
    else:
        order = np.random.default_rng(config.seed).permutation(len(rows))
        held = set(order[:min(config.validation_rows, len(rows) // 10)].tolist())
        validation_rows = [r for i, r in enumerate(rows) if i in held]
        rows = [r for i, r in enumerate(rows) if i not in held]
    validation_sets = {task: data.Dataset(group, tokenizer, max_len=config.max_len, name=f'validation/{task}')
                       for task, group in _by_task(validation_rows).items()}

    steps = config.max_steps or math.ceil(len(rows) * config.epochs / config.batch_size)
    per_round = min(config.updates_per_round or steps, steps)
    rounds = math.ceil(steps / per_round)
    first_round = progress['round'] if progress else 0
    if progress and (progress['total_updates'], progress['updates_per_round']) != (steps, per_round):
        raise ValueError('resume: the run being resumed had a different length or round size')

    key = jax.random.PRNGKey(config.seed)
    if parent_lora is not None:
        parent_rank = parent_lora['q_proj']['A'].shape[-1]
        if parent_rank > config.lora_rank:
            raise ValueError(f'lora_rank {config.lora_rank} is below the parent model runtime LoRA rank {parent_rank}')
        print(f'Continuing the parent model runtime LoRA (rank {parent_rank} -> {config.lora_rank})', flush=True)
        lora = parent_lora if parent_rank == config.lora_rank else runtime_lora.widen(parent_lora, key, config.lora_rank)
    else:
        lora = runtime_lora.init(params, key, config.lora_rank)
    trainable = jax.tree.map(jnp.asarray, {'model': params, 'lora': lora})
    print(f'Trainable parameters: {sum(x.size for x in jax.tree.leaves(trainable)):,} '
          f'(runtime LoRA {sum(x.size for x in jax.tree.leaves(trainable["lora"])):,})', flush=True)

    optimizer = make_optimizer(trainable, steps, config, start=first_round * per_round)
    state = optimizer.init(trainable)
    model_config.remat = True
    teacher = decisions = None
    if config.teacher and config.distill_weight:
        print(f'Loading the reasoning teacher {config.teacher} (decision weight {config.distill_weight})', flush=True)
        teacher, teacher_config, _, teacher_lora = archive.load(config.teacher)
        if (teacher_config.num_layers, teacher_config.d_model, teacher_config.monarch) != \
                (model_config.num_layers, model_config.d_model, model_config.monarch):
            raise ValueError('The teacher must have the same architecture as the model')
        extra = model_config.qkv_conv_taps - teacher_config.qkv_conv_taps
        if extra:  # the teacher runs with the model's settings: give it the same taps, the extra ones zero
            if extra < 0 or not teacher_config.qkv_conv_taps:
                raise ValueError('The teacher has more Q/K/V convolution taps than the model')
            attention = teacher['stack']['layers']['block']['self_attn']
            for name in ('q_taps', 'k_taps', 'v_taps'):
                attention[name] = np.pad(attention[name], ((0, 0), (0, extra), (0, 0)))
        # archive.load already gives the 4-bit values the engine uses, so no quantizer here.
        teacher = jax.tree.map(jnp.asarray, teacher if teacher_lora is None else runtime_lora.apply(teacher, teacher_lora))
        decisions = decision.Decisions(tokenizer)
    loss_sums, rounding = make_loss(model_config, codebook, tokenizer.pad_id, config.span_weight, config.distill_weight, decisions,
                          archive.codebook(parent, 4) if bits != 4 else None, bits == lattice.E8)
    round_weights, accumulate, apply = make_steps(loss_sums, rounding, optimizer, devices)
    evaluate = jax.jit(lambda t, b, offsets: loss_sums(t, b, offsets, with_spans=False))
    host_rounding = jax.jit(lambda t: rounding(t['model']))  # for validation, on the host copy
    eval_size = per_call * 4
    zero = (jnp.float32(0), jnp.float32(0), jax.tree.map(jnp.zeros_like, trainable))

    log = (output / 'training-log.jsonl').open('a' if progress else 'w')

    def record(entry):
        log.write(json.dumps(entry) + '\n')
        log.flush()
        print(json.dumps(entry), flush=True)

    validation_points = []  # (update, mean validation loss) for loss.png
    if validation_sets:
        print('Validation before training', flush=True)
        before = _validate(evaluate, host_rounding, trainable, validation_sets, eval_size)
        validation_points.append((first_round * per_round, float(np.mean(list(before.values())))))
        record({'step': first_round * per_round, 'validation_loss': before})

    if devices > 1:
        # One copy per device: pmap splits the leading axis over the devices on the first call.
        trainable, state, zero, teacher = jax.tree.map(
            lambda x: np.broadcast_to(np.asarray(x), (devices, *np.shape(x))), (trainable, state, zero, teacher))
        if teacher is not None:  # copy it to the devices once, not on every call
            teacher = jax.pmap(lambda t: t)(teacher)
    host = (lambda tree: jax.tree.map(lambda x: x[0], tree)) if devices > 1 else (lambda tree: tree)

    def fits(row):
        return len(data.encode(row, tokenizer)[0]) <= config.max_len

    print(f'Training {steps:,} updates of {config.batch_size} rows in {rounds} round(s) of {per_round:,}', flush=True)
    step, model_path, round_minutes, train_rows = first_round * per_round, None, 0., 0
    trend = LossTrend(start=step)
    last_round = min(rounds, first_round + config.rounds) if config.rounds else rounds
    for round_index in range(first_round, last_round):
        round_started = time.monotonic()
        elapsed = (round_started - started) / 60
        if config.max_minutes and round_minutes and elapsed + round_minutes * 1.05 > config.max_minutes:
            print(f'Stopping before round {round_index + 1}: {elapsed:.0f} min used, a round takes '
                  f'{round_minutes:.0f} min, limit {config.max_minutes:.0f} min', flush=True)
            break
        updates = min(per_round, steps - step)
        print(f'Round {round_index + 1}/{rounds}: preparing data', flush=True)
        chunk = _round_rows(rows, round_index, updates * config.batch_size)
        if config.tool_variety or config.absent_tools:
            # Tool lists are only edited on tool-calling rows: for extraction and classification the
            # one given tool is the task itself.
            fixed = [r for r in chunk if r.get('task') in ('extract', 'classify')]
            chunk = fixed + tool_lists.augment([r for r in chunk if r.get('task') not in ('extract', 'classify')],
                                               config.tool_variety, config.absent_tools,
                                               config.seed + round_index, fits)
        train_set = data.Dataset(chunk, tokenizer, max_len=config.max_len, argument_weight=config.argument_weight,
                                 no_call_weight=config.no_call_weight, with_spans=config.span_weight > 0,
                                 decisions=decisions,
                                 name=f'round {round_index + 1}')
        train_rows += len(train_set)
        rng = np.random.default_rng(config.seed + round_index)
        order = np.concatenate([rng.permutation(len(train_set))
                                for _ in range(math.ceil(updates * config.batch_size / len(train_set)))])
        losses, train_started = [], time.monotonic()
        for index in range(updates):
            picked = order[index * config.batch_size:(index + 1) * config.batch_size].tolist()
            if step == first_round * per_round and index == 0:
                print('Compiling the training step (the first updates are slow)', flush=True)
            carry, offsets = zero, round_weights(trainable)
            for group in train_set.groups(picked, config.group_rows):
                carry = accumulate(trainable, carry, _shard(train_set.batch(group, config.group_rows),
                                                            devices, config.micro_batch), teacher, offsets)
            trainable, state, loss, grad_norm = apply(trainable, state, carry)
            loss = float(np.mean(loss))
            if not math.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at update {step + 1}')
            losses.append(loss)
            trend.add(loss)
            step += 1
            if step % config.log_every == 0 or index + 1 == updates:
                spent = time.monotonic() - train_started
                record({'step': step, 'round': round_index + 1, 'loss': round(sum(losses) / len(losses), 4),
                        **trend.summary(), 'grad_norm': round(float(np.mean(grad_norm)), 3),
                        'seconds_per_update': round(spent / (index + 1), 2),
                        'round_left_min': round(spent / (index + 1) * (updates - index - 1) / 60, 1)})
                losses = []
            chart = trend.chart()
            if chart:
                print(chart, flush=True)

        current = jax.device_get(host(trainable))
        validation = _validate(evaluate, host_rounding, current, validation_sets, eval_size) if validation_sets else {}
        if validation:
            validation_points.append((step, float(np.mean(list(validation.values())))))
        trend.plot(output / 'loss.png', validation_points)
        print(f'Round {round_index + 1}/{rounds}: exporting', flush=True)
        model_path = archive.save(parent, jax.tree.map(np.asarray, current['model']), model_config,
                                  output / 'model.cact', jax.tree.map(np.asarray, current['lora']), bits)
        round_minutes = (time.monotonic() - round_started) / 60
        state_now = {'round': round_index + 1, 'rounds': rounds, 'updates': step, 'total_updates': steps,
                     'updates_per_round': per_round, 'validation_loss': validation,
                     'round_minutes': round(round_minutes, 1),
                     'minutes': round((time.monotonic() - started) / 60, 1)}
        (output / 'progress.json').write_text(json.dumps(state_now, indent=2) + '\n')
        record({'step': step, 'round_done': round_index + 1, 'validation_loss': validation,
                'round_minutes': round(round_minutes, 1), 'model': str(model_path)})
    log.close()

    (output / 'run.json').write_text(json.dumps(
        {'parent': str(parent), 'train': str(train_path), 'validation': str(validation_path or ''),
         'train_rows': train_rows, 'updates': step, 'devices': devices,
         'minutes': round((time.monotonic() - started) / 60, 1), 'config': asdict(config)}, indent=2) + '\n')
    if model_path:
        print(f'Saved {model_path} ({model_path.stat().st_size / 1e6:.1f} MB) after {step:,} updates in '
              f'{(time.monotonic() - started) / 60:.1f} min', flush=True)
    return model_path
