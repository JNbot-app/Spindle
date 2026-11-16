"""Fine-tuning settings (each one is also a `spindle finetune` flag)."""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 1
    max_steps: int = 0            # total updates (0: every row `epochs` times)
    updates_per_round: int = 0    # export the model after every N updates (0: only at the end)
    rounds: int = 0               # stop this run after N rounds (0: run them all); go on later with `resume`
    resume: str = ''              # folder of a stopped run (model.cact + progress.json) to go on from
    max_minutes: float = 0.       # do not start a round that would end after this many minutes (0: no limit)
    batch_size: int = 32          # rows per update
    micro_batch: int = 1          # rows per device per forward pass
    group_rows: int = 8           # rows padded to one length together (similar lengths are grouped)
    max_len: int = 512
    lr: float = 1e-5              # backbone
    hadamard_lr: float = 1e-3
    lora_lr: float = 3e-3
    lora_rank: int = 16           # may be higher than the parent's rank: the extra rows start at zero
    span_weight: float = 0.5      # 0 turns span attention supervision off
    argument_weight: float = 4.0
    no_call_weight: float = 20.0
    tool_variety: float = 0.5     # chance that a call row gets a varied tool list (see tool_lists.py)
    absent_tools: float = 0.02    # tool-absent rows (answer: no call) added, as a share of the training rows
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    final_lr_fraction: float = 0.1
    clip: float = 1.0
    validation_rows: int = 256    # held-out rows taken from the training file when no validation file is given
    log_every: int = 10
    seed: int = 0
