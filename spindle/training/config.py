"""Fine-tuning settings (each one is also a `spindle finetune` flag)."""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 1
    max_steps: int = 0            # stop after this many updates (0: run every epoch)
    batch_size: int = 32          # rows per update
    micro_batch: int = 1          # rows per device per forward pass
    max_len: int = 512
    lr: float = 1e-5              # backbone
    hadamard_lr: float = 1e-3
    lora_lr: float = 3e-3
    lora_rank: int = 16
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
    eval_every: int = 0           # validation every N updates (0: only before and after)
    log_every: int = 10
    seed: int = 0
