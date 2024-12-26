from omegaconf import DictConfig, OmegaConf
import hydra
from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from .util import register_config

from enum import Enum

@dataclass
class ProfilerConfig:
    enable: bool = False
    # name of the saved trace file
    filename: str = MISSING
    # profiling setting
    profile_memory: bool = True
    record_shapes: bool = True
    with_flops: bool = True
    with_stack: bool = False
    # scheduler setting
    skip_first: int = 5
    wait: int = 2
    warmup: int = 2
    active: int = 5
    repeat: int = 1
    # whether to upload to wandb
    upload_to_wandb: bool = False

@dataclass
class WandbConfig:
    enable: bool = False
    project_name: str = MISSING
    run_name: str = MISSING

@dataclass
class TorchCompileConfig:
    enable: bool = False
    mode: str = "default"

"""
There are two modes of DDP:
1. (spawn) Either we create new processes by forking current process
2. (torchrun) Or we directly launching new processes.
Why do we want to support two modes?
Because it's easier to debug 1 in jupyter notebook (by calling trainer.run()).
But multi-node training is difficult in mode 1 (need to manually set the RANK)
That's why we have two modes here
"""
@dataclass
class DDPMode(str, Enum):
    spawn = "spawn"
    torchrun = "torchrun"

"""
Only need to provide a world_size here
Because either the processes are created by torchrun, or we allocate rank automatically
"""
@dataclass
class DDPConfig:
    enable: bool = True
    mode: DDPMode = field(default_factory = lambda : DDPMode.spawn)
    world_size: int = MISSING

@register_config(group="trainer", name="base")
@dataclass
class TrainerConfig:
    # ddp setting
    ddp: DDPConfig = field(default_factory=DDPConfig)
    # whether use gpu for training
    use_cuda: bool = True
    # training setting
    epoch: int = MISSING
    # support for gradient accumulation
    gradient_accum_step: int = 1
    # support for mixed precision training
    dtype: str = "torch.float32"
    # whether to scale the loss using amp.GradScaler
    use_amp_scaler : bool = True
    # support for norm clipping
    clip_norm: float = 1.0
    # validation and evaluation setting
    eval_n_sample: int = MISSING
    save_n_sample: int = MISSING
    val_n_sample: int = MISSING
    # torch.compile setting
    compile_model: TorchCompileConfig = field(default_factory=TorchCompileConfig)
    # wandb setting
    wandb: WandbConfig = field(default_factory=WandbConfig)
    # profiler setting
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
