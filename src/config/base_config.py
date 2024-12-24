from omegaconf import DictConfig, OmegaConf
import hydra
from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from .util import register_config

@dataclass
class ProfilerConfig:
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

@register_config(group="trainer", name="base")
@dataclass
class TrainerConfig:
    # ddp setting
    world_size: int = MISSING
    # training setting
    epoch: int = MISSING
    gradient_accum_step: int = 1
    dtype: str = "torch.float32"
    clip_norm: float = 1.0
    # evaluation setting
    eval_n_sample: int = MISSING
    save_n_sample: int = MISSING
    # torch.compile setting
    compile_model: bool = False
    # wandb setting
    enable_wandb: bool = False
    project_name: str = MISSING
    run_name: str = MISSING
    # profiler setting
    enable_profile: bool = False
    profiler_config: ProfilerConfig = field(default_factory=ProfilerConfig)
