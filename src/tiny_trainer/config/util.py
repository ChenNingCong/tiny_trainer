from omegaconf import DictConfig, OmegaConf
import hydra
from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

# def generate_template():
#     cfg = TrainerConfig()
#     print(OmegaConf.to_yaml(cfg))

"""
Check whether there are any missing keys in the configuration
"""
def check_config(cfg):
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Got missing keys in config:\n{missing_keys}")

"""
Register a configuration object in the group. The name is typically `base`/
For example, a trainer configuration consists of many parts : profiler, wandb...
We place them into differernt groups so it's easier to compose them.
"""
def register_config(group : str, name : str):
    def wrapper(node_type):
        cs = ConfigStore.instance()
        cs.store(group=group, name=name, node=node_type)
        return node_type
    return wrapper