from tiny_trainer import *
import numpy as np
import torch
import hydra
import torch.nn as nn
from torch.utils.data import *
from typing import *
# force the same seed
set_all_seed(114514)

class TestFactory(AbstractTrainerFactory):
    def make_model(self) -> nn.Module:
        return nn.Linear(10, 10, bias=True)
    def make_optimizer(self, model) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=1e-4)
    def make_dataloader(self, rank : int) -> Tuple[Any, DataLoader, Optional[DistributedSampler]]:     
        dataset = [(np.random.randn(10), 0) for i in range(1000)]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
        # rank and world_size is fetched automatically
        # seed must be the same across the cluster
        # sampler = torch.utils.data.DistributedSampler(dataset=dataset, seed=0)
        sampler = None
        return (None, dataloader, sampler)
    def make_scheduler(self, optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        # no change...
        return torch.optim.lr_scheduler.ConstantLR(optimizer=optimizer, factor=1)

class TestTrainer(DefaultTrainer):
    def val_model(self, i, is_debug=False):
        pass
    def eval_model(self, i, is_debug=False):
        pass
    def prepare_input(self, obj):
        return {"x" : torch.tensor(obj[0]).float(), "y" : torch.tensor(obj[1]).long()}
    def calculate_loss(self, x, y):
        return nn.CrossEntropyLoss()(self.model(x), y)
    def on_macro_step_end(self):
        super().on_macro_step_end()
        if self.total_step % 100 == 0:
            print(self.log_data)

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path=".")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    trainer = TestTrainer(factory=TestFactory(), config=cfg.trainer)
    trainer.run()

if __name__ == "__main__":
    my_app()