from dataclasses import dataclass, asdict
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer as Optimizer
from tqdm.autonotebook import tqdm
import contextlib
import torch
import torch.nn as nn
from typing import Any, Union, Final, Callable, Optional, Tuple
from timer import Timer
from util import MonotonicCounter
from abc import abstractmethod, ABC
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import os
import torch.multiprocessing as mp
from contextlib import nullcontext
from default_profiler import *

import faulthandler
import signal
faulthandler.enable(all_threads=True)
faulthandler.register(signal.SIGUSR1.value, all_threads=True)
import gc
import random
import numpy as np
# import tracemalloc
# import objgraph
@dataclass
class TrainerConfig:
    # ddp setting
    world_size : int
    # training setting
    epoch : int
    gradient_accum_step : int
    dtype : Any
    clip_norm : float
    # evaluation setting
    eval_n_sample : int
    save_n_sample : int
    # torch.compile setting
    compile_model : bool
    # wandb setting
    enable_wandb : bool
    project_name : str
    run_name : str
    # profiler setting
    profiler : Any
    enable_profile : bool
    
class Trainer(ABC):
    rank : Final[int]
    model : Union[nn.Module, DDP]
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler.LRScheduler
    dataloader : DataLoader
    sampler : Optional[DistributedSampler]
    def __init__(self,
                 model : Callable[[], nn.Module],
                 optimizer : Callable[[Union[nn.Module, DDP]], torch.optim.Optimizer],
                 scheduler : Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler],
                 dataloader_sampler : Callable[[int], Tuple[DataLoader, Optional[DistributedSampler]]],
                 config  : TrainerConfig):
        self.model_fn = model
        self.optimizer_fn = optimizer
        self.scheduler_fn = scheduler
        self.dataloader_sampler_fn = dataloader_sampler

        # constant
        self.config = config
        self.epoch = config.epoch
        self.gradient_accum_step = config.gradient_accum_step
        self.dtype = config.dtype
        self.scaler = torch.GradScaler()
        self.clip_norm : float = config.clip_norm
        self.world_size : Final[int] = config.world_size
    
        # mutable state
        self.ep : int = 0
        self.total_step : int = 0
        self.micro_step : int = 0
        self.profiler_fn = config.profiler

    def setup_ddp(self):
        rank = self.rank
        world_size = self.world_size
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        # initialize the process group
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        # === reproducibility ===
        # we give a different seed on every model
        # this won't impact model weight, because DDP will copy weights
        # and this won't impact data sampler
        # we need to apply a different seed
        # otherwise, we always get the same seed on every device, which may break the diffusion model!

        random.seed(rank)
        np.random.seed(rank)
        torch.manual_seed(rank)
        torch.cuda.manual_seed(rank)
        # enable fp32 here
        torch.set_float32_matmul_precision("high")

    def initialize_on_rank(self, rank):
        self.rank = rank
        # disable profiler on other ranks
        if self.rank != 0:
            self.profiler_fn = None
        # set up ddp
        self.setup_ddp()
        # give a chance to setup dataloader
        self.datasetinfo, self.dataloader, self.sampler = self.dataloader_sampler_fn(rank)
        # then we actually create the model on each device
        self.model = self.model_fn().to(rank)
        if self.config.compile_model:
            # we can actually use disable here.
            self.model = torch.compile(self.model, dynamic=False)
        # TODO : masked transformer will drop mask at late training stage
        # which causes some parameters unused, so we need to turn on this option here
        self.model = DDP(self.model.to(rank), find_unused_parameters=True)
        self.optimizer = self.optimizer_fn(self.model)
        self.scheduler = self.scheduler_fn(self.optimizer)

    def run(self):
        if self.world_size > 1:
            mp.spawn(self._run, args=(), nprocs=self.world_size, join=True)
        else:
            self._run(0)

    def _run(self, rank):
        self.initialize_on_rank(rank)
        self.train()

    @abstractmethod
    def prepare_input(self, obj):
        raise NotImplementedError()
    
    @abstractmethod
    def calculate_loss(self, *args, **kwargs):
        raise NotImplementedError()
    
    def on_epoch_begin(self):
        pass

    def on_macro_step_end(self):
        pass

    def on_training_begin(self):
        pass

    def on_training_end(self):
        dist.destroy_process_group()

    def train(self):
        self.on_training_begin()
        self.total_step = 0
        # maintain a loss counter here
        self.avg_loss : torch.Tensor = torch.zeros((1,), device=self.rank, requires_grad=False)
        _profiler = self.profiler_fn
        if _profiler is None:
            _profiler = nullcontext()
        else:
            _profiler = self.profiler_fn()
        # tracemalloc.start(10)
        # self.begin_snapshot = tracemalloc.take_snapshot()
        with _profiler as profiler:
            for ep in tqdm(range(self.epoch), position=0, leave=True):
                self.ep = ep
                self.on_epoch_begin()
                self.optimizer.zero_grad(set_to_none = True)
                self.micro_step = 0
                if self.sampler is not None:
                    self.sampler.set_epoch(ep)
                process_bar = tqdm(self.dataloader, position=0, leave=True)
                for obj in process_bar:
                    self.model.train()
                    self.total_step += 1
                    if (self.micro_step + 1) % self.gradient_accum_step == 0:
                        # enable sync
                        sync_context = contextlib.nullcontext()
                    else:
                        sync_context = self.model.no_sync()
                    with sync_context:     
                        # this is important
                        # we prepare inputs outside of amp, otherwise inputs will be computed in low precision  
                        model_inputs = self.prepare_input(obj)
                        with torch.autocast(enabled=(self.dtype != torch.float32), device_type="cuda", dtype=self.dtype):
                            loss = self.calculate_loss(**model_inputs)
                            loss = loss / self.gradient_accum_step
                        # we must detach the gradient
                        # otherwise the computation graph will leak memory...
                        with torch.no_grad():
                            self.avg_loss[0] += loss.detach()
                        # call backward for every sample
                        self.scaler.scale(loss).backward()
                        # important : no_sync must wrap both forward and backward pass
                        if (self.micro_step + 1) % self.gradient_accum_step == 0:
                            # Unscales the gradients of optimizer's assigned params in-place
                            self.scaler.unscale_(self.optimizer)
                            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                            self.grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                            self.scaler.step(self.optimizer)
                            # Updates the scale for next iteration.
                            self.scaler.update()
                            self.scheduler.step()
                            self.optimizer.zero_grad(set_to_none = True)

                            # step the profiler every macro step
                            # if there is a profiler
                            if self.profiler_fn is not None:
                                profiler.step()
                    # we only perform these operations at the end of microstep
                    if (self.micro_step + 1) % self.gradient_accum_step == 0:
                        dist.all_reduce(self.avg_loss, op=dist.ReduceOp.AVG)
                        self.on_macro_step_end()
                        # we reset the values at the end of microstep for every process
                        # reset it...
                        self.avg_loss = torch.zeros_like(self.avg_loss)
                    self.micro_step += 1
                # at the end of one epoch, we synchronize manually
                dist.barrier()
                # print("Analyzing memory allocation")
                # gc.collect()
                # print(objgraph.show_most_common_types())
                # print(objgraph.show_growth())
                # top_stats = tracemalloc.take_snapshot().compare_to(self.begin_snapshot, 'lineno')
                # print("[ Top differences ]")
                # with open(f"memory-leak-epoch-{self.ep}-{self.rank}.txt", "w") as f:
                #     for stat in top_stats[:20]:       
                #         print("{} new KiB, {} total KiB, {} new, {} total memory blocks: ".format(stat.size_diff/1024, stat.size / 1024, stat.count_diff ,stat.count), file=f)        
                #         for line in stat.traceback.format():            
                #             print(line, file=f)
        self.on_training_end()

import wandb
class DefaultTrainer(Trainer):
    avg_loss : torch.Tensor
    def __init__(self, 
                 model: Callable[[], nn.Module], 
                 optimizer: Callable[[nn.Module | DDP], Optimizer], 
                 scheduler: Callable[[nn.Module | DDP, Optimizer], LRScheduler], 
                 dataloader_sampler: Callable[[int], Tuple[DataLoader, DistributedSampler | None]], 
                 config: TrainerConfig):
        super().__init__(model, optimizer, scheduler, dataloader_sampler, config)
        self.timer : Timer = Timer()
        self.eval_logger = MonotonicCounter(i = config.eval_n_sample, f = self.eval_model)
        self.save_logger = MonotonicCounter(i = config.save_n_sample, f = self.save_model)
        if config.enable_profile and config.profiler is None:
            self.profiler_fn = make_default_profiler
        else:
            self.profiler_fn = None
    
    @abstractmethod
    def eval_model(self, i, is_debug = False):
        pass

    def on_training_begin(self):
        if self.rank == 0:
            wandb.init(
                # Set the project where this run will be logged
                project=self.config.project_name,
                # Track hyperparameters and run metadata
                config=asdict(self.config),
                name = self.config.run_name,
                mode = None if self.config.enable_wandb else 'disabled'
            )
        self.save_model(0, is_debug = True)
        self.eval_model(0, is_debug = True)

    def on_training_end(self):
        if self.rank == 0:
            wandb.finish()
        return super().on_training_end()
    
    def save_model(self, i, is_debug : bool = False):
        if self.rank == 0:
            model_name = f"model-{i}.pt"
            # save the model here, firstly we need to unwrap the DDP module, then the torch.compile module
            module = self.model.module
            # we support both compiled and uncompiled module, so check the existence of _orig_mod
            if hasattr(module, "_orig_mod"):
                module = module._orig_mod
            # Save a model file manually from the current directory:
            if not is_debug:
                torch.save(module, os.path.join(wandb.run.dir, model_name))
                wandb.save(model_name)

    def on_macro_step_end(self):
        total_sample = self.total_step * self.world_size * self.dataloader.batch_size
        self.eval_logger.update(total_sample)
        self.save_logger.update(total_sample)
        if self.rank == 0:
            self.timer.step()
            # processed batches in a macro step
            wandb.log({"sample_per_second" : self.gradient_accum_step * self.world_size * self.dataloader.batch_size * self.timer.rate()}, commit=False)
            wandb.log({"grad_scale" : self.scaler.get_scale()}, commit=False)
            wandb.log({"grad_norm": self.grad_norm.item()}, commit = False)
            wandb.log({"lr": self.scheduler.get_last_lr()[-1]}, commit = False)
            wandb.log({"epoch": self.ep}, commit = False)
            wandb.log({"loss": self.avg_loss[0].item()}, commit=True)
        
