from dataclasses import dataclass, asdict
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer as Optimizer
from tqdm.autonotebook import tqdm
import contextlib
import torch
import torch.nn as nn
from typing import Any, Union, Final, Callable, Optional, Tuple
from .util import Timer, MonotonicCounter
from abc import abstractmethod, ABC
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import os
import torch.multiprocessing as mp
from contextlib import nullcontext
from .default_profiler import *
from .util import set_all_seed

# configuration file
from .config import TrainerConfig, check_config, DDPMode

import faulthandler
import signal
faulthandler.enable(all_threads=True)
faulthandler.register(signal.SIGUSR1.value, all_threads=True)
import gc
import random
import numpy as np
# import tracemalloc
# import objgraph

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
        check_config(config)
        # model and optimizer setting
        self.model_fn = model
        self.optimizer_fn = optimizer
        self.scheduler_fn = scheduler
        self.dataloader_sampler_fn = dataloader_sampler

        # constant - derived from config file
        self.config = config
        self.epoch = config.epoch
        self.gradient_accum_step = config.gradient_accum_step
        self.dtype = eval(config.dtype)
        # whether to use ddp or not
        self.use_ddp = self.config.ddp.enable

        # backward compatible
        # try:
        #     self.scaler = torch.cuda.amp.GradScaler()
        # except:
        #     self.scaler = torch.GradScaler()
        self.scaler = torch.amp.GradScaler("cuda" if self.config.use_cuda else "cpu")
        self.clip_norm : float = config.clip_norm
        self.world_size : Final[int] = config.ddp.world_size
    
        # mutable state - Important - we need to preserve them during checkpoint
        self.ep : int = 0
        self.total_step : int = 0
        self.micro_step : int = 0
        # log data, not checkpointed
        self.log_data = {}
        # profiler is not saved
        if self.config.profiler.enable:
            self.profiler_fn = make_default_profiler
        else:
            self.profiler_fn = None

    def setup_ddp(self):
        rank = self.rank
        world_size = self.world_size
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        # initialize the process group
        if not dist.is_initialized() and self.use_ddp:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        if self.device != "cpu":
            torch.cuda.set_device(rank)
        # we use a different seed on each rank
        set_all_seed(rank)
        # === reproducibility ===
        # we give a different seed on every model
        # this won't impact model weight, because DDP will copy weights
        # and this won't impact data sampler
        # we need to apply a different seed
        # otherwise, we always get the same seed on every device, which may break the diffusion model!
        # enable fp32 here
        # TODO : add a flag here
        torch.set_float32_matmul_precision("high")
    @property
    def device(self):
        if self.config.use_cuda:
            return self.rank
        else:
            return "cpu"

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
        self.model = self.model_fn().to(self.device)
        # TODO : change the order of these two functions
        # firstly DDP then torch.compile
        if self.config.compile_model.enable:
            # we can actually use disable here.
            self.model = torch.compile(self.model, dynamic=False, mode=self.config.compile_model.mode)
        # TODO : masked transformer will drop mask at late training stage
        # which causes some parameters unused, so we need to turn on this option here
        self.model = self.model.to(self.device)
        if self.use_ddp:
            self.model = DDP(self.model, find_unused_parameters=True)
        self.optimizer = self.optimizer_fn(self.model)
        self.scheduler = self.scheduler_fn(self.optimizer)

    def run(self):
        if self.config.ddp.mode == DDPMode.spawn:
            if self.world_size > 1:
                mp.spawn(self._run, args=(), nprocs=self.world_size, join=True)
            else:
                self._run(0)
        else:
            raise NotImplementedError("torchrun is not tested yet")

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
        if self.use_ddp:
            dist.destroy_process_group()

    def checkpoint_model(self):
        model = self.model
        for i in range(2):
            if hasattr(model, "module"):
                model = model.module
            if hasattr(model, "_orig_mod"):
                model = model._orig_mod
        return model
    
    # def create_checkpoint(self):
    #     # checkpoint is typically huge
    #     # because we need to save the state for all optimizers and all ranks
    #     trainer_state = {
    #         "total_step" : self.total_step,
    #         "ep" : self.ep
    #     }
    #     seed = {
    #         "torch_seed" : torch.get_rng_state(),
    #         "torch_cuda_seed" : torch.cuda.get_rng_state(),
    #         "python_seed" : random.getstate(),
    #         "numpy_seed" : np.random.get_state()
    #     }
    #     model_opt = {
    #         "model" : self.checkpoint_model().state_dict(),
    #         "optimizer" : self.optimizer.state_dict(),
    #         "scheduler" : self.scheduler.state_dict(),
    #         "scaler" : self.scaler.state_dict(),
    #         "dataloader" : self.dataloader.state_dict()
    #     }
    #     return {
    #         "trainer_state" : trainer_state,
    #         "seed" : seed,
    #         "model_opt" : model_opt
    #     }

    def train(self, checkpoint = None):
        self.on_training_begin()
        # maintain a loss counter here
        self.avg_loss : torch.Tensor = torch.zeros((1,), device=self.device, requires_grad=False)
        _profiler = self.profiler_fn
        if _profiler is None:
            _profiler = nullcontext()
        else:
            _profiler = self.profiler_fn(self.config.profiler)
        # tracemalloc.start(10)
        # self.begin_snapshot = tracemalloc.take_snapshot()
        with _profiler as profiler:
            for ep in tqdm(range(self.ep, self.epoch), position=0, leave=True):
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
                    # if we don't use ddp, then there's no need sync
                    if (self.micro_step + 1) % self.gradient_accum_step == 0 or (not self.use_ddp):
                        # enable sync
                        sync_context = contextlib.nullcontext()
                    else:
                        sync_context = self.model.no_sync()
                    with sync_context:     
                        # this is important
                        # we prepare inputs outside of amp, otherwise inputs will be computed in low precision  
                        model_inputs = self.prepare_input(obj)
                        with torch.autocast(enabled=(self.dtype != torch.float32), device_type="cuda" if self.config.use_cuda else "cpu", dtype=self.dtype):
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
                        if self.use_ddp:
                            dist.all_reduce(self.avg_loss, op=dist.ReduceOp.AVG)
                        self.log_data["total_step"] = self.total_step
                        self.log_data["train/avg_loss"] = self.avg_loss.item()
                        self.log_data["train/epoch"] = self.ep
                        self.log_data["train/grad_scale"] = self.scaler.get_scale()
                        self.log_data["train/grad_norm"] = self.grad_norm.item()
                        self.log_data["train/lr"] = self.scheduler.get_last_lr()[-1]
                        self.on_macro_step_end()
                        if self.rank == 0:
                            if self.config.wandb.enable:
                                wandb.log(self.log_data, commit=True)

                        # we reset the values at the end of microstep for every process
                        # reset it...
                        self.avg_loss = torch.zeros_like(self.avg_loss)
                        # reset logging data
                        self.log_data = {}

                    self.micro_step += 1
                # at the end of one epoch, we synchronize manually
                if self.use_ddp:
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
