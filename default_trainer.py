from .abstract_trainer import *
import wandb
from omegaconf import OmegaConf

class AbstractTrainerFactory(ABC):
    @abstractmethod
    def make_model(self) -> nn.Module:
        raise NotImplementedError()
    @abstractmethod
    def make_optimizer(self, model) -> torch.optim.Optimizer:
        raise NotImplementedError()
    @abstractmethod
    def make_dataloader(self, rank : int) -> Tuple[Any, DataLoader, Optional[DistributedSampler]]:     
        raise NotImplementedError()
    @abstractmethod
    def make_scheduler(self, optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        raise NotImplementedError()

class DefaultTrainer(Trainer):
    avg_loss : torch.Tensor
    def __init__(self, 
                 factory: AbstractTrainerFactory,
                 config: TrainerConfig):
        model = factory.make_model
        optimizer = factory.make_optimizer
        scheduler = factory.make_scheduler
        dataloader_sampler = factory.make_dataloader
        super().__init__(model, optimizer, scheduler, dataloader_sampler, config)
        self.timer : Timer = Timer()
        self.eval_logger = MonotonicCounter(i = config.eval_n_sample, f = self.eval_model)
        self.save_logger = MonotonicCounter(i = config.save_n_sample, f = self.save_model)
    
    @abstractmethod
    def eval_model(self, i, is_debug = False):
        pass

    def on_training_begin(self):
        if self.rank == 0:
            wandb.init(
                # Set the project where this run will be logged
                project=self.config.project_name,
                # Track hyperparameters and run metadata
                config=OmegaConf.to_container(self.config),
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
        