from torch.profiler import profile, record_function, ProfilerActivity, schedule
import os
import socket
import time
from typing import *
import glob
import wandb
import tempfile
import torch
"""
Copied fromm https://pytorch.org/docs/stable/_modules/torch/profiler/profiler.html#tensorboard_trace_handler
"""
class TensorBoardTracerHandler:
    def __init__(self, dir_name: str, worker_name: Optional[str] = None, use_gzip: bool = False):
        self.dir_name = dir_name
        self.worker_name = worker_name
        self.use_gzip = use_gzip
        
    def __call__(self, prof) -> None:
        dir_name = self.dir_name
        worker_name = self.worker_name
        use_gzip = self.use_gzip
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # Use nanosecond here to avoid naming clash when exporting the trace
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        prof.export_chrome_trace(os.path.join(dir_name, file_name))

class WandbTracerHandler(TensorBoardTracerHandler):
    def __init__(self, worker_name: Optional[str] = None, use_gzip: bool = False):
        self.dir = tempfile.TemporaryDirectory()
        dir_name = self.dir.name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        super().__init__(dir_name, worker_name, use_gzip)
    def __call__(self, prof) -> None:
        super().__call__(prof)
        profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
        profile_art.add_file(glob.glob(os.path.join(self.dir_name, "*.pt.trace.json"))[0], "trace.pt.trace.json")
        wandb.run.log_artifact(profile_art)
        
def make_default_profiler():
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True, 
        record_shapes=True,
        with_flops=True,
        # add stack trace here
        with_stack=True,
        schedule = schedule(
                    skip_first=5,
                    wait=2,
                    warmup=2,
                    active=5,
                    repeat=1),
        # Workaround by https://github.com/pytorch/pytorch/issues/100253
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        on_trace_ready = WandbTracerHandler())
        
        

