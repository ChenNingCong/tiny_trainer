
import trainer
from trainer import DefaultTrainer
from trainer import *
import torch.nn.functional as F
from torchvision.transforms import v2
import torchvision
from dataclasses import dataclass
import torch

from streaming.vision.base import StreamingDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
import streaming
from dataclasses import dataclass

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

from typing import *
@dataclass
class ImageDatasetInfo:
    dataset : Any
    is_streaming_dataset : bool
    image_shape : Tuple[int, int, int]
    classes : List[str]

    is_latent : bool
    vae : Any
    vae_preprocessor : Any
    vae_postprocessor : Any
    class_condition : bool

def get_dataset(rank : int, dataset_type, data_dir, temp_dir, batch_size : int) -> ImageDatasetInfo:
    # the batch_size here is per device batch_size
    image_shape = None
    is_streaming_dataset = False
    classes = None
    is_latent = False
    vae = None
    vae_preprocessor = None
    vae_postprocessor = None
    class_condition = True
    if dataset_type == "mnist":    
        transform = v2.Compose(
            [v2.ToImage(),
             v2.ToDtype(torch.float32, scale=True),
             v2.ToPureTensor(),
             v2.Normalize(mean=[0.5], std=[0.5])], 
            )
        dataset = torchvision.datasets.MNIST(data_dir, download=True, transform=transform)
    elif dataset_type == "fashion-mnist":
        transform = v2.Compose(
            [v2.ToImage(),
             v2.RandomHorizontalFlip(0.5),
             v2.ToDtype(torch.float32, scale=True),
             v2.ToPureTensor(),
             v2.Normalize(mean=[0.5], std=[0.5])] 
            )
        dataset = torchvision.datasets.FashionMNIST(data_dir, download=True, transform=transform)
    elif dataset_type == "cifar-10":
        transform = v2.Compose(
            [v2.ToImage(),
             v2.RandomHorizontalFlip(0.5),
             v2.ToDtype(torch.float32, scale=True),
             v2.ToPureTensor(),
             v2.Normalize(mean=[0.5], std=[0.5])], 
            )
        dataset = torchvision.datasets.CIFAR10(data_dir, download=True, transform=transform)
    elif dataset_type == "imagenet-32":
        from imagenet import ImageNetDownSample
        transform = v2.Compose(
            [v2.ToImage(),
             v2.RandomHorizontalFlip(0.5),
             v2.ToDtype(torch.float32, scale=True),
             v2.ToPureTensor(),
             v2.Normalize(mean=[0.5], std=[0.5])], 
            )
        dataset = ImageNetDownSample(os.path.join(data_dir, "Imagenet32"), transform=transform)
    elif dataset_type == 'imagenet.uint8':
        is_streaming_dataset = True
        image_shape = (4, 32, 32)
        is_latent = True  
        class uint8(Encoding):
            def encode(self, obj: Any) -> bytes:
                return obj.tobytes()

            def decode(self, data: bytes) -> Any:
                x=  np.frombuffer(data, np.uint8).astype(np.float32)
                return (x / 255.0 - 0.5) * 24.0
        _encodings['uint8'] = uint8
        # we must use the same temporary data directory here
        local = temp_dir
        remote = os.path.join(data_dir, "vae_mds")
        # we should use this new scaling factor
        # the variance of input now will be 1
        scaling_factor = 1 / 0.13025
        class DatasetAdaptor(StreamingDataset):
            def __init__(self,
                        *args, 
                        **kwargs
                        ) -> None:
                super().__init__(*args, **kwargs)
            def __getitem__(self, idx:int):
                obj = super().__getitem__(idx)
                x = obj['vae_output']
                y = obj['label']
                return x.reshape(4, 32, 32) / scaling_factor, int(y)
        # the raw data is a tensor with value in range [-12, 12]
        # this is difficult for the model to learn, so we firstly scale the tensor into [-1, 1]
        # then we rescale the learnt model into [-12, 12] before sampling
        dataset = DatasetAdaptor(
            local=local, 
            remote=remote, 
            split=None,
            shuffle=True,
            shuffle_algo="naive",
            num_canonical_nodes=1,
            batch_size = batch_size)
        classes = [str(i) for i in range(1000)]
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(rank)
        vae_preprocessor = lambda x : x * scaling_factor
        vae = vae.eval()
        vae_postprocessor = VaeImageProcessor(do_normalize=True)
    elif dataset_type == "mj":
        is_streaming_dataset = False
        image_shape = (4, 32, 32)
        is_latent = True  
        class_condition = False
        from torch import Tensor
        """
        Convert latent to unit variance
        """
        class LatentProcessor:
            def __init__(self):
                self.mean = 0.8965
                self.scale_factor = 0.13025
            @torch.no_grad
            def preprocess(self, latents : Tensor):
                latents = (latents - self.mean) * self.scale_factor
                return latents.float()
            @torch.no_grad
            def postprocess(self, latents : Tensor):
                assert isinstance(latents, Tensor)
                latents = self.mean + (latents / self.scale_factor)
                # the vae is in fp16 so...
                return latents.half()
            def test(self, latents):
                return (self.postprocess(self.preprocess(latents)) - latents).abs().mean()

        latent_processor = LatentProcessor()
        class DatasetAdaptor(torch.utils.data.Dataset):
            def __init__(self) -> None:
                self.latents = np.load(os.path.join(data_dir, "small_ldt/mj_latents.npy"), mmap_mode='r') 
                self.text_emb = np.load(os.path.join(data_dir, "small_ldt/mj_text_emb.npy"), mmap_mode='r') 
            def __len__(self):
                return len(self.latents)
            def __getitem__(self, idx:int):
                x = latent_processor.preprocess(torch.tensor(self.latents[idx], requires_grad=False))
                y = torch.tensor(self.text_emb[idx], requires_grad=False)
                return x, y
        class ImageProcessor:
            def postprocess(self, image, *args, **kwargs):
                return (image + 1) / 2
        dataset = DatasetAdaptor()
        # unconditional generation, we only use one class
        classes = [str(i) for i in range(1)]
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(rank)
        vae = vae.eval()
        # covert unit variance latent back to unscaled latent
        vae_preprocessor = latent_processor.postprocess
        # convert [-1, 1] image to [0, 1] scaled
        vae_postprocessor = ImageProcessor()
    if image_shape is None:
        image_shape = dataset[0][0].shape
    if classes is None:
        classes = dataset.classes
    return ImageDatasetInfo(dataset, is_streaming_dataset, image_shape, classes, is_latent, vae, vae_preprocessor, vae_postprocessor, class_condition)

def patchify(x, patch_size):
    N, C, H, W = x.shape
    x = x.reshape((N, C, H//patch_size, patch_size, W//patch_size, patch_size))
    x = x.permute((0, 2,4,3,5, 1))
    x = x.reshape((N, -1, patch_size * patch_size * C))
    return x

import math
def lmpdf(y):
    x1 = math.exp(-(math.log(y/(1-y))**2)/2)
    x2 = math.sqrt(2 * math.pi) * y * (1-y)
    return x1/x2

    
class MaskDiTTrainer(DefaultTrainer):
    def __init__(self, diff_config, other_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_type = "lm"
        self.max_timestep = 512
        self.sampling_timestep = 50
        self.max_class = 10
        self.other_config = other_config
        self.finetune_epoch = diff_config.finetune_epoch
        self.mask_ratio = diff_config.mask_ratio
        self.cfg = 1.0
        self.lam = 0.1
    def on_training_begin(self):
        super().on_training_begin()
        if self.rank == 0:
            wandb.config.update(self.other_config)
    def prepare_input(self, obj):
        # disable masking for finetuning
        # we train the cifar for 512 epoch, we finetune it for 100 epoch...
        if self.ep > self.finetune_epoch:
            self.mask_ratio = 0.0
        device = self.rank
        sample_type = self.sample_type
        max_timestep = self.max_timestep
        model = self.model
        mask_ratio = self.mask_ratio
        image, class_labels = obj
        image = image.to(device)
        class_labels = class_labels.to(device)
        noise = torch.randn(image.shape).to(device)
        assert image.dtype == noise.dtype
        if sample_type == "uniform":
            timestep = torch.rand(size=(image.size(0),)).to(device) * max_timestep
        elif sample_type == "lm":
            x = torch.tensor([lmpdf((i + 1/2)/max_timestep) + 0.1 for i in range(max_timestep)]).to(device)
            x = x/x.sum()
            timestep = torch.multinomial(x, image.size(0)).to(torch.float32) # [0, max_timestep-1]
            timestep += torch.rand(size=(image.size(0),)).to(device) # [0, max_timestep)
        alpha = (timestep / max_timestep).view(-1, *([1]*(len(image.shape) - 1)))
        point = (1 - alpha) * noise + alpha * image
        patch_size = model.module.patch_size
        N, C, H, W = image.shape
        patch_num = (H//patch_size)*(W//patch_size)
        if mask_ratio == 0.0:
            mask = None
        else:
            z = torch.randn(N, patch_num, device=self.rank)
            _, indices = torch.sort(z,dim=-1)
            # identity : values[i] = z[indices[i]]
            # True == not masked
            mask = indices >= mask_ratio * patch_num
            # mask is of shape (N, L)

        # True == not mask, False == mask
        target = image - noise
        return {
            "point" : point,
            "timestep" : timestep,
            "class_labels" : class_labels,
            "mask" : mask,
            "target" : target
        }
    def calculate_loss(self, point, timestep, class_labels, mask, target):
        # prepare input here
        pred = self.model(point, timestep=timestep, class_labels = class_labels, mask = mask, apply_unpatchify=True)
        patch_size = self.model.module.patch_size
        pred = pred.sample
        if mask is None:
            return F.mse_loss(pred, target)
        # pred is of the same shape as target
        # pred is of shape
        pred = patchify(pred, patch_size)
        point = patchify(point, patch_size)
        target = patchify(target, patch_size)
        loss = F.mse_loss(mask_gather_2d(pred, mask), mask_gather_2d(target, mask))
        # reconstruction of masked noise
        n_mask = torch.logical_not(mask)
        loss += self.lam * F.mse_loss(mask_gather_2d(pred, n_mask), mask_gather_2d(point, n_mask))
        return loss
    @torch.no_grad
    def sample_image(self, class_labels):
        datasetinfo = self.datasetinfo
        image_shape = datasetinfo.image_shape
        max_timestep = self.max_timestep
        sampling_timestep = self.sampling_timestep
        
        device = self.rank
        use_classlabel = True
        model = self.model
        batch_size = class_labels.size(0)
        # we add a generator to better monitor the quality here
        X0 = torch.randn((batch_size, *image_shape)).to(device)
        null_conds = None
        if self.cfg > 1.0:
            null_conds = model.y_embedder.embedding_table.weight[-1].unsqueeze(0).expand((X0.shape[0], -1))
        for i in range(sampling_timestep):
            timestep = torch.full(size=(batch_size,), fill_value=i).to(device)
            timestep = timestep * (max_timestep / sampling_timestep)
            if use_classlabel:
                vc = model(X0, timestep=timestep, class_labels = class_labels, mask=None, apply_unpatchify=True)
            else:
                vc = model(X0, timestep=timestep, mask=None, apply_unpatchify=True)
            vc = vc.sample
            if null_conds is not None:
                vu = model(X0, timestep=timestep, class_labels=null_conds, mask=None, apply_unpatchify=True).sample
                vc = vu + cfg_sceduler(i, sampling_timestep, gain=cfg) * (vc - vu)
            X0 += vc * (1/sampling_timestep)
        if datasetinfo.is_latent:
            X0 = datasetinfo.vae_preprocessor(X0)
            X0 = datasetinfo.vae.decode(X0).sample
        return X0
    def eval_model(self, i, is_debug : bool = False):
        model = self.model
        world_size = self.world_size
        rank = self.rank
        max_class = self.max_class
        datasetinfo = self.datasetinfo
        
        def normalize_image(X):
            X = (X + 1)/2
            return X
        model.eval()
        assert world_size < 64 and 64 % world_size == 0
        total_batch_size = 64
        if datasetinfo.class_condition:
            class_labels = torch.arange(total_batch_size, device=rank) % max_class      
        else:
            class_labels = []
            for i in range(total_batch_size):
                class_labels.append(datasetinfo.dataset[i][1])
            class_labels = torch.stack(class_labels, dim=0).to(device)
        step = total_batch_size // world_size
        class_labels = class_labels[(rank*step):((rank+1)*step)]
        image_array = self.sample_image(class_labels)
        image_arrays = [torch.zeros_like(image_array, device=rank) for _ in range(world_size)]
        dist.all_gather(image_arrays, image_array)
        if rank == 0:
            image_array = torch.cat(image_arrays, dim = 0)
            if datasetinfo.is_latent:
                image_array = datasetinfo.vae_postprocessor.postprocess(image = image_array, output_type='pt')
            else:
                image_array = normalize_image(image_array)
            image_array = torchvision.utils.make_grid(image_array).permute((1,2,0))
            image_array = torch.clip(image_array, min = 0, max = 1).detach().cpu().numpy()
            if not is_debug:
                # we need to clip the value otherwise the generation will sometimes output extremely large or small value...
                images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")
                wandb.log({"examples": images}, commit=False) 
            return image_array
from torchvision import datasets, transforms  

class Factory:
    def __init__(self, temp_dir, file_config, model_config, diff_config, world_size):
        self.file_config = file_config
        self.diff_config = diff_config
        self.model_config = model_config
        self.world_size = world_size
        import tempfile
        self.temp_dir = temp_dir
    def make_model(self):
        return DiTModelWrapper(**self.model_config)
    @staticmethod
    def make_optimizer(model):
        return torch.optim.AdamW(model.parameters(), lr=1e-4)
    def make_dataloader(self, rank):     
        streaming.base.util.clean_stale_shared_memory()
        per_device_batch_size = self.file_config.get("batch_size", 256) // self.world_size
        datasetinfo = get_dataset(rank, self.diff_config.dataset_type, "../data", self.temp_dir, per_device_batch_size)
        dataset = datasetinfo.dataset
        if datasetinfo.is_streaming_dataset:
            sampler = None
            num_workers = 0
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank, seed=0, drop_last=True)
            num_workers = 2
        return datasetinfo, torch.utils.data.DataLoader(dataset, batch_size=per_device_batch_size, sampler=sampler, drop_last=True, num_workers=num_workers), sampler
    @staticmethod
    def make_scheduler(optimizer):
        return torch.optim.lr_scheduler.ConstantLR(optimizer) 
@dataclass
class DiffConfig:
    dataset_type : str    
    mask_ratio : float
    finetune_epoch : int
from maskDiT import *
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--ddp', type=int)
    parser.add_argument('--no_wandb', action="store_true")
    parser.add_argument('--profile', action="store_true")
    parser.add_argument('--compile', action="store_true")
    args = parser.parse_args()
    import json
    with open(args.config) as f:
        file_config = json.load(f)
    model_config = file_config["model"]
    
    dataset_type = file_config["dataset_type"]
    config = TrainerConfig(
        world_size=int(args.ddp),
        epoch=file_config["epochs"],
        gradient_accum_step=file_config.get("gradient_accum_step", 1),
        dtype=torch.bfloat16,
        clip_norm=file_config["clip_norm"],
        eval_n_sample=file_config["eval_n_sample"],
        save_n_sample=file_config["save_n_sample"],
        compile_model=args.compile,
        enable_wandb=not args.no_wandb,
        project_name="test-traininer",
        run_name=dataset_type,
        profiler=None,
        enable_profile=args.profile)
    import tempfile
    diff_config = DiffConfig(dataset_type=dataset_type, 
                             mask_ratio=file_config["mask_ratio"],
                             finetune_epoch=file_config["finetune_epoch"]) 
    with tempfile.TemporaryDirectory() as tempdir:
        factory = Factory(tempdir, file_config, model_config, diff_config, world_size=args.ddp)
        trainer = MaskDiTTrainer(
            diff_config = diff_config,
            other_config = {
                "args" : args,
                "diff_config" : diff_config,
                "file_config" : file_config},
            model = factory.make_model,
            optimizer= Factory.make_optimizer,
            scheduler= Factory.make_scheduler,
            dataloader_sampler = factory.make_dataloader,
            config=config)
        trainer.run()