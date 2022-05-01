import torch
import random
import numpy as np
from hydra.utils import instantiate
import albumentations as A
from typing import Optional, NoReturn
import os


def set_global_seed(seed: int) -> NoReturn:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_gpus_ids(num_gpus: int):
    return range(torch.cuda.device_count() if num_gpus is None else num_gpus)

def data_loader():
    pass


def optimizer(model, optimizer, optim_params):
    parameters = {"params": model.parameters()}
    parameters
    optim = instantiate(optimizer, )


class Augmentation:
    def __init__(self, augmentations: Optional[str] = None):
        self.augmentations = A.load(augmentations)

    def __call__(self, **kwargs):
        transformed = self.augmentations(**kwargs)
        return transformed


def worker_init_fn(worker_id: int, initial_seed: int = 3407) -> NoReturn:
    """Fixes bug with identical augmentations.
    More info: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    """
    seed = initial_seed**2 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
