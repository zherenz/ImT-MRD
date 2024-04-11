"""
Utility functions to set up the training.
"""
import os
import torch
import logging
import numpy as np
from collections import OrderedDict
from datetime import datetime
from torchinfo import summary
import torch.distributed as dist

# -------------------------------------------------------------------------------------------------
# setup logger

def setup_logger(config):
    """
    logger setup to be called from any process
    """
    os.makedirs(config.log_path, exist_ok=True)
    log_file_name = os.path.join(config.log_path, f"{config.run_name}_{config.date}.log")
    level = logging.INFO
    format = "%(asctime)s [%(levelname)s] %(message)s"
    file_handler = logging.FileHandler(log_file_name, 'a', 'utf-8')
    file_handler.setFormatter(logging.Formatter(format))
    stream_handler = logging.StreamHandler()

    logging.basicConfig(level=level, format=format, handlers=[file_handler,stream_handler])

    file_only_logger = logging.getLogger("file_only") # separate logger for files only
    file_only_logger.addHandler(file_handler)
    file_only_logger.setLevel(logging.INFO)
    file_only_logger.propagate=False

# -------------------------------------------------------------------------------------------------

def get_bar_format():
    """Get the default bar format
    """
    return '{desc}{percentage:3.0f}%|{bar:10}{r_bar}'

# -------------------------------------------------------------------------------------------------
# setup the run

def setup_run(config, dirs=["log_path", "results_path", "model_path", "check_path"]):
    """
    sets up datetime, logging, seed and ddp
    @args:
        - config (Namespace): runtime namespace for setup
        - dirs (str list): the directories from config to be created
    """
    # get current date
    now = datetime.now()
    now = now.strftime("%H-%M-%S-%Y%m%d") # make sure in ddp, different nodes have the save file name
    config.date = now

    # setup logging
    setup_logger(config)

    # create relevant directories
    try:
        config_dict = dict(config)
    except TypeError:
        config_dict = vars(config)
    for dir in dirs:
        os.makedirs(config_dict[dir], exist_ok=True)
        logging.info(f"Run:{config.run_name}, {dir} is {config_dict[dir]}")
    
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # setup dp/ddp
    if not dist.is_initialized():
        config.device = get_device(config.device)
        world_size = torch.cuda.device_count()
        
        if config.ddp:
            if config.device == torch.device('cpu') or world_size <= 1:
                config.ddp = False
            
        config.world_size = world_size if config.ddp else -1
    else:
        world_size = int(os.environ["WORLD_SIZE"])
        config.world_size = world_size
        
    logging.info(f"Training on {config.device} with ddp set to {config.ddp}")
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
       
    # pytorch loader fix
    if config.num_workers==0: config.prefetch_factor = 2

# -------------------------------------------------------------------------------------------------
def compute_total_steps(config, num_samples):
    if config.ddp: 
        num_samples /= dist.get_world_size()

    total_steps = int(np.ceil(num_samples/(config.batch_size*config.iters_to_accumulate))*config.num_epochs)
    
    return total_steps

# -------------------------------------------------------------------------------------------------
# # wrapper around getting device

def get_device(device=None):
    """
    @args:
        - device (torch.device): if not None this device will be returned
            otherwise check if cuda is available
    @rets:
        - device (torch.device): the device to be used
    """

    return device if device is not None else \
            "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------------------------------------
         
def clean_after_training():
    """Clean after the training
    """
    os.system("kill -9 $(ps aux | grep torchrun | grep -v grep | awk '{print $2}') ")
    os.system("kill -9 $(ps aux | grep wandb | grep -v grep | awk '{print $2}') ")
    os.system("kill -9 $(ps aux | grep python3 | grep -v grep | awk '{print $2}') ")
    
# -------------------------------------------------------------------------------------------------

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
                        
# -------------------------------------------------------------------------------------------------    

def create_generic_class_str(obj : object, exclusion_list=[torch.nn.Module, OrderedDict]) -> str:
    """
    Create a generic name of a class
    @args:
        - obj (object): the class to make string of
        - exclusion_list (object list): the objects to exclude from the class string
    @rets:
        - class_str (str): the generic class string
    """
    name = type(obj).__name__

    vars_list = []
    for key, value in vars(obj).items():
        valid = True
        for type_e in exclusion_list:
            if isinstance(value, type_e) or key.startswith('_'):
                valid = False
                break
        
        if valid:
            vars_list.append(f'{key}={value!r}')
            
    vars_str = ',\n'.join(vars_list)
    return f'{name}({vars_str})'

# -------------------------------------------------------------------------------------------------

if __name__=="__main__":
    pass