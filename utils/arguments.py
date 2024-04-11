"""
Argument and parsers
"""

import argparse
from collections import OrderedDict
from datetime import datetime

# -------------------------------------------------------------------------------------------------
# from https://stackoverflow.com/questions/18668227/argparse-subcommands-with-nested-namespaces
class Nestedspace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group,name = name.split('.',1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value
            
def none_or_str(value):
    if value == 'None':
        return None
    return value

# -------------------------------------------------------------------------------------------------
# parser for commonly shared args (subject to change over time)
def add_shared_args(parser=argparse.ArgumentParser("Argument parser for transformer projects")):
    """
    Add shared arguments between trainers
    @args:
        parser (argparse, optional): parser object
    @rets:
        parser : new/modified parser
    """
    # common paths
    parser.add_argument("--log_path", type=str, default=None, help='directory for log files')
    parser.add_argument("--results_path", type=str, default=None, help='folder to save results in')
    parser.add_argument("--model_path", type=str, default=None, help='directory for saving the final model')
    parser.add_argument("--check_path", type=str, default=None, help='directory for saving checkpoints (model weights)')

    # wandb
    parser.add_argument("--project", type=str, default='STCNNT', help='project name')
    parser.add_argument("--run_name", type=str, default='cifar', help='current run name')
    parser.add_argument("--run_notes", type=str, default='cifar_train', help='notes for the current run')
    parser.add_argument("--wandb_entity", type=str, default="gadgetron", help='wandb entity to link with')
    parser.add_argument("--sweep_id", type=str, default="none", help='sweep id for hyper parameter searching')
    parser.add_argument("--sweep_count", type=int, default=50, help='number of sweep per agent to run')

    # dataset arguments
    parser.add_argument("--ratio", nargs='+', type=float, default=[100,100,0], help='Ratio (as a percentage) for train/val/test divide of given data. Does allow for using partial dataset')    

    # dataloader arguments
    parser.add_argument("--num_workers", type=int, default=8, help='number of workers for data loading')
    parser.add_argument("--prefetch_factor", type=int, default=8, help='number of batches loaded in advance by each worker')

    # trainer arguments
    parser.add_argument("--num_epochs", type=int, default=150, help='number of epochs to train for')
    parser.add_argument("--batch_size", type=int, default=128, help='size of each batch')
    parser.add_argument("--save_cycle", type=int, default=5, help='Number of epochs between saving model weights')
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help='gradient norm clip, if <=0, no clipping')
    parser.add_argument("--with_timer", action="store_true", help='whether to train with timing')
 
    # loss, optimizer, and scheduler arguments
    parser.add_argument("--optim", type=str, default="adamw", help='what optimizer to use, "adamw", "nadam", "sgd", "sophia"')
    parser.add_argument("--global_lr", type=float, default=5e-4, help='step size for the optimizer')
    parser.add_argument("--beta1", type=float, default=0.90, help='beta1 for the default optimizer')
    parser.add_argument("--beta2", type=float, default=0.95, help='beta2 for the default optimizer')

    parser.add_argument("--scheduler_type", type=str, default="ReduceLROnPlateau", help='"ReduceLROnPlateau", "StepLR", or "OneCycleLR"')
    parser.add_argument('--scheduler.ReduceLROnPlateau.patience', dest='scheduler.ReduceLROnPlateau.patience', type=int, default=2, help="number of epochs to wait for further lr adjustment")
    parser.add_argument('--scheduler.ReduceLROnPlateau.cooldown', dest='scheduler.ReduceLROnPlateau.cooldown', type=int, default=2, help="after adjusting the lr, number of epochs to wait before another adjustment")
    parser.add_argument('--scheduler.ReduceLROnPlateau.min_lr', dest='scheduler.ReduceLROnPlateau.min_lr', type=float, default=1e-7, help="minimal lr")
    parser.add_argument('--scheduler.ReduceLROnPlateau.factor', dest='scheduler.ReduceLROnPlateau.factor', type=float, default=0.9, help="lr reduction factor, multiplication")
        
    parser.add_argument('--scheduler.StepLR.step_size', dest='scheduler.StepLR.step_size', type=int, default=5, help="number of epochs to reduce lr")
    parser.add_argument('--scheduler.StepLR.gamma', dest='scheduler.StepLR.gamma', type=float, default=0.8, help="multiplicative factor of learning rate decay")
    parser.add_argument('--scheduler.OneCycleLR.pct_start', dest='scheduler.OneCycleLR.pct_start', type=float, default=0.3, help="range of taking off to reach max learning rate")
    
    parser.add_argument("--weight_decay", type=float, default=0.1, help='weight decay for regularization')
    parser.add_argument("--all_w_decay", action="store_true", help='option of having all params have weight decay. By default norms and embeddings do not')
    parser.add_argument("--use_amp", action="store_true", help='whether to train with mixed precision')
    parser.add_argument("--ddp", action="store_true", help='whether to train with ddp')
    
    parser.add_argument("--iters_to_accumulate", type=int, default=1, help='Number of iterations to accumulate gradients; if >1, gradient accumulation')

    # misc arguments
    parser.add_argument("--seed", type=int, default=3407, help='seed for randomization')
    parser.add_argument("--device", type=str, default=None, help='device to train on')
    parser.add_argument("--load_path", type=str, default=None, help='path to load model weights from')
    parser.add_argument("--debug", "-D", action="store_true", help='option to run in debug mode')
    parser.add_argument("--summary_depth", type=int, default=6, help='depth to print the model summary till')

    return parser

def add_shared_STCNNT_args(parser=argparse.ArgumentParser("Argument parser for STCNNT")):
    """
    Add shared arguments for all STCNNT models
    @args:
        parser (argparse, optional): parser object
    @rets:
        parser : new/modified parser
    """
        
    # base model arguments
    parser.add_argument("--cell_type", type=str, default="sequential", help='cell type, sequential or parallel')
    
    parser.add_argument("--C_in", type=int, default=3, help='number of channels in the input')
    parser.add_argument("--C_out", type=int, default=16, help='number of channels in the output')
    parser.add_argument("--time", type=int, default=12, help='training time series length')
    parser.add_argument("--height", nargs='+', type=int, default=[64, 128], help='heights of the training images')
    parser.add_argument("--width", nargs='+', type=int, default=[64, 128], help='widths of the training images')
    
    parser.add_argument("--block_dense_connection", type=int, default=1, help='whether to add dense connections between cells in a block')
        
    parser.add_argument("--a_type", type=str, default="conv", help='type of attention in the spatial attention modules')
    parser.add_argument("--mixer_type", type=str, default="conv", help='conv or lin, type of mixer in the spatial attention modules; only conv is possible for the temporal attention')
    
    parser.add_argument("--window_size", nargs='+', type=int, default=[64, 64], help='size of window for spatial attention. This is the number of pixels in a window. Given image height and weight H and W, number of windows is H/windows_size * W/windows_size')
    parser.add_argument("--patch_size", nargs='+', type=int, default=[16, 16], help='size of patch for spatial attention. This is the number of pixels in a patch. An image is first split into windows. Every window is further split into patches.')
    
    parser.add_argument("--window_sizing_method", type=str, default="mixed", help='method to adjust window_size between resolution levels, "keep_window_size", "keep_num_window", "mixed".\
                        "keep_window_size" means number of pixels in a window is kept after down/upsample the image; \
                        "keep_num_window" means the number of windows is kept after down/upsample the image; \
                        "mixed" means interleave both methods.')
    
    parser.add_argument("--n_head", type=int, default=8, help='number of transformer heads')
    parser.add_argument("--kernel_size", type=int, default=3, help='size of the square kernel for CNN')
    parser.add_argument("--stride", type=int, default=1, help='stride for CNN (equal x and y)')
    parser.add_argument("--padding", type=int, default=1, help='padding for CNN (equal x and y)')
    parser.add_argument("--stride_t", type=int, default=2, help='stride for temporal attention cnn (equal x and y)') 
    
    parser.add_argument("--mixer_kernel_size", type=int, default=5, help='conv kernel size for the mixer')
    parser.add_argument("--mixer_stride", type=int, default=1, help='stride for the mixer')
    parser.add_argument("--mixer_padding", type=int, default=2, help='padding for the mixer')
      
    parser.add_argument("--normalize_Q_K", action="store_true", help='whether to normalize Q and K before computing attention matrix')
    parser.add_argument("--cosine_att", type=int, default=0, help='whether to use cosine attention; if True, normalize_Q_K is ignored')   
    parser.add_argument("--att_with_relative_postion_bias", type=int, default=1, help='whether to use relative position bias')   
            
    parser.add_argument("--att_dropout_p", type=float, default=0.0, help='pdrop for the attention coefficient matrix')
    parser.add_argument("--dropout_p", type=float, default=0.1, help='pdrop regulization for stochastic residual connections')
    
    parser.add_argument("--att_with_output_proj", type=int, default=1, help='whether to add output projection in attention layer')
    parser.add_argument("--scale_ratio_in_mixer", type=float, default=4.0, help='the scaling ratio to increase/decrease dimensions in the mixer of an attention layer')
    
    parser.add_argument("--norm_mode", type=str, default="instance2d", help='normalization mode: "layer", "batch2d", "instance2d", "batch3d", "instance3d"')
    
    parser.add_argument("--shuffle_in_window", type=int, default=0, help='whether to shuffle patches in a window for the global attention')    
    
    parser.add_argument("--is_causal", action="store_true", help='treat timed data as causal and mask future entries')
    parser.add_argument("--interp_align_c", action="store_true", help='align corners while interpolating')
    
    parser = add_shared_args(parser)

    return parser

def add_backbone_STCNNT_args(parser=argparse.ArgumentParser("Argument parser for backbone models")):
    """
    Add backbone model specific parameters
    """
    
    parser.add_argument('--backbone', type=str, default="hrnet", help="which backbone model to use, 'hrnet', 'unet', 'LLM', 'small_unet' ")
    
    # hrnet
    parser.add_argument('--backbone_hrnet.C', dest='backbone_hrnet.C', type=int, default=16, help="number of channels in main body of hrnet")
    parser.add_argument('--backbone_hrnet.num_resolution_levels', dest='backbone_hrnet.num_resolution_levels', type=int, default=2, help="number of resolution levels; image size reduce by x2 for every level")
    parser.add_argument('--backbone_hrnet.block_str', dest='backbone_hrnet.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
        to define the attention layers in blocks; if multiple strings are given, each is for a resolution level.")    
    parser.add_argument('--backbone_hrnet.use_interpolation', dest='backbone_hrnet.use_interpolation', type=int, default=1, help="whether to use interpolation in downsample layer; if False, use stride convolution")
    
    # unet            
    parser.add_argument('--backbone_unet.C', dest='backbone_unet.C', type=int, default=16, help="number of channels in main body of unet")
    parser.add_argument('--backbone_unet.num_resolution_levels', dest='backbone_unet.num_resolution_levels', type=int, default=2, help="number of resolution levels for unet; image size reduce by x2 for every level")
    parser.add_argument('--backbone_unet.block_str', dest='backbone_unet.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
        to define the attention layers in blocks; if multiple strings are given, each is for a resolution level.")    
    parser.add_argument('--backbone_unet.use_unet_attention', dest='backbone_unet.use_unet_attention', type=int, default=1, help="whether to add unet attention between resolution levels")
    parser.add_argument('--backbone_unet.use_interpolation', dest='backbone_unet.use_interpolation', type=int, default=1, help="whether to use interpolation in downsample layer; if False, use stride convolution")
    parser.add_argument('--backbone_unet.with_conv', dest='backbone_unet.with_conv', type=int, default=1, help="whether to add conv in down/upsample layers; if False, only interpolation is performed")
    
    # LLMs
    parser.add_argument('--backbone_LLM.C', dest='backbone_LLM.C', type=int, default=16, help="number of channels in main body of LLM net")
    parser.add_argument('--backbone_LLM.num_stages', dest='backbone_LLM.num_stages', type=int, default=2, help="number of stages")
    parser.add_argument('--backbone_LLM.block_str', dest='backbone_LLM.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
        to define the attention layers in stages; if multiple strings are given, each is for a stage.")    
    parser.add_argument('--backbone_LLM.add_skip_connections', dest='backbone_LLM.add_skip_connections', type=int, default=1, help="whether to add skip connections between stages; if True, densenet type connections are added; if False, LLM type network is created.")
                     
    # small unet
    parser.add_argument("--backbone_small_unet.channels", dest='backbone_small_unet.channels', nargs='+', type=int, default=[16,32,64], help='number of channels in each layer')
    parser.add_argument('--backbone_small_unet.block_str', dest='backbone_small_unet.block_str', nargs='+', type=str, default=['T1L1G1'], help="block string \
        to define the attention layers in stages; if multiple strings are given, each is for a stage.")   
       
    parser = add_shared_STCNNT_args(parser=parser)
            
    return parser

