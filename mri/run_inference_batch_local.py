"""
Run MRI inference data in the batch mode
"""
import argparse
import torch
import os
import copy
import pickle
import sys

import numpy as np
import nibabel as nib

from time import time
from colorama import Fore
from PIL import Image
from tqdm import tqdm
from pathlib import Path

Project_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(1, str(Project_DIR))

from utils import *

# -------------------------------------------------------------------------------------------------
# setup for testing from cmd

def arg_parser():
    """
    @args:
        - No args
    @rets:
        - config (Namespace): runtime namespace for setup
    """
    parser = argparse.ArgumentParser("Argument parser for STCNNT MRI test evaluation")

    parser.add_argument("--input_dir", default=None, help="folder to load the batch data, go to all subfolders")
    parser.add_argument("--input_fname", type=str, default="im", help='input npy prefix')
    parser.add_argument("--output_dir", default=None, help="folder to save the data; subfolders are created for each case")
    parser.add_argument("--power_norm", type=float, default=1600.0, help="normalize input signal power")
    parser.add_argument("--im_scaling", type=float, default=1.0, help="extra scaling applied to image")
    parser.add_argument("--gmap_scaling", type=float, default=1.0, help="extra scaling applied to gmap")
    parser.add_argument("--saved_model_path", type=str, default=None, help='model path. endswith ".pt" or ".pts" or ".pth"')
    parser.add_argument("--pad_time", action="store_true", help="with to pad along time")
    parser.add_argument("--patch_size_inference", type=int, default=-1, help='patch size for inference; if <=0, use the config setup')
    parser.add_argument("--save_gif", type=int, default=0, help="save input and output as gifs")
    parser.add_argument("--save_nii", type=int, default=0, help="save input and output as niis")
    parser.add_argument("--save_input_npy", action="store_true", help="with to save input npy files")
    parser.add_argument("--save_input_imgs", action="store_true", help="with to save input gif and/or nii files")
    parser.add_argument("--separate_complex", action="store_true", help="save real and imag part separatly")
    
    return parser.parse_args()

def check_args(args):
    """
    checks the cmd args to make sure they are correct
    @args:
        - args (Namespace): runtime namespace for setup
    @rets:
        - args (Namespace): the checked and updated argparse for MRI
    """
    assert args.saved_model_path.endswith(".pt") or args.saved_model_path.endswith(".pts"),\
            f"Saved model should either be \"*.pt\" or \"*.pts\""

    # get the args path
    fname = os.path.splitext(args.saved_model_path)[0]
    args.saved_model_config  = fname + '.config'

    return args

# -------------------------------------------------------------------------------------------------
# load model from pts, pt, pth

def load_model(args):
    """
    load a ".pt" or ".pts" model
    @args:
        - args (Namespace): runtime namespace for setup
    @rets:
        - model (torch model): the model ready for inference
    """
    
    config = []
    config_file = args.saved_model_config
    
    if os.path.isfile(config_file):
        print(f"{Fore.YELLOW}Load in config file - {config_file}")
        with open(config_file, 'rb') as f:
            config = pickle.load(f)

    if args.saved_model_path.endswith(".pt"):
        '''
        status = torch.load(args.saved_model_path, map_location=get_device())
        config = status['config']
        if not torch.cuda.is_available():
            config.device = torch.device('cpu')
        model = STCNNT_MRI(config=config)
        model.load_state_dict(status['model'])
        '''
        raise NotImplementedError("currently not taking in .pt")
    elif args.saved_model_path.endswith(".pts"):
        model = torch.jit.load(args.saved_model_path, map_location=get_device())
    else:
        raise NotImplementedError("currently not taking in .onnx")
    return model, config


def load_model_pth(args):
    '''
    config = []
    status = torch.load(args.saved_model_path, map_location=get_device())
    config = status['config']   
    if not torch.cuda.is_available():
        config.device = torch.device('cpu')
    model = STCNNT_MRI(config=config)
    model.load_state_dict(status['model_state'])
    return model, config
    '''
    raise NotImplementedError("currently not taking in .pth")


# -------------------------------------------------------------------------------------------------
# apply model, main inference function

def apply_model(data, model, gmap, config):
    '''
    Input 
        data : [RO E1 N SLC], remove any extra scaling
        gmap : [RO E1 N SLC], no scaling added
    '''

    t0 = time()
    device = get_device()

    if(data.ndim==2):
        data = data[:,:,np.newaxis,np.newaxis]
    if(data.ndim<4):
        data = np.expand_dims(data, axis=3)
    RO, E1, SLC, PHS = data.shape

    print(f"---> apply_model, preparation took {time()-t0} seconds ")
    print(f"---> apply_model, complex_i {config.complex_i}")
    print(f"---> apply_model, gmap array {not config.train_without_gmap if config.train_without_gmap else gmap.shape}")
    print(f"---> apply_model, input array {data.shape}")
    print(f"---> apply_model, pad_time {config.pad_time}")
    print(f"---> apply_model, height and width {config.height, config.width}")
    
    c = config
    
    try:
        for k in range(PHS):
            
            imgslab = data[:,:,:,k]
            H, W, SLC = imgslab.shape
            x = np.transpose(imgslab, [2, 0, 1]).reshape([1, SLC, 1, H, W])
            print(f"---> load data, input {x.shape} for volume")
            
            # trained_with_gmap
            if config.train_without_gmap:
                if config.complex_i:
                    input = np.concatenate((x.real, x.imag), axis=2)
                else:
                    input = np.abs(x)
            # trained_without_gmap
            else:
                gmapslab = gmap[:,:,:,k]
                g = np.transpose(gmapslab, [2, 0, 1]).reshape([1, SLC, 1, H, W])
                if config.complex_i:
                    input = np.concatenate((x.real, x.imag, g), axis=2)
                else:
                    input = np.concatenate((np.abs(x), g), axis=2)
            
            if not c.pad_time:
                cutout = (SLC, c.height[-1], c.width[-1])
                overlap = (0, c.height[-1]//2, c.width[-1]//2)
            else:
                cutout = (c.time, c.height[-1], c.width[-1])
                overlap = (c.time//2, c.height[-1]//4, c.width[-1]//4)
                
            print(f"---> running_inference, input {x.shape} for volume")

            try:
                _, output = running_inference(model, input, cutout=cutout, overlap=overlap, batch_size=4, device=device)
            except Exception as e:
                print(e)
                print(f"{Fore.YELLOW}---> call inference on cpu ...")
                _, output = running_inference(model, input, cutout=cutout, overlap=overlap, device=torch.device('cpu')) 
                
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()

            output = np.transpose(output, (3, 4, 2, 1, 0))
            
            
            if config.complex_i:
                data_filtered = output[:,:,0,:,0] + 1j*output[:,:,1,:,0]
            else:
                data_filtered = output

            data_filtered = np.reshape(data_filtered, (H, W, SLC))

    except Exception as e:
        print(e, "inference failed")
        data_filtered = np.reshape(copy.deepcopy(data), (H, W, SLC))

    t1 = time()
    print(f"---> apply_model_3D took {t1-t0} seconds ")

    return data_filtered


# -------------------------------------------------------------------------------------------------
# the main function for setup, eval call and saving results
def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

def main():
    # load model
    args = arg_parser()
    if args.saved_model_path.endswith(".pth"):
        model, config = load_model_pth(args)
        print(f"{Fore.YELLOW}Load in model file - {args.saved_model_path}")
    else:
        args = check_args(arg_parser())
        print(args)
        print(f"{Fore.YELLOW}Load in model file - {args.saved_model_path}")
        model, config = load_model(args)
    
    # load configs
    config.pad_time = args.pad_time
    config.ddp = False
    print("---> train_without_gmap:", config.train_without_gmap)
    print("---> train_with_complex:", config.complex_i)
    patch_size_inference = args.patch_size_inference
    if patch_size_inference > 0:
        config.height[-1] = patch_size_inference
        config.width[-1] = patch_size_inference
    
    # -------------------------------------------------------------------------------------------------
    # load the cases
    case_dirs = fast_scandir(args.input_dir)
    case_dirs = sorted(case_dirs)
    os.makedirs(args.output_dir, exist_ok=True)
    
    selected_cases = []
    images = []
    gmaps = []
    power_adjust_factors = []
    
    with tqdm(total=len(case_dirs), bar_format=get_bar_format()) as pbar:
        for c in case_dirs:
            fname = os.path.join(c, f"{args.input_fname}_real.npy")
            if os.path.isfile(fname):    
                image = np.load(os.path.join(c, f"{args.input_fname}_real.npy")) + np.load(os.path.join(c, f"{args.input_fname}_imag.npy")) * 1j
                image /= args.im_scaling
                
                # adjust power
                if args.power_norm:
                    power = np.mean(np.abs(image) ** 2)
                    power_adjust_factor = np.sqrt(power / args.power_norm)
                    image /= power_adjust_factor
                    print("---> adjust power, power=", power)
                    print("---> adjust power, factor=", power_adjust_factor, 1/power_adjust_factor)
                else:
                    power_adjust_factor = 1.0

                if len(image.shape) == 2:
                    image = image[:,:,np.newaxis,np.newaxis]
                elif len(image.shape) == 3:
                    image = image[:,:,:,np.newaxis]

                if(image.shape[3]>20):
                    print("---> transpose, image dim4 > 20")
                    image = np.transpose(image, (0, 1, 3, 2))

                RO, E1, slices, frames = image.shape
                print(f"{c}, images - {image.shape}")

                # read gfactor, generate fake gfactor input if the model doesn't take one.
                if not os.path.isfile(f"{c}/gfactor.npy"):
                    gmap = np.ones_like(image.real)
                else:
                    gmap = np.load(f"{c}/gfactor.npy")
                gmap /= args.gmap_scaling

                if(gmap.ndim==2):
                    gmap = np.expand_dims(gmap, axis=2)
                if(gmap.ndim==3):
                    gmap = gmap[:,:,:,np.newaxis]
                if gmap.shape[2] != slices:
                    print("---> skip, gmap and image not match")
                    continue
                else:
                    images.append(image)
                    gmaps.append(gmap)
                    power_adjust_factors.append(power_adjust_factor)
                    selected_cases.append(c)
            
            pbar.update(1)
                
    print(f"Loaded {len(images)} samples ... ")
    
    # -------------------------------------------------------------------------------------------------
    # run inference
    for ind in range(len(images)):
        case_dir = selected_cases[ind]
        print(f"-----------> Process {selected_cases[ind]} <-----------")
        
        image = images[ind]
        gmap = gmaps[ind]
        power_adjust_factor = power_adjust_factors[ind]
        output = apply_model(image.astype(np.complex64), model, gmap.astype(np.float32), config=config)
        
        # retrieve adjusted power
        if args.power_norm:
            output *= power_adjust_factor
            image *= power_adjust_factor
               
        case = os.path.basename(case_dir)                
        output_dir = os.path.join(args.output_dir, case)        
        os.makedirs(output_dir, exist_ok=True)
        
        if image.shape[3] == 1:
            image = np.squeeze(image)

        # -------------------------------------------------------------------------------------------------
        # save input
        if args.save_input_npy:
            if args.separate_complex and config.complex_i:
                res_name = os.path.join(output_dir, 'input_real.npy')
                print(res_name)
                np.save(res_name, image.real)
                res_name = os.path.join(output_dir, 'input_imag.npy')
                print(res_name)
                np.save(res_name, image.imag)
                res_name = os.path.join(output_dir, 'input.npy')
                print(res_name)
                np.save(res_name, image)
            else:
                res_name = os.path.join(output_dir, 'input.npy')
                print(res_name)
                np.save(res_name, image)

        if args.save_input_imgs:    
            # save nii
            if ind < args.save_nii:
                nib.save(nib.Nifti1Image(np.rot90(np.abs(image), k=1, axes=(1,0)), affine=np.eye(4)), os.path.join(output_dir, 'input_' + str(ind+1) + '.nii'))

            # save gifs
            if ind < args.save_gif:
                input_imgs = np.transpose(np.abs(image), (2,0,1))[:,::-1,:]
                input_imgs_rescaled = ((input_imgs - np.min(input_imgs)) / (np.max(input_imgs) - np.min(input_imgs)) * 255).astype(np.uint8)
                _imgs = [Image.fromarray(_img).convert('L') for _img in input_imgs_rescaled]
                _imgs[0].save(os.path.join(output_dir, 'input_' + str(ind+1) + '.gif'), save_all=True, append_images=_imgs[1:], duration=100, loop=0)

        # save output
        if args.separate_complex and config.complex_i:
            res_name = os.path.join(output_dir, 'output_real.npy')
            print(res_name)
            np.save(res_name, output.real)
        
            res_name = os.path.join(output_dir, 'output_imag.npy')
            print(res_name)
            np.save(res_name, output.imag)

            res_name = os.path.join(output_dir, 'output.npy')
            print(res_name)
            np.save(res_name, output)
            
        else:
            res_name = os.path.join(output_dir, 'output.npy')
            print(res_name)
            np.save(res_name, output)
            
        # save nii
        if ind < args.save_nii:
            nib.save(nib.Nifti1Image(np.abs(np.rot90(np.abs(output), k=1, axes=(1,0))), affine=np.eye(4)), os.path.join(output_dir, 'output_' + str(ind+1) + '.nii'))
        
        # save gifs
        if ind < args.save_gif:
            output_imgs = np.transpose(np.abs(output), (2,0,1))[:,::-1,:]
            output_imgs_rescaled = ((output_imgs - np.min(output_imgs)) / (np.max(output_imgs) - np.min(output_imgs)) * 255).astype(np.uint8)
            imgs = [Image.fromarray(img).convert('L') for img in output_imgs_rescaled]
            imgs[0].save(os.path.join(output_dir, 'output_' + str(ind+1) + '.gif'), save_all=True, append_images=imgs[1:], duration=100, loop=0)
        
        print("--" * 30)
        
if __name__=="__main__":
    main()

    