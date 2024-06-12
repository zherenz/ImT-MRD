"""
Inference given a model and complete image
Cuts the image into overlapping patches of given
patch size and overlap size
"""
import torch
import numpy as np
from tqdm import tqdm
from skimage.util.shape import view_as_windows


def running_inference(model, image, cutout=(16,256,256), overlap=(4,64,64), batch_size=4, device=torch.device('cpu')):
    """
    Runs inference by breaking image into overlapping patches
    Runs the patches through the model and then stiches them back
    @args:
        - model (torch or onnx model): the model to run inference with
        - image (numpy.ndarray or torch.Tensor): the image to run inference on
            - requires the image to have ndim==3 or ndim==4 or ndim==5
                [T,H,W] or [T,C,H,W] or [B,T,C,H,W]
                ndim==5 requires 0th dim size to be 1 (B==1)
        - cutout (int 3-tuple): the patch shape for each cutout [T,H,W]
        - overlap (int 3-tuple): the number of pixels to overlap [T,H,W]
            - required to be smaller than cutout
        - batch_size (int): number of patches per model call
        - device (torch.device): the device to run inference on
    @rets:
        - image_fin (4D numpy.ndarray): result as numpy array [T,C,H,W]
            if input image.ndim==3 then C=1, otw same as input
        - image_fin (5D torch.Tensor or numpy array): result as [B,T,C,H,W]
            always B=1. If input image.ndim==3 then C=1, otw same as input
    """
    # ---------------------------------------
    # setup the model and image
    is_torch_model = isinstance(model, torch.nn.Module)
    is_script_model = isinstance(model, torch.jit._script.RecursiveScriptModule)
    
    if device == torch.device('cpu'):
        batch_size = 32
    # Compute capability 8.0 or higher supports bfloat16
    else:
        cur_device = torch.cuda.current_device()
        enable_bfloat16 = torch.cuda.get_device_properties(cur_device).major >= 8
        print("---> enable_bfloat16: ", enable_bfloat16)
    
    if is_torch_model or is_script_model:
        if is_script_model:
            model.cuda()
        else:
            model = model.to(device)
        model.eval()

    try:
        image = image.cpu().detach().numpy()
    except:
        image = image

    if image.ndim == 5:
        assert image.shape[0]==1
        image = image[0]
    elif image.ndim == 4:
        pass
    elif image.ndim == 3:
        image = image[:,:,np.newaxis]
    else:
        raise NotImplementedError(f"Image dimensions not yet implemented: {image.ndim}")
    
    assert (cutout > overlap), f"cutout should be greater than overlap"
    # ---------------------------------------------------------------------------------------------
    # some constants, used several times
    d_type = image.dtype

    TO, CO, HO, WO = image.shape        # original
    Tc, Hc, Wc = cutout                 # cutout
    To, Ho, Wo = overlap                # overlap
    Ts, Hs, Ws = Tc-To, Hc-Ho, Wc-Wo    # sliding window shape
    # ---------------------------------------------------------------------------------------------
    # padding the image so we have a complete coverup
    # in each dim we pad the left side by overlap
    # and then cover the right side by what remains from the sliding window
    image_pad = np.pad(image, (
                        (To, -TO%Ts),
                        (0,0),
                        (Ho, -HO%Hs),
                        (Wo, -WO%Ws)),
                        "symmetric")
    # ---------------------------------------------------------------------------------------------
    # breaking the image down into patches
    # and remembering the length in each dimension
    image_patches = view_as_windows(image_pad, (Tc,CO,Hc,Wc), (Ts, 1, Hs, Ws))
    Ntme, _, Nrow, Ncol, _, _, _, _ = image_patches.shape

    image_batch = image_patches.reshape(-1,Tc,CO,Hc,Wc) # shape:(num_patches,T,C,H,W)
    #print(f"norm = {np.linalg.norm(image_batch)}")
    
    # ---------------------------------------------------------------------------------------------
    # inferring each patch in length of batch_size
    image_batch_pred = None

    if is_torch_model:
        with torch.inference_mode():
            for i in range(0, image_batch.shape[0], batch_size):
                x_in = torch.from_numpy(image_batch[i:i+batch_size]).to(device=device)
                
                if (not is_script_model) and enable_bfloat16:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        res = model(x_in).cpu().detach().numpy()
                else:
                    res = model(x_in).cpu().detach().numpy()
                    
                if image_batch_pred is None:
                    image_batch_pred = np.empty((image_batch.shape[0], Tc, res.shape[2], Hc, Wc), dtype=d_type)
                    
                image_batch_pred[i:i+batch_size] = res
    else:
        for i in range(0, image_batch.shape[0], batch_size):
            x_in = image_batch[i:i+batch_size]
            ort_inputs = {model.get_inputs()[0].name: x_in.astype('float32')}
            res = model.run(None, ort_inputs)[0]
            if image_batch_pred is None:
                    image_batch_pred = np.empty((image_batch.shape[0], Tc, res.shape[2], Hc, Wc), dtype=d_type)
                    
            image_batch_pred[i:i+batch_size] = res
            
    C_out = image_batch_pred.shape[2]
    image_patches_ot_shape = (*image_patches.shape[:-3],C_out,*image_patches.shape[-2:])
    image_pad_ot_shape = (image_pad.shape[0], C_out, *image_pad.shape[2:])
    
    # ---------------------------------------------------------------------------------------------
    # setting up the weight matrix
    # matrix_weight defines how much a patch contributes to a pixel
    # image_wgt is the sum of all weights. easier calculation for result
    matrix_weight = np.ones((cutout), dtype=d_type)

    for t in range(To):
        matrix_weight[t] *= ((t+1)/To)
        matrix_weight[-t-1] *= ((t+1)/To)

    for h in range(Ho):
        matrix_weight[:,h] *= ((h+1)/Ho)
        matrix_weight[:,-h-1] *= ((h+1)/Ho)

    for w in range(Wo):
        matrix_weight[:,:,w] *= ((w+1)/Wo)
        matrix_weight[:,:,-w-1] *= ((w+1)/Wo)

    image_wgt = np.zeros(image_pad_ot_shape, dtype=d_type) # filled in the loop below
    matrix_weight = np.repeat(matrix_weight[:,np.newaxis], C_out, axis=1)
    matrix_rep = np.repeat(matrix_weight[np.newaxis], Ntme*Nrow*Ncol, axis=0)
    matrix_rep = matrix_rep.reshape(image_patches_ot_shape)
    # ---------------------------------------------------------------------------------------------
    # Putting the patches back together
    image_batch_pred = image_batch_pred.reshape(image_patches_ot_shape)
    image_prd = np.zeros(image_pad_ot_shape, dtype=d_type)

    for nt in range(Ntme):
        for nr in range(Nrow):
            for nc in range(Ncol):
                image_wgt[Ts*nt:Ts*nt+Tc, :, Hs*nr:Hs*nr+Hc, Ws*nc:Ws*nc+Wc] += matrix_rep[nt, 0, nr, nc]
                image_prd[Ts*nt:Ts*nt+Tc, :, Hs*nr:Hs*nr+Hc, Ws*nc:Ws*nc+Wc] += matrix_weight * image_batch_pred[nt, 0, nr, nc]

    image_prd /= image_wgt
    # ---------------------------------------------------------------------------------------------
    # remove the extra padding
    image_fin = image_prd[To:To+TO, :, Ho:Ho+HO, Wo:Wo+WO]

    # return a 4D numpy and 5D torch.tensor for easier followups
    if is_torch_model:
        res = image_fin, torch.from_numpy(image_fin[np.newaxis]).to(device)
    else:
        res = image_fin, image_fin[np.newaxis]

    return res
