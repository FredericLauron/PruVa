import json

import numpy as np
import bisect

from utils.engine import AverageMeter
import torch
from torch.utils.data import DataLoader

from utils.masks import lambda_percentage,generate_mask_from_unstructured
from utils.dataset import TestClicDataset
from evaluate import make_plot
from custom_comp.zoo import models,model_loading,Cheng2020Attention,rewire_g_a_s
from utils import compute_metrics,crop, pad
from tqdm import tqdm
import wandb

def inference_with_scale(model,x, x_padded, padding,factor,s=2):
    
    out_enc = model.compress(x_padded, s, factor)
    # out_enc = model.compress(x_padded)
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], s, factor)

    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
    # out_dec["x_hat"] = F.pad(out_dec["x_hat"], padding) 

    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    rate = bpp*num_pixels

    del out_enc
    del out_dec

    return metrics, torch.tensor([bpp]), rate

if __name__ == "__main__":

    log_wandb = False

    dataset = TestClicDataset(data_dir = "/home/ids/flauron-23/clic")
    dataloader = DataLoader(
    dataset, 
    shuffle=False, 
    batch_size=1, 
    pin_memory=True, 
    num_workers= 30 
    )

    net = models["qvrf"]().eval().to("cuda")

    if log_wandb:
        wandb.init(
            project='training',
            entity='MagV',
            name='inference_qvrf_cheng',
        )

    # make_plot("cheng",net,oldMask,None,parameters_to_prune,"cuda",dataloader,"inference_cheng_14pts",log_wandb=log_wandb,max_images=30,anchor = True)
    
    #make_plot("qvrf",net,None,None,None,"cuda",dataloader,"inference_qvrf_cheng",log_wandb=log_wandb,max_images=30,anchor = True)
    
    psnr_res = {}
    mssim_res = {}
    bpp_res = {} 
    qvref_factor = [0.5,1.25,2.4,4.0,6.5,8.3]

    res = {}
    for qp in qvref_factor:
        res[qp] = {
                    "psnr": AverageMeter(),
                    "ms_ssim": AverageMeter(),
                    "bpps": AverageMeter(),
                    "rate": AverageMeter(),
                    "criterion": None,
                    "loss": AverageMeter(),
                    }

  
    
    with torch.no_grad():
        for j,x in enumerate(tqdm(dataloader)):

            if j>= 30:
                break
            x = x.to("cuda")
            x_padded, padding = pad(x, 128)

            for factor in qvref_factor:
                metrics, bpp, rate = inference_with_scale(net, x, x_padded, padding, factor, s=2) 
                
                res[factor]['psnr'].update(metrics["psnr"])
                res[factor]['ms_ssim'].update(metrics["ms-ssim"])
                res[factor]['bpps'].update(bpp.item())
                res[factor]['rate'].update(rate)   

   
    model_res = {}
    for qp in qvref_factor:
        model_res[qp] = {
            'psnr': res[qp]['psnr'].avg,
            'mssim': res[qp]['ms_ssim'].avg,
            'bpp': res[qp]['bpps'].avg,
            'rate': res[qp]['rate'].avg,
            'loss': res[qp]['loss'].avg
        }
            #print(f'{qp}: {model_res[qp]}')


    file_path = f"/home/ids/flauron-23/MagV/src/results/metrics_QVRF_clic2.json"
    with open(file_path, 'w') as outfile:
        json.dump(model_res, outfile)

    print("success")