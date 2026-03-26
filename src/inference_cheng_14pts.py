import numpy as np
import bisect

import torch
from torch.utils.data import DataLoader

from utils.masks import lambda_percentage,generate_mask_from_unstructured
from utils.dataset import TestClicDataset
from evaluate import make_plot
from custom_comp.zoo import models,model_loading,Cheng2020Attention,rewire_g_a_s

import wandb

def mask_to_device(mask, device="cuda"):
    for _, mask_list in mask.items():
        for i in range(len(mask_list)):
            for k, v in mask_list[i].items():
                mask_list[i][k] = v.to(device)

def remove_ga_gs_aliases(state_dict):
    return {
        k: v for k, v in state_dict.items()
        if not k.startswith("g_a.") and not k.startswith("g_s.")
    }

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

    # Load model and mask
    checkpoint_path = "/home/ids/flauron-23/MagV/data/magv_40_cheng_14_pts/models/magv_40_cheng_14_pts_checkpoint.pth.tar"    
    mask_path ="/home/ids/flauron-23/MagV/data/magv_40_cheng_14_pts/masks/mask_magv_40_cheng_14_pts.pth"
    
    net = model_loading(model_class = Cheng2020Attention,state_dict_path=checkpoint_path,preprocess=None,postprocess=None)[1]
    oldMask = torch.load(mask_path, map_location='cpu')

    # parameters_to_prune 
    parameters_to_prune = {}  
    parameters_to_prune["g_a"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], net.g_a.modules())]
    parameters_to_prune["g_s"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], net.g_s.modules())]
    
    mask_to_device(oldMask, device="cuda")
    net = net.to("cuda")

    #refnet = models["stf"]().to("cuda")

    if log_wandb:
        wandb.init(
            project='training',
            entity='MagV',
            name='inference_cheng_14pts',
        )

    # make_plot("cheng",net,oldMask,None,parameters_to_prune,"cuda",dataloader,"inference_cheng_14pts",log_wandb=log_wandb,max_images=30,anchor = True)
    
    make_plot("cheng",net,oldMask,None,parameters_to_prune,"cuda",dataloader,"inference_cheng_14pts",log_wandb=log_wandb,max_images=30,anchor = True)
    
    print("succeess")