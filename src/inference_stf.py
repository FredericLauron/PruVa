import numpy as np
import bisect

import torch
from torch.utils.data import DataLoader

from utils.masks import lambda_percentage,generate_mask_from_unstructured
from utils.dataset import TestKodakDataset
from evaluate import make_plot
from custom_comp.zoo import models,model_loading,SymmetricalTransFormer,rewire_g_a_s

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

    kodak_dataset = TestKodakDataset(data_dir = "/home/ids/flauron-23/kodak")
    kodak_dataloader = DataLoader(
    kodak_dataset, 
    shuffle=False, 
    batch_size=1, 
    pin_memory=True, 
    num_workers= 30 
    )

    # Load model and mask
    checkpoint_path = "/home/ids/flauron-23/MagV/data/magv_60_stf_unstruct_41_epochs_14_points/models/magv_60_stf_unstruct_41_epochs_14_points_checkpoint.pth.tar"    
    mask_path ="/home/ids/flauron-23/MagV/data/magv_60_stf_unstruct_41_epochs_14_points/masks/mask_magv_60_stf_unstruct_41_epochs_14_points.pth"
    
    net = model_loading(model_class = SymmetricalTransFormer,state_dict_path=checkpoint_path,preprocess=remove_ga_gs_aliases,postprocess=rewire_g_a_s)[1]
    oldMask = torch.load(mask_path, map_location='cpu')

    # parameters_to_prune 
    parameters_to_prune = {}  
    parameters_to_prune["g_a"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], net.g_a.modules())]
    parameters_to_prune["g_s"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], net.g_s.modules())]
    
    mask_to_device(oldMask, device="cuda")
    net = net.to("cuda")

    refnet = models["stf"]().to("cuda")

    if log_wandb:
        wandb.init(
            project='training',
            entity='MagV',
            name='inference_stf_14pt',
        )

    # newPoint = np.linspace(0, 0.6, 14)[::-1]
    
    # # Get the corresponding percentage for the new pruning point on the exponential curve 
    # newPointPercentage =  lambda_percentage(newPoint, amount = 0.6)[1]
 
    # # Generate new mask from unstructured pruning on reference model
    # newMask={}
    # newMask_g_a ,_ = generate_mask_from_unstructured(refnet.g_a,newPointPercentage)
    # newMask_g_s ,_ = generate_mask_from_unstructured(refnet.g_s,newPointPercentage)


    # #Put mask on GPU
    # for k, v in newMask_g_a[0].items():
    #      newMask_g_a[0][k] = v.to("cuda")

    # for k, v in newMask_g_s[0].items():
    #      newMask_g_s[0][k] = v.to("cuda")
    

    # Get the index of the new pruning
    # linAmount = (np.linspace(0.0, 0.6, 14)[::-1]).tolist()
    # for i, point in enumerate(newPoint):
    #     idx = bisect.bisect_left([-x for x in linAmount], -point)
    #     linAmount.insert(idx, point)

    #     print(f"Applying new mask at index {idx} corresponding to pruning point {point} with percentage {newPointPercentage[i]}")

    #     # Apply the new mask to both g_a and g_s
    #     mask["g_a"].insert(idx,newMask_g_a[i])
    #     mask["g_s"].insert(idx,newMask_g_s[i])
    # newMask["g_a"] = newMask_g_a
    # newMask["g_s"] = newMask_g_s

    make_plot("stf",net,oldMask,None,parameters_to_prune,"cuda",kodak_dataloader,"inference_stf_14pt",log_wandb=log_wandb)

    print("succeess")