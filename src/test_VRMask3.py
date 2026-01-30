import torch
import torch.fx as fx
import torch.nn as nn
from torch.nn.utils import prune

from simplify import simplify
import simplify.utils as sutils

from compressai.zoo import cheng2020_attn,mbt2018_mean
from compressai.models.waseda import Cheng2020Attention
from compressai.layers.gdn import GDN


from utils import apply_saved_mask,save_mask,group_by_module,delete_mask
from utils.masks import lambda_percentage,generate_mask_from_unstructured
from utils.engine import compress_one_epoch
from experiment import Experiment 
from collections import OrderedDict
from custom_comp.zoo import  load_state_dict,models

from itertools import chain
from collections import defaultdict

from opt import parse_args
import os
import sys
import warnings

import wandb
import numpy as np
import bisect

import copy
warnings.simplefilter("ignore", FutureWarning)

warnings.filterwarnings(
    "ignore",
    message=r"autocast",
    category=FutureWarning,
)

log_wandb = True

def load_model_from_checkpoint(checkpoint_path,factory_function,quality=6,pretrained=True,adapter=False):

    # Load the checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    print(state_dict["epoch"])
    state_dict = load_state_dict(state_dict=state_dict)

    # Create the model
    net = Cheng2020Attention().to("cuda")
    net.load_state_dict(state_dict["state_dict"])

    # #update entropy model
    net.update(force=True)

    return net

def load_mask(mask_path):
    mask = torch.load(mask_path, map_location='cpu')
    return mask

def remove_pruning(model):
    for module in model.modules():
        for param_name in ["weight", "bias"]:
            # Check if this param was pruned
            if hasattr(module, f"{param_name}_mask"):
                prune.remove(module, param_name)

def count_neurons_weight(model):
    """
    Counts the number of neurons in Conv2d and Linear layers
    based on the weight tensors.
    """
    total_neurons = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # For Conv2d: one neuron per output channel
            total_neurons += layer.out_channels
        elif isinstance(layer, nn.Linear):
            # For Linear: one neuron per output feature
            total_neurons += layer.out_features
    return total_neurons


###############################################################################################""



def get_superset_mask(mask,idx_inf,idx_sup):
    #Retrive element of the superset mask that are non in the subset mask 
    superset_Mask ={}
    for (k,v),(k2,v2) in zip(mask[idx_inf].items(),mask[idx_sup].items()):
        superset_Mask[k] = torch.bitwise_xor(v.bool(),v2.bool()).float()
    return superset_Mask

def get_superset_nonzero_parameters(net):
    superSet_data = []
    for _, module in net.named_modules():
        if torch.nn.utils.prune.is_pruned(module) and hasattr(module, "weight"):

            w = torch.flatten(module.weight.data)

            # Retrive non zero indices 
            NonZeroIndex = torch.nonzero(w).squeeze()

            # Retrive non zero values
            NonZeroValue=(w[NonZeroIndex])

            superSet_data.extend( [(module, idx,v) for idx ,v in zip(NonZeroIndex,NonZeroValue)])
    return superSet_data

def get_superset_param_to_prune(superSet_data,amount=0.5):
    superSet_data_sorted_desc = sorted(superSet_data, key=lambda x: x[2]) #sort ascending to target param to prune
    # Find the neurons to prune based on the sorted superset data
    superSet_idxPruning = int(len(superSet_data_sorted_desc) * amount)
    superSet_paramToPrune = superSet_data_sorted_desc[:superSet_idxPruning]
    return superSet_paramToPrune

def update_superset_mask(net,module_to_indices):
    # Find the neuron's parameters indices to prune
    # Put a zero into the superset mask at these indices
    for _, module in net.named_modules():
        if torch.nn.utils.prune.is_pruned(module) and hasattr(module, "weight"):
            for idx in module_to_indices[module]:
                idx = torch.unravel_index(idx, module.weight.shape)
                module.weight_mask.data[idx] = 0

def fuse_masks(superset_mask,inter_mask):
    # Fuse the smaller mask with the modified superset mask to create a new mask
    for name in inter_mask.keys():
        inter_mask[name] = inter_mask[name].to("cuda").bool()
        superset_mask[name] = superset_mask[name].bool()
        inter_mask[name] = torch.bitwise_or(inter_mask[name], superset_mask[name])
    return inter_mask

def mask_to_device(mask, device="cuda"):
    for key, mask_list in mask.items():
        for i in range(len(mask_list)):
            for k, v in mask_list[i].items():
                mask_list[i][k] = v.to(device)


if __name__ == "__main__":


    # Load model and mask

    # checkpoint_path = "/home/ids/flauron-23/MagV/data/magv_ablation_0.4/models/magv_ablation_0.4_checkpoint.pth.tar"
    # mask_path="/home/ids/flauron-23/MagV/data/magv_ablation_0.4/masks/mask_magv_ablation_0.4.pth"
    #parameters_to_prune_path = "/home/ids/flauron-23/MagV/data/magv_04_cheng_unstructured/masks/parameters_to_prune_magv_04_cheng_unstructured.pth"


    mask_path ="/home/ids/flauron-23/MagV/data/magv_06_stf_unstructured_40_epochs/masks/mask_magv_06_stf_unstructured_40_epochs.pth"
    checkpoint_path = "MagV/data/magv_06_stf_unstructured_40_epochs/models/magv_06_stf_unstructured_40_epochs_checkpoint.pth.tar"
    net = load_model_from_checkpoint(checkpoint_path=checkpoint_path,factory_function=cheng2020_attn)
    mask = load_mask(mask_path=mask_path)

    # parameters_to_prune 
    parameters_to_prune = {}  
    parameters_to_prune["g_a"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], net.g_a.modules())]
    parameters_to_prune["g_s"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], net.g_s.modules())]
    
    mask_to_device(mask, device="cuda")
    net = net.to("cuda")

    args = parse_args()

    # Load pretrained reference model to generate new mask
    #refnet = cheng2020_attn(quality=6, pretrained=True).to("cuda")
    refnet = models[args.model]().to("cuda")

    project_run_path=os.path.dirname(__file__)

    if log_wandb:
        wandb.init(
            project='training',
            entity='MagV',
            name=f'{args.nameRun}',
            config=vars(args)
        )

    exp = Experiment(args,project_run_path)
    #==================
    # test applying the new mask
    #==================

    # Define new pruning point
    #maxpruning = 0.6
    # newPoint = 0.2 #Here define the new pruning point you want to test
    newPoint = np.linspace(0, 0.4, 14)[::-1][1:-1]
    
    # Get the corresponding percentage for the new pruning point on the exponential curve 
    newPointPercentage =  lambda_percentage(newPoint, amount = args.maxPrunning)[1]
 
    # Generate new mask from unstructured pruning on reference model
    newMask_g_a ,_ = generate_mask_from_unstructured(refnet.g_a,newPointPercentage)
    newMask_g_s ,_ = generate_mask_from_unstructured(refnet.g_s,newPointPercentage)


    print(type(newMask_g_a[0]))
    print(type(mask["g_a"][0]))
    for k, v in newMask_g_a[0].items():
         newMask_g_a[0][k] = v.to("cuda")

    for k, v in newMask_g_s[0].items():
         newMask_g_s[0][k] = v.to("cuda")
    
    # Get the index of the new pruning
    linAmount = (np.linspace(0.0, args.maxPrunning, args.maxPoint)[::-1]).tolist()
    for i, point in enumerate(newPoint):
        idx = bisect.bisect_left([-x for x in linAmount], -point)
        linAmount.insert(idx, point)

        print(f"Applying new mask at index {idx} corresponding to pruning point {point} with percentage {newPointPercentage[i]}")

        # Apply the new mask to both g_a and g_s
        mask["g_a"].insert(idx,newMask_g_a[i])
        mask["g_s"].insert(idx,newMask_g_s[i])

    exp.ctx.net = net
    exp.ctx.all_mask = mask
    exp.ctx.parameters_to_prune = parameters_to_prune

    # apply_saved_mask(exp.ctx.net.g_a, exp.ctx.all_mask["g_a"][idx])
    # apply_saved_mask(exp.ctx.net.g_s, exp.ctx.all_mask["g_s"][idx])

    # bpp_ac, psnr_ac, mssim_ac = compress_one_epoch(exp.ctx.net, exp.ctx.kodak_dataloader, exp.ctx.device)

    # delete_mask(exp.ctx.net.g_a,exp.ctx.parameters_to_prune["g_a"])
    # delete_mask(exp.ctx.net.g_s,exp.ctx.parameters_to_prune["g_s"])

    exp.make_plot(epoch=0)

    print("succeess")