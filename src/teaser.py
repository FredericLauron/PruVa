import json

import numpy as np
import bisect

from utils.functions import compute_metrics
import torch
from torch.utils.data import DataLoader

from utils.masks import lambda_percentage,generate_mask_from_unstructured
from utils.dataset import TestClicDataset
from evaluate import make_plot
from custom_comp.zoo import models,model_loading,Cheng2020Attention,rewire_g_a_s

import wandb

from PIL import Image
from torchvision import transforms
from utils.masks import apply_saved_mask, delete_mask
from utils.engine import compress_one_epoch
from utils.engine import AverageMeter, crop, pad
from torchvision.utils import save_image
from custom_comp.zoo import models,model_loading,SymmetricalTransFormer,rewire_g_a_s

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

    img = Image.open("/home/ids/flauron-23/MagV/src/test.png").convert("RGB")
    transform = transforms.ToTensor()
    x = transform(img)
    x = x.unsqueeze(0).to("cuda")

    # x = torch.randn(1,3,512,512).to("cuda")

    checkpoint_path = "/home/ids/flauron-23/MagV/data/magv_60_stf_unstruct_41_epochs_14_points/models/magv_60_stf_unstruct_41_epochs_14_points_checkpoint.pth.tar"    
    mask_path ="/home/ids/flauron-23/MagV/data/magv_60_stf_unstruct_41_epochs_14_points/masks/mask_magv_60_stf_unstruct_41_epochs_14_points.pth"
    
    net = model_loading(model_class = SymmetricalTransFormer,state_dict_path=checkpoint_path,preprocess=remove_ga_gs_aliases,postprocess=rewire_g_a_s)[1]
    masks = torch.load(mask_path, map_location='cpu')

    parameters_to_prune = {}  
    parameters_to_prune["g_a"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], net.g_a.modules())]
    parameters_to_prune["g_s"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], net.g_s.modules())]
    
    mask_to_device(masks, device="cuda")
    net = net.to("cuda")

    bpp_metric = []
    psnr_metric = []

    x_padded, padding = pad(x, 128)

    out_enc = net.compress(x_padded)
    out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)

    save_image(out_dec["x_hat"].cpu(), f"/home/ids/flauron-23/MagV/src/image_anchor.png")

    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    
    psnr_metric.append(metrics["psnr"])
    bpp_metric.append(bpp)

    for index in range(len(masks["g_a"])):# +1 to include no pruning case

        #if index < len(masks["g_a"]):#if index != len(self.ctx.all_mask["g_a"]): # Last index is no pruning 
            
        apply_saved_mask(net.g_a, masks["g_a"][index])
        apply_saved_mask(net.g_s, masks["g_s"][index])

        out_enc = net.compress(x_padded)
        out_dec = net.decompress(out_enc["strings"], out_enc["shape"])
        out_dec["x_hat"] = crop(out_dec["x_hat"], padding)

        delete_mask(net.g_a,parameters_to_prune["g_a"])
        delete_mask(net.g_s,parameters_to_prune["g_s"])


        #save_image
        save_image(out_dec["x_hat"].cpu(), f"/home/ids/flauron-23/MagV/src/image_{index}.png")

        metrics = compute_metrics(x, out_dec["x_hat"], 255)
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
        
        psnr_metric.append(metrics["psnr"])
        bpp_metric.append(bpp)

    results = {
        "bpp": bpp_metric,
        "psnr": psnr_metric,
    }
    file_path = "/home/ids/flauron-23/MagV/src/teaser_metrics.json"
    with open(file_path, "w") as f:
        json.dump(results, f)  

    print("Done!")