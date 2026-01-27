# import sys
# sys.path.append("/home/ids/flauron-23/STF-QVRF")
# from compressai.entropy_models import GaussianConditional
# from compressai.models.priors import CompressionModel

#from Cheng2020Attention import Cheng2020Attention

from opt import parse_args
from utils import seed_all
# import os
# import wandb

from experiment import Experiment, Context
from utils import save_checkpoint

from custom_comp.zoo import models

from custom_comp.models import SymmetricalTransFormer, WACNN,QVRFCheng,stfQVRF
from custom_comp.zoo import load_model_from_checkpoint,load_mask,models
from custom_comp.zoo.pretrained import load_pretrained as load_state_dict
from utils.dataset import TestKodakDataset,TestClicDataset
from utils.engine import AverageMeter
from utils import compute_metrics, crop,pad,apply_saved_mask,delete_mask
# from evaluate import plot_rate_distorsion_psnr

from compressai.models.waseda import Cheng2020Attention
from compressai.models.google import MeanScaleHyperprior
from compressai.zoo import cheng2020_attn,mbt2018_mean
# from compressai.zoo.pretrained import load_pretrained

from torch import load,zeros_like
from torch.utils.data import DataLoader
from typing import Union

import torch
from tqdm import tqdm
import numpy as np

import json
import matplotlib.pyplot as plt
Colors = {
    "cheng": ["g",'-'],
    "ours": ["b",'-'],
    "msh": ["c",'-'],
    "qvrf": ["r",'-'],
    "gained": ["m",'-'],
}

def inference(model,x, x_padded, padding):

    out_enc = model.compress(x_padded)
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
    # out_dec["x_hat"] = F.pad(out_dec["x_hat"], padding) 

    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    rate = bpp*num_pixels 

    del out_enc
    del out_dec

    return metrics, torch.tensor([bpp]), rate

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

def eval_models(models, dataloader, device):

    print("Starting inferences")
    # estimate_entropy = False # False -> Use the AC for compression 

    res_metrics = {}
    with torch.no_grad():
        for j,x in enumerate(tqdm(dataloader)):

            if j>= 30:
                break


            
            
            for model_type in list(models.keys()):
                print("Evaluating model type:", model_type)
                x = x.to(device[model_type])
                # TCM padding 
                x_padded, padding = pad(x, 128)

                for qp in list(models[model_type].keys()):
                    print("Evaluating model:", qp)
                    model = models[model_type][qp]['model']
                    #model.update(force=True)

                    # if not estimate_entropy:
                    #     inference_fn = inference
                    # else:
                    #     inference_fn = inference_entropy_estimation

                    if model_type == "ours":
                        parameters_to_prune = {}
                        parameters_to_prune["g_a"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], model.g_a.modules())]
                        parameters_to_prune["g_s"] = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], model.g_s.modules())]

                        #Apply mask
                        
                        if qp < 5 :
                            apply_saved_mask(model.g_a, mask["g_a"][qp])
                            apply_saved_mask(model.g_s, mask["g_s"][qp])
                        
                        metrics, bpp, rate = inference(model, x, x_padded, padding)

                        if qp < 5 :
                            delete_mask(model.g_a, parameters_to_prune["g_a"])
                            delete_mask(model.g_s, parameters_to_prune["g_s"])

                    elif model_type in ["cheng", "msh"]:

                        metrics, bpp, rate = inference(model, x, x_padded, padding)

                    elif model_type == "qvrf":
                        #model.update(force=True)
                        factor = qvref_factor[qp]  #-1 qp start from 1
                        metrics, bpp, rate = inference_with_scale(model, x, x_padded, padding, factor, s=2)
                    
                    elif model_type == "gained":
                        #model.update(force=True)
                        s= gained_factor[qp]  #-1 qp start from 1
                        metrics, bpp, rate = inference_with_scale(model, x, x_padded, padding, 0.0, s=s)


                    models[model_type][qp]['psnr'].update(metrics["psnr"])
                    models[model_type][qp]['ms_ssim'].update(metrics["ms-ssim"])
                    models[model_type][qp]['bpps'].update(bpp.item())
                    models[model_type][qp]['rate'].update(rate)



    for model_type in list(models.keys()):
        model_res = {}
        for qp in list(models[model_type].keys()):
            model_res[qp] = {
                'psnr': models[model_type][qp]['psnr'].avg,
                'mssim': models[model_type][qp]['ms_ssim'].avg,
                'bpp': models[model_type][qp]['bpps'].avg,
                'rate': models[model_type][qp]['rate'].avg,
                'loss': models[model_type][qp]['loss'].avg
            }
            print(f'{qp}: {model_res[qp]}')
        res_metrics[model_type] = model_res

    return res_metrics   



def mask_to_device(mask, device="cuda"):
    for key, mask_list in mask.items():
        for i in range(len(mask_list)):
            for k, v in mask_list[i].items():
                mask_list[i][k] = v.to(device)


def extract_specific_model_performance(metrics, dataset, model, qp):

    nms = list(metrics[dataset][model][qp].keys())

    psnr = []
    mssim = []
    bpp = []
    rate = []
    for names in nms:
        psnr.append(metrics[dataset][model][qp]["psnr"])
        mssim.append(metrics[dataset][model][qp]["mssim"])
        bpp.append(metrics[dataset][model][qp]["bpp"])
        rate.append(metrics[dataset][model][qp]["rate"])

    
    return sorted(psnr), sorted(mssim), sorted(bpp), sorted(rate)


def plot_rate_distorsion_psnr(metrics, savepath, colors = Colors):

    print(f'plotting on {savepath}')

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    # plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')
    # dataset, model,quality
    for dataset in metrics.keys():
        for model in metrics[dataset].keys():
            #labeled = False  # flag for legend
            all_bpp = []
            all_psnr = []
            for qp in metrics[dataset][model].keys():

                psnr, mssim, bpp, rate = extract_specific_model_performance(metrics, dataset, model, qp)
                all_bpp.extend(bpp)
                all_psnr.extend(psnr)      
                cols = colors[model]
                #if not labeled:

            axes.plot(all_bpp, all_psnr, linestyle=cols[1], marker='o', color=cols[0], label=model)
                #    labeled = True
                #else:
                #    axes.plot(bpp, psnr, linestyle=cols[1], marker='o', color=cols[0])

                #axes.plot(bpp, psnr,'o',color = cols[0])
                #axes.plot(bpp, psnr,cols[1],color =  cols[0])



    axes.set_ylabel('PSNR [dB]')
    axes.set_xlabel('Bit-rate [bpp]')

    # axes.set_ylim([30.5, 38.3])

    axes.title.set_text(f'PSNR comparison')
    axes.grid()
    axes.legend(loc='best')

    # for ax in axes:
    axes.grid(True)
    plt.savefig(savepath,bbox_inches='tight', transparent="True", pad_inches=0.5)
    plt.close()      



#Load Cheng magv mask
mask_path="/home/ids/flauron-23/MagV/data/magv_ablation_0.4/masks/mask_magv_ablation_0.4.pth"
mask = load_mask(mask_path=mask_path)
mask_to_device(mask, "cuda")

#Load STF magv
#checkpoint_path = "/home/ids/flauron-23/MagV/data/magv_04_stf_unstructured/models/magv_04_stf_unstructured_checkpoint_best.pth.tar"
#mask_path="/home/ids/flauron-23/MagV/data/magv_04_stf_unstructured/masks/mask_magv_04_stf_unstructured.pth"
#stf_magv =  load_model_from_checkpoint(SymmetricalTransFormer,checkpoint_path=checkpoint_path)
#stf_mask = load_mask(mask_path=mask_path)
# qvref_factor = [0.5, 0.7, 0.9 ]#1.1, 1.25, 1.45, 1.7, 2.0, 2.4, 2.8, 3.3, 3.8, 4.0,4.6, 5.4, 5.8, 6.5, 6.8, 7.5, 7.9, 8.3, 9.1, 9.4, 9.7, 10.5, 11, 12]
qvref_factor = [0.5,1.25,2.4,4.0,6.5]#,8.3,10.5,12]
# gained_lambda =  [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]
# gained_factor = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
gained_factor = [0,1,2,3,4,5,6]

nets = {}
metrics = {}
qp_range = {
    "cheng": range(1, 7),
    "msh": range(1, 9),
    "qvrf": range(5),  #   index of qvrf factor
    "gained": range(7), #   index of gained factor
    "ours": range(6),   #   index of mask in the mask list 0 to 4 mask , 5 anchor model
}
device = {
    "cheng": "cuda",
    "msh": "cpu",
    "qvrf": "cuda",  #   index of qvrf factor
    "gained": "cpu", #   index of gained factor
    "ours": "cuda",   #   index of mask in the mask list 0 to 4 mask , 5 anchor model
}
def make_entry(model):
    return {
        "model": model,
        "psnr": AverageMeter(),
        "ms_ssim": AverageMeter(),
        "bpps": AverageMeter(),
        "rate": AverageMeter(),
        "criterion": None,
        "loss": AverageMeter(),
    }

def build_model(arch, qp, device):
    if arch == "cheng":
        return cheng2020_attn(quality=qp, pretrained=True).eval().to(device)
    if arch == "msh":
        return mbt2018_mean(quality=qp, pretrained=True).eval().to(device)
    # others reuse the same model
    return models[arch]().eval().to(device)


nets = {}
for arch, qps in qp_range.items():
    res = {}

    print("arch:", arch)
    #Non VR models
    if arch in["cheng","msh"]:
        for qp in qps:
            print("  qp:", qp)
            model = build_model(arch, qp, device[arch])
            model.update(force=True)
            res[qp] = make_entry(model)
            print(res.keys())
    # VR models
    else:
        model = build_model(arch, None, device[arch])
        model.update(force=True)
        for qp in qps:
            print("  qp:", qp)
            res[qp] = make_entry(model)
            print(res.keys())
    
    nets[arch] = res
    print("nets keys:", nets.keys())

for dataset_name in ["kodak","clic"]:

    if dataset_name=="kodak":
        dataset = TestKodakDataset("/home/ids/flauron-23/kodak")
    if dataset_name=="clic":
        dataset = TestClicDataset("/home/ids/flauron-23/clic")

    test_dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=4)

    metrics[dataset_name] = eval_models(nets, test_dataloader, device)

    file_path = f"/home/ids/flauron-23/MagV/src/results/metrics_{dataset_name}.json"
    img_path = f"/home/ids/flauron-23/MagV/src/results/metrics_{dataset_name}.pdf"
    with open(file_path, 'w') as outfile:
        json.dump(metrics, outfile)

    plot_rate_distorsion_psnr(metrics,img_path,colors=Colors)

