import torch 
import os 
from torchvision import transforms
from PIL import Image, ImageChops
import torch
import torch.nn.functional as F
# from training.step import AverageMeter
from utils.engine import AverageMeter, crop, pad
import math 
import matplotlib.pyplot as plt
import json
import argparse
from compressai.zoo import *
from utils.dataset import TestKodakDataset
from torch.utils.data import DataLoader
from os.path import join 
import wandb
from custom_comp.zoo import models

from torch.profiler import profile, record_function, ProfilerActivity

from torchvision.utils import save_image

import sys

from evaluate.bd_metrics import *

# from comp.zoo.pretrained import load_pretrained
# from compressai.zoo.pretrained import load_pretrained
from custom_comp.zoo.pretrained import load_pretrained
from compressai.zoo import cheng2020_attn, mbt2018_mean, bmshj2018_hyperprior, mbt2018
from evaluate.colors_models import Colors
from utils import compute_metrics, seed_all
from utils.masks import apply_saved_mask, delete_mask
from utils.engine import compress_one_epoch

from collections import defaultdict
import json

from tqdm import tqdm
import compressai




import seaborn as sns
palette = sns.color_palette("tab10")

def plot_rate_distorsion(bpp_res, psnr_res,epoch, eest = "compression",metric = 'PSNR', save_fig = False, file_name = None, log_wandb = True, is_psnr = True):

    chiavi_da_mettere = list(psnr_res.keys())


    legenda = {}
    for i,c in enumerate(chiavi_da_mettere):
        legenda[c] = {}
        legenda[c]["colore"] = [palette[i],'-']
        legenda[c]["legends"] = c
        legenda[c]["symbols"] = ["*"]*300
        legenda[c]["markersize"] = [5]*300    



    plt.figure(figsize=(12,8)) # fig, axes = plt.subplots(1, 1, figsize=(8, 5))


    list_names = list(psnr_res.keys())

    if is_psnr:
        minimo_bpp, minimo_psnr = 10000,1000
        massimo_bpp, massimo_psnr = 0,0

    for _,type_name in enumerate(list_names): 

        bpp = bpp_res[type_name]
        psnr = psnr_res[type_name]
        colore = legenda[type_name]["colore"][0]
        #symbols = legenda[type_name]["symbols"]
        #markersize = legenda[type_name]["markersize"]
        leg = legenda[type_name]["legends"]


        bpp = torch.tensor(bpp).cpu()
        psnr = torch.tensor(psnr).cpu()
    
        plt.plot(bpp,psnr,"-" ,color = colore, label =  leg ,markersize=8)
        
        plt.plot(bpp, psnr, marker="o", markersize=4, color =  colore)

        if is_psnr:
            for j in range(len(bpp)):
                if bpp[j] < minimo_bpp:
                    minimo_bpp = bpp[j]
                if bpp[j] > massimo_bpp:
                    massimo_bpp = bpp[j]
                
                if psnr[j] < minimo_psnr:
                    minimo_psnr = psnr[j]
                if psnr[j] > massimo_psnr:
                    massimo_psnr = psnr[j]


    if is_psnr:
        minimo_psnr = int(minimo_psnr)
        massimo_psnr = int(massimo_psnr)
        psnr_tick =  [round(x) for x in range(minimo_psnr, massimo_psnr + 2)]
        plt.yticks(psnr_tick)
    
    plt.ylabel(metric, fontsize = 30)   
    


    #print(minimo_bpp,"  ",massimo_bpp)

    if is_psnr:
        bpp_tick = [round(x)/10 for x in range(int(minimo_bpp*10), int(massimo_bpp*10 + 2))]
        plt.xticks(bpp_tick)
    plt.xlabel('Bit-rate [bpp]', fontsize = 30)
    plt.yticks(fontsize=27)
    plt.xticks(fontsize=27)
    plt.grid()

    plt.legend(loc='best', fontsize = 25)



    plt.grid(True)
    if log_wandb:
        wandb.log({f"{eest}":epoch,
                f"{eest}/rate distorsion trade-off": wandb.Image(plt)}, step=epoch)      

    if save_fig:
        plt.savefig(file_name) 
    plt.close()  

def load_models(models_path, model_checkpoints, device, model_name, force_alpha = None):

    res = {}
    for model_checkpoint in model_checkpoints:
        
        model_path = join(models_path, model_checkpoint)

    
        checkpoint = torch.load(model_path, map_location=device)

        if 'adapt_stf' in model_name: 
            print(f'Loading stf w/ adapters model {model_checkpoint}')

            model = models['stf']()

            if 'vanilla' not in model_name:
                model = get_lora_model(model, '../configs/lora_8_8.yml', force_alpha=force_alpha)

            model.load_state_dict(checkpoint["state_dict"])

            if 'merge' in model_name:
                print('merging!')
                model = get_merged_lora(model)

            model = model.to(device)
            model.update(force = True)
            model.eval()

        elif(model_name in ['stf']):
            print(f'Loading stf model {model_checkpoint}')

            state_dict = load_pretrained(checkpoint['state_dict'])
            model =  models[model_name].from_state_dict(state_dict)


            model = model.to(device)
            model.update(force = True)
            model.eval()

        else:
            raise NotImplementedError(f'Model {model_name} not yet implemented')

        key_name = f'{model_name}_{model_checkpoint}'
        # print(f'Creating model w/ key: {key_name}')
        res[key_name] = {
            "model": model,
            "psnr": AverageMeter(),
            "ms_ssim": AverageMeter(),
            "bpps": AverageMeter(),
            "rate": AverageMeter(),
            "criterion": None,
            "loss": AverageMeter()
            }
        # print(f'{model_path} loaded')
    return res





@torch.no_grad()
def inference_entropy_estimation(model,x, x_padded, padding, criterion = None):

    out  = model(x_padded)
    if criterion is not None:
        out_criterion = criterion(out, x_padded)
    # out["x_hat"] = F.pad(out["x_hat"], unpad)
    out["x_hat"] = crop(out["x_hat"], padding)

    metrics = compute_metrics(x, out["x_hat"], 255)
    size = out['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]

    y_bpp = torch.log(out["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
                        
    bpp = y_bpp
    rate = bpp.item()*num_pixels 


    return metrics, bpp, rate, out["x_hat"] #, out_criterion["loss"].item()


@torch.no_grad()
def inference(model,x, x_padded, padding):

    out_enc = model.compress(x_padded)
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
    # out_dec["x_hat"] = F.pad(out_dec["x_hat"], padding) 

    metrics = compute_metrics(x, out_dec["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    rate = bpp*num_pixels 

    return metrics, torch.tensor([bpp]), rate, out_dec["x_hat"]




@torch.no_grad()
def eval_models(models, dataloader, device):

    print("Starting inferences")
    estimate_entropy = False # False -> Use the AC for compression 

    res_metrics = {}
    for j,x in enumerate(tqdm(dataloader)):

        if j>= 30:
            break

        x = x.to(device)
            
        # TCM padding 
        x_padded, padding = pad(x, 128)
            
        for model_type in list(models.keys()):

            for qp in list(models[model_type].keys()):
                model = models[model_type][qp]['model']


                if not estimate_entropy:
                    inference_fn = inference
                else:
                    inference_fn = inference_entropy_estimation

                metrics, bpp, rate, x_hat = inference_fn(model, x, x_padded, padding)

                    
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




def extract_specific_model_performance(metrics, type):

    nms = list(metrics[type].keys())

    psnr = []
    mssim = []
    bpp = []
    rate = []
    for names in nms:
        psnr.append(metrics[type][names]["psnr"])
        mssim.append(metrics[type][names]["mssim"])
        bpp.append(metrics[type][names]["bpp"])
        rate.append(metrics[type][names]["rate"])

    
    return sorted(psnr), sorted(mssim), sorted(bpp), sorted(rate)


def plot_rate_distorsion_psnr(metrics, savepath, colors = Colors):

    print(f'plotting on {savepath}')

    fig, axes = plt.subplots(1, 1, figsize=(7, 5))
    # plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')
    for type_name in metrics.keys():

        psnr, mssim, bpp, rate = extract_specific_model_performance(metrics, type_name)      
        cols = colors[type_name]      
        axes.plot(bpp, psnr,cols[1],color = cols[0], label = type_name)
        axes.plot(bpp, psnr,'o',color = cols[0])
        axes.plot(bpp, psnr,cols[1],color =  cols[0])


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

     

def produce_metrics(configs, dataset_path = 'kodak', saved_checkpoint = True):

    # Loading dict of models
    models = {}
    device = "cuda"


    if saved_checkpoint:
        for config in configs:
            with open(config) as f:
                args = json.load(f)

            model_name = args['model']
            models_path = args['checkpoints_path']

            if 'alpha' in args.keys():
                force_alpha = args['alpha']
            else:
                force_alpha = None

            models_checkpoint = []
            for entry in os.listdir(models_path):
                if('best.pth.tar' in entry):
                    models_checkpoint.append(entry) # checkpoints models  
            print(f'Model: {model_name}')
            print(f'Founded checkpoints: {models_checkpoint}')
            print(f'Founded force_alpha: {force_alpha}')


            res = load_models(models_path, models_checkpoint, device, model_name, force_alpha)
            models[model_name] = res
    else:
        for model_arch in configs:
            res = {}

            for qp in range(1,7):
                if model_arch == 'Cheng2020':
                    net = cheng2020_attn(quality=qp, pretrained=True).eval().to(device)
                elif model_arch == 'Minnen2018':
                    net = mbt2018(quality=qp, pretrained=True).eval().to(device)
                elif model_arch == 'Ballé2018':
                    net = bmshj2018_hyperprior(quality=qp, pretrained=True).eval().to(device)
                
                res[f'q{qp}-model'] = {   
                    "model": net,
                    "psnr": AverageMeter(),
                    "ms_ssim": AverageMeter(),
                    "bpps": AverageMeter(),
                    "rate": AverageMeter(),
                    "criterion": None,
                    "loss": AverageMeter()
                }
            models[model_arch] = res

    # models = {
    #     'stf': {
    #         'model': model,
    #         'psnr': AverageMeter,
    #         '...'
    #     },
    #     'Ballè': {
    #         '...'
    #     }
    # }

    # Test Set

    test_dataset = TestKodakDataset(data_dir= dataset_path)

    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=4)  

    metrics = eval_models(models, test_dataloader, device)

    return metrics




if __name__ == "__main__":

    seed_all(42)

    compressai.available_entropy_coders()[0]
    
    my_parser = argparse.ArgumentParser(description= "path to read the configuration of the evaluation")
    
    my_parser.add_argument("--metrics", default=None, type=str,  help='metrics json file')
    
    my_parser.add_argument("--test-dir", type = str, help = "Kodak Test directory", default = "/data/kodak")

    my_parser.add_argument("--file-name", default='res_kodak', type=str)
    my_parser.add_argument("--save-path", default='../pretrained/stf/', type=str)


    args = my_parser.parse_args()
    
    configs = [
        '../pretrained/stf/inference.json',
        '../pretrained/adapt_0483_seed_42_conf_lora_8_8_opt_adam_sched_cosine_lr_0_0001/inference.json',
        # '../pretrained/adapt_0483_seed_42_conf_lora_8_8_opt_adam_sched_cosine_lr_0_0001/inference_merge.json',
        '../pretrained/adapt_0483_seed_42_conf_vanilla_adapt_opt_adam_sched_cosine_lr_0_0001/inference.json'
    ]

    # configs = [
    #     'Ballé2018',
    #     'Minnen2018',
    #     'Cheng2020'
    # ]


    new_metrics = {}
    if(args.metrics is None):

        os.makedirs(args.save_path, exist_ok=True)
        print(f'Results will be saved on {args.save_path}')


        new_metrics = produce_metrics(configs, dataset_path = args.test_dir, saved_checkpoint=True)

        file_path = join(args.save_path,f'{args.file_name}.json')
        with open(file_path, 'w') as outfile:
            json.dump(new_metrics, outfile)

        save_path_img = join(args.save_path,f'{args.file_name}.pdf')

    else:
        work_path = '/'.join(args.metrics.split('/')[:-1])

        with open(args.metrics) as json_file:
            new_metrics = json.load(json_file)
        save_path_img = join(work_path,f'{args.file_name}.pdf')
        args.save_path = work_path
        



    plot_rate_distorsion_psnr(new_metrics,save_path_img, colors=Colors)

def compress_with_masks(net,masks ,dataloader, parameters_to_prune,device,max_images = None,anchor = False):
    """
    Docstring for compress_with_masks
    
    :param net: The model to be evaluated
    :param masks: The masks to be applied for pruning
    :param dataloader: The dataloader for the test set
    :param parameters_to_prune: The parameters to be pruned in the model
    :param device: The device to be used for computation
    :param max_images: The maximum number of images to be evaluated (default: None, which means all images will be evaluated)
    :param anchor: If True, include an anchor point (i.e., no pruning case). Some checkpoints contains only the pruned masks, so the anchor point is not always available (default: False)
    :return: The bpp, psnr and mssim lists for the given masks
    """
    bpp_list = []
    psnr_list = []
    mssim_list = []

    print("Make actual compression")
    net.update(force = True)

    
    for index in range(len(masks["g_a"])):# +1 to include no pruning case

        #if index < len(masks["g_a"]):#if index != len(self.ctx.all_mask["g_a"]): # Last index is no pruning 
            
        apply_saved_mask(net.g_a, masks["g_a"][index])
        apply_saved_mask(net.g_s, masks["g_s"][index])

        bpp_ac, psnr_ac, mssim_ac = compress_one_epoch(net, dataloader, device, max_images=max_images)

        delete_mask(net.g_a,parameters_to_prune["g_a"])
        delete_mask(net.g_s,parameters_to_prune["g_s"])

        # # No mask for 0.483 lambda 0.0 amount pruning   
        # else: 
        #         bpp_ac, psnr_ac, mssim_ac = compress_one_epoch(net, dataloader, device, max_images=max_images)
        
        bpp_list.append(bpp_ac)
        psnr_list.append(psnr_ac)
        mssim_list.append(mssim_ac)

    if anchor:
        bpp_ac, psnr_ac, mssim_ac = compress_one_epoch(net, dataloader, device, max_images=max_images)

        bpp_list.append(bpp_ac)
        psnr_list.append(psnr_ac)
        mssim_list.append(mssim_ac)

    return bpp_list, psnr_list, mssim_list
    
def make_plot(model,net,old_masks,new_masks,parameters_to_prune,device,dataloader,name,log_wandb=False,max_images = None,anchor = False):
            psnr_res = {}
            mssim_res = {}
            bpp_res = {} 

            if parameters_to_prune is not None:
                if old_masks is not None :
                    bpp_res["old_masks"],psnr_res["old_masks"],mssim_res["old_masks"] = compress_with_masks(net,old_masks,dataloader,parameters_to_prune,device,max_images,anchor)
                if new_masks is not None :
                    bpp_res["new_masks"],psnr_res["new_masks"],mssim_res["new_masks"] = compress_with_masks(net,new_masks,dataloader,parameters_to_prune,device,max_images,anchor)
            else:#qvrf
                bpp_res[model],psnr_res[model],mssim_res[model] = compress_one_epoch(net, dataloader, device, max_images=max_images)    
            # Reference results from json
            # with open("/home/ids/flauron-23/MagV/json/ref_results.json", "r") as f:
            #     ref_results = json.load(f)

            # bpp_res[model] = ref_results["bpp"][model]
            # psnr_res[model] = ref_results["psnr"][model]
            # mssim_res[model] = ref_results["mssim"][model]

            # plot_rate_distorsion(bpp_res, psnr_res, 
            #                     "inference_stf", eest="compression", 
            #                     metric = 'PSNR',
            #                     save_fig=True,
            #                     file_name=os.path.join("results", f"psnr_{name}.png"),
            #                     log_wandb=log_wandb)

            # plot_rate_distorsion(bpp_res, 
            #                     mssim_res, 
            #                     "inference_stf", 
            #                     eest="compression_mssim", 
            #                     metric = 'MS-SSIM',
            #                     save_fig=True,
            #                     file_name=os.path.join("results", f"mssim_{name}.png"),
            #                     log_wandb=log_wandb, is_psnr=False)

            #Save data dictionnary
            results = {"psnr": psnr_res,"mssim": mssim_res,"bpp": bpp_res}
            file_path = os.path.join("results",f"{name}.json")
            with open(file_path, "w") as f:
                json.dump(results, f)