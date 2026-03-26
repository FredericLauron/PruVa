import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from custom_comp.datasets import ImageFolder
from custom_comp.zoo import models

from utils import CustomDataParallel,configure_optimizers
from utils.loss import RateDistortionLoss
from utils.dataset import TestKodakDataset

from utils.masks import lambda_percentage,\
                        generate_mask_from_unstructured,\
                        generate_mask_from_structured

class Context:
    """Experiment context"""

    def __init__(self, args,project_run_path):
        self.args = args
        self.project_run_path = project_run_path

        self.log_wandb = True

        #File paths for saving images and data
        self.img_dir = self._get_folder_path("imgs")
        self._make_dir(self.img_dir)

        self.data_dir = self._get_folder_path("data")
        self._make_dir(self.data_dir)

        self.mask_dir = self._get_sub_folder_path(self.data_dir,'masks')
        self._make_dir(self.mask_dir)

        self.res_dir = self._get_sub_folder_path(self.data_dir,'results')
        self._make_dir(self.res_dir)

        self.model_dir = self._get_sub_folder_path(self.data_dir,'models')
        self._make_dir(self.model_dir)

        # configure device, dataloaders, model, 
        self.device = self._get_device(args)
        self.train_dataloader, self.val_dataloader, self.kodak_dataloader = self._get_dataloaders(args)
        self.net = self._get_model(args)

        #Put model on GPU after injecting adapter
        self.net=self.net.to(self.device)

        # configure optimizers, criterion, lr scheduler
        self.optimizer, self.aux_optimizer = self._get_optimizers(self.net, args)
        self.criterion = self._get_criterion(args)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer, args)
        
        # Adapter
        # Done after optimizer to avoid trigger error
        # Froze all the parameters except the switches

        self.last_epoch = 0
        self.best_val_loss = float("inf")
        self.best_kodak_loss = float("inf")
        
        # checkpoint
        if args.checkpoint is not None:
            self._get_checkpoint(args)

        # Need to be set even for adapter to avoid errors
        self.alpha = np.linspace(args.minPruning, args.maxPrunning, args.maxPoint)[::-1]
        self.lambda_list , self.amounts = lambda_percentage(self.alpha, amount = args.maxPrunning, lambda_max=args.lambda_max,lambda_min=args.lambda_min)

        if args.put_lambda_max:
            self.lambda_list[-1] == args.lambda_max

        # Masks
        if args.mask:
            if args.pruningType == "unstructured":
                self.all_mask, self.parameters_to_prune = self._get_mask_unstructured(args)
            elif args.pruningType == "structured":
                self.all_mask = self._get_mask_structured(args)
                self.parameters_to_prune = None

    
    def _get_folder_path(self,folder_name):
        return os.path.join(self.project_run_path, "..", folder_name, self.args.nameRun)       

    def _get_sub_folder_path(self,folder,subfolder):
        return os.path.join(folder,subfolder)
       
    def _make_dir(self,dir,exist_ok=True):
        os.makedirs(dir, exist_ok=exist_ok)

    def _get_device(self,args):
        return "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    # def _multi_gpu(self,net,args):
    #     if args.cuda and torch.cuda.device_count() > 1:
    #         print(f'Training on: {torch.cuda.device_count()} GPUs')
    #         net = CustomDataParallel(net)
    #     return net
    
    def _get_dataloaders(self,args):

        train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
        test_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])

        train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
        test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
        kodak_dataset = TestKodakDataset(data_dir = args.test_dir) # Kodak test set

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=(self.device == "cuda"),
            )

        val_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(self.device == "cuda"),
            )

        kodak_dataloader = DataLoader(
            kodak_dataset, 
            shuffle=False, 
            batch_size=1, 
            pin_memory=(self.device == "cuda"), 
            num_workers= args.num_workers 
            )
        
        return train_dataloader, val_dataloader, kodak_dataloader
    
    @staticmethod
    def get_kodak_dataloader(batch_size=1, num_workers=30, test_dir="/home/ids/flauron-23/kodak", device="cuda"):
        kodak_dataset = TestKodakDataset(data_dir = test_dir) # Kodak test set

        kodak_dataloader = DataLoader(
            kodak_dataset, 
            shuffle=False, 
            batch_size=batch_size, 
            pin_memory=(device=="cuda"), 
            num_workers= num_workers 
            )
        
        return kodak_dataloader

    @staticmethod
    def get_dataloaders(patch_size=(256,256),batch_size=16,test_batch_size=64,num_workers=30,dataset="/home/ids/flauron-23/fiftyone/open-images-v6",test_dir="/home/ids/flauron-23/kodak",device="cuda"):
    
        train_transforms = transforms.Compose([transforms.RandomCrop(patch_size), transforms.ToTensor()])
        test_transforms = transforms.Compose([transforms.CenterCrop(patch_size), transforms.ToTensor()])

        train_dataset = ImageFolder(dataset, split="train", transform=train_transforms)
        test_dataset = ImageFolder(dataset, split="test", transform=test_transforms)
        kodak_dataset = TestKodakDataset(data_dir = test_dir) # Kodak test set

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=(device=="cuda"),
            )

        val_dataloader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=(device=="cuda"),
            )

        kodak_dataloader = DataLoader(
            kodak_dataset, 
            shuffle=False, 
            batch_size=1, 
            pin_memory=(device=="cuda"), 
            num_workers= num_workers 
            )
        
        return train_dataloader, val_dataloader, kodak_dataloader

    def _get_model(self,args):
        net = models[args.model]()
        #net = net.to(self.device)

        # multi GPUs
        if args.cuda and torch.cuda.device_count() > 1:
            print(f'Training on: {torch.cuda.device_count()} GPUs')
            net = CustomDataParallel(net)
        else:
            print('Training on a single GPU or CPU')

        return net

    def _get_optimizers(self,net,args):
        optimizer, aux_optimizer = configure_optimizers(net, args)
        return optimizer, aux_optimizer
    
    def _get_criterion(self,args):
        return RateDistortionLoss(lmbda=args.lmbda)
    
    def _get_lr_scheduler(self,optimizer,args):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
 
    def _get_mask_unstructured(self,args):
        
            all_mask,parameters_to_prune = {}, {}

            all_mask["g_a"], parameters_to_prune["g_a"] = generate_mask_from_unstructured(self.net.g_a, self.amounts)
            all_mask["g_s"], parameters_to_prune["g_s"] = generate_mask_from_unstructured(self.net.g_s, self.amounts)

            # Save masks 
            torch.save(all_mask, f"{self.mask_dir}/mask_{args.nameRun}.pth")
            torch.save(parameters_to_prune, f"{self.mask_dir}/parameters_to_prune_{args.nameRun}.pth")
            
            return all_mask, parameters_to_prune
              
    def _get_mask_structured(self,args):

        all_mask = generate_mask_from_structured(self.net,self.amounts)
        
        torch.save(all_mask, f"{self.mask_dir}/mask_{args.nameRun}.pth")

        return all_mask

    def _get_checkpoint(self,args):

        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=self.device)
        self.net.load_state_dict(checkpoint["state_dict"])

        if args.resume_train:
            print('Loading elements from checkpoint to resume the training')
            self.last_epoch = checkpoint["epoch"] + 1

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.aux_optimizer is not None and checkpoint["aux_optimizer"] is not None:
                self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            self.best_val_loss = checkpoint["best_val_loss"]
            self.best_kodak_loss = checkpoint["best_kodak_loss"] 