import torch.optim as optim

from utils import save_checkpoint
from utils.engine import test_epoch,train_one_epoch, compress_one_epoch #, AverageMeter,pad,crop
from utils.masks import delete_mask, apply_saved_mask

from evaluate import plot_rate_distorsion

from compressai.zoo import cheng2020_attn,mbt2018_mean

from experiment.context import Context

import os
import json
import wandb

import wandb, matplotlib.pyplot as plt,inspect

class Experiment:
    def __init__(self,args,project_run_path):
        super().__init__()
        self.args = args
        self.ctx  = Context(args,project_run_path)
    
    def make_baseline_plot(self):
            bpp_list = []
            psnr_list = []
            mssim_list = []

            ref_bpp_list = []
            ref_psnr_list = []
            ref_mssim_list = []         

            print("Make actual compression")
            self.ctx.net.update(force = True)

            for index in range(len(self.ctx.lambda_list)):

                if index != len(self.ctx.lambda_list)-1:
                    
                    apply_saved_mask(self.ctx.net.g_a, self.ctx.all_mask["g_a"][index])
                    apply_saved_mask(self.ctx.net.g_s, self.ctx.all_mask["g_s"][index])

                    bpp_ac, psnr_ac, mssim_ac = compress_one_epoch(self.ctx.net, self.ctx.kodak_dataloader, self.ctx.device)

                    delete_mask(self.ctx.net.g_a,self.ctx.all_mask["g_a"][index])
                    delete_mask(self.ctx.net.g_s,self.ctx.all_mask["g_s"][index])

                # No mask for 0.483 lambda 0.0 amount pruning   
                else:
                    bpp_ac, psnr_ac, mssim_ac = compress_one_epoch(self.ctx.net, self.ctx.kodak_dataloader, self.ctx.device)
                
                bpp_list.append(bpp_ac)
                psnr_list.append(psnr_ac)
                mssim_list.append(mssim_ac)

            # for index in range(6):
            #     refnet = cheng2020_attn(quality=index+1,pretrained=True).to(self.ctx.device) # TODO change that to model agnostic
            #     ref_bpp_ac, ref_psnr_ac, ref_mssim_ac = compress_one_epoch(refnet, self.ctx.kodak_dataloader, self.ctx.device)

            #     ref_bpp_list.append(ref_bpp_ac)
            #     ref_psnr_list.append(ref_psnr_ac)
            #     ref_mssim_list.append(ref_mssim_ac)

            psnr_res = {}
            mssim_res = {}
            bpp_res = {} 

            bpp_res["ours"] = bpp_list
            psnr_res["ours"] = psnr_list
            mssim_res["ours"] = mssim_list
            
            with open("/home/ids/flauron-23/MagV/json/ref_results.json", "r") as f:
                ref_results = json.load(f)
                
            bpp_res[self.args.model] = ref_results["bpp"][self.args.model]
            psnr_res[self.args.model] = ref_results["psnr"][self.args.model]
            mssim_res[self.args.model] = ref_results["mssim"][self.args.model]

            plot_rate_distorsion(bpp_res, psnr_res, 
                                0, eest="compression_before_training", 
                                metric = 'PSNR',
                                save_fig=True,
                                file_name=os.path.join(self.ctx.img_dir, f"{self.args.nameRun}_psnr_minus_1.png"), #TODO change naming to avoid duplicate code
                                log_wandb=self.ctx.log_wandb)

            plot_rate_distorsion(bpp_res, 
                                mssim_res, 
                                0, 
                                eest="compression_mssim_before_training", 
                                metric = 'MS-SSIM',
                                save_fig=True,
                                file_name=os.path.join(self.ctx.img_dir, f"{self.args.nameRun}_mssim_minus_1.png"),
                                log_wandb=self.ctx.log_wandb, is_psnr=False)
            

            #Save data dictionnary
            results = {"psnr": psnr_res,"mssim": mssim_res,"bpp": bpp_res}
            file_path = os.path.join(self.ctx.res_dir, f"data_{self.args.nameRun}_minus_1.json")
            with open(file_path, "w") as f:
                json.dump(results, f)

    def make_plot(self,epoch):
            bpp_list = []
            psnr_list = []
            mssim_list = []

            ref_bpp_list = []
            ref_psnr_list = []
            ref_mssim_list = []         

            print("Make actual compression")
            self.ctx.net.update(force = True)

            if self.args.mask:
                for index in range(len(self.ctx.all_mask["g_a"])+1):# +1 to include no pruning case

                    if index < len(self.ctx.all_mask["g_a"]):#if index != len(self.ctx.all_mask["g_a"]): # Last index is no pruning 
                        
                        apply_saved_mask(self.ctx.net.g_a, self.ctx.all_mask["g_a"][index])
                        apply_saved_mask(self.ctx.net.g_s, self.ctx.all_mask["g_s"][index])

                        bpp_ac, psnr_ac, mssim_ac = compress_one_epoch(self.ctx.net, self.ctx.kodak_dataloader, self.ctx.device)

                        delete_mask(self.ctx.net.g_a,self.ctx.all_mask["g_a"][index])
                        delete_mask(self.ctx.net.g_s,self.ctx.all_mask["g_s"][index])

                    # No mask for 0.483 lambda 0.0 amount pruning   
                    else:
                        if not self.args.put_lambda_max: 
                            bpp_ac, psnr_ac, mssim_ac = compress_one_epoch(self.ctx.net, self.ctx.kodak_dataloader, self.ctx.device)
                    
                    bpp_list.append(bpp_ac)
                    psnr_list.append(psnr_ac)
                    mssim_list.append(mssim_ac)


                
                # ref_bpp_list.append(ref_results["bpp"][self.args.model])
                # ref_psnr_list.append(ref_results["psnr"][self.args.model])
                # ref_mssim_list.append(ref_results["mssim"][self.args.model])

                # for index in range(6):
                #     refnet = self._get_ref_model(quality=index+1).to(self.ctx.device)
                #     ref_bpp_ac, ref_psnr_ac, ref_mssim_ac = compress_one_epoch(refnet, self.ctx.kodak_dataloader, self.ctx.device)

                #     ref_bpp_list.append(ref_bpp_ac)
                #     ref_psnr_list.append(ref_psnr_ac)
                #     ref_mssim_list.append(ref_mssim_ac)
            
            elif self.args.pruningType=="adapter":
                for index in range(6): #0 = max pruning, 5 = no pruning

                    # Update index 
                    # net.set_index(index)
                    set_index_switch(self.ctx.net,index)

                    bpp_ac, psnr_ac, mssim_ac = compress_one_epoch(self.ctx.net, self.ctx.kodak_dataloader, self.ctx.device)

                    bpp_list.append(bpp_ac)
                    psnr_list.append(psnr_ac)
                    mssim_list.append(mssim_ac)

                    #Compute reference from cheng202_attn
                    # refnet = self._get_ref_model(quality=index+1).to(self.ctx.device)
                    # ref_bpp_ac, ref_psnr_ac, ref_mssim_ac = compress_one_epoch(refnet, self.ctx.kodak_dataloader, self.ctx.device)

                    # ref_bpp_list.append(ref_bpp_ac)
                    # ref_psnr_list.append(ref_psnr_ac)
                    # ref_mssim_list.append(ref_mssim_ac)

            psnr_res = {}
            mssim_res = {}
            bpp_res = {} 

            bpp_res["ours"] = bpp_list
            psnr_res["ours"] = psnr_list
            mssim_res["ours"] = mssim_list
            
            # Reference results from json
            with open("/home/ids/flauron-23/MagV/json/ref_results.json", "r") as f:
                ref_results = json.load(f)

            # bpp_res["cheng"] = ref_results["bpp"]["cheng"]
            # psnr_res["cheng"] = ref_results["psnr"]["cheng"]
            # mssim_res["cheng"] = ref_results["mssim"]["cheng"]

            bpp_res[self.args.model] = ref_results["bpp"][self.args.model]
            psnr_res[self.args.model] = ref_results["psnr"][self.args.model]
            mssim_res[self.args.model] = ref_results["mssim"][self.args.model]

            plot_rate_distorsion(bpp_res, psnr_res, 
                                epoch, eest="compression", 
                                metric = 'PSNR',
                                save_fig=True,
                                file_name=os.path.join(self.ctx.img_dir, f"{self.args.nameRun}_psnr_{epoch}.png"), #TODO change naming to avoid duplicate code
                                log_wandb=self.ctx.log_wandb)

            plot_rate_distorsion(bpp_res, 
                                mssim_res, 
                                epoch, 
                                eest="compression_mssim", 
                                metric = 'MS-SSIM',
                                save_fig=True,
                                file_name=os.path.join(self.ctx.img_dir, f"{self.args.nameRun}_mssim_{epoch}.png"),
                                log_wandb=self.ctx.log_wandb, is_psnr=False)
            

            if self.args.pruningType=="adapter":
                measure_switch_sparcity(self.ctx.net,epoch)
                measure_sparsity_induce_by_switch(self.ctx.net,epoch)


            #Save data dictionnary
            results = {"psnr": psnr_res,"mssim": mssim_res,"bpp": bpp_res}
            file_path = os.path.join(self.ctx.res_dir, f"data_{self.args.nameRun}_{epoch}.json")
            with open(file_path, "w") as f:
                json.dump(results, f)

    def train(self,epoch):
        print(f"Learning rate: {self.ctx.optimizer.param_groups[0]['lr']}")

        # Training 
        loss_tot_train, bpp_train, mse_train, aux_train_loss = train_one_epoch(
            self.ctx.net,
            self.ctx.criterion,
            self.ctx.train_dataloader,
            self.ctx.optimizer,
            self.ctx.aux_optimizer,
            epoch,
            self.args.clip_max_norm,
            self.args.put_lambda_max,
            self.args.lambda_max,
            args_mask=self.args.mask,
            all_mask=self.ctx.all_mask if self.args.mask  else None,
            lambda_list=self.ctx.lambda_list if self.args.mask else None,
            probs=None
        )

        if self.ctx.log_wandb:
            wandb.log({
                "train/epoch":epoch,
                "train/loss": loss_tot_train,
                "train/bpp_loss": bpp_train,
                "train/mse_loss": mse_train,
                "train/aux_loss":aux_train_loss
            },step = epoch)      

    def validate(self,epoch, dataloader,tag):
        total_losses=[]
        is_best_kodak = False
        is_best_val = False
        for i in range(len(self.ctx.lambda_list)):

            if self.args.mask :
                if i != len(self.ctx.lambda_list)-1:

                    apply_saved_mask(self.ctx.net.g_a,self.ctx.all_mask["g_a"][i])
                    apply_saved_mask(self.ctx.net.g_s,self.ctx.all_mask["g_s"][i])

                    loss_tot_val, bpp_loss_val, mse_loss_val, aux_loss_val, psnr_val, ssim_val = test_epoch(epoch, dataloader, self.ctx.net, self.ctx.criterion, tag)
                    
                    delete_mask(self.ctx.net.g_a,self.ctx.all_mask["g_a"][i])
                    delete_mask(self.ctx.net.g_s,self.ctx.all_mask["g_s"][i])
                else:
                    loss_tot_val, bpp_loss_val, mse_loss_val, aux_loss_val, psnr_val, ssim_val = test_epoch(epoch,dataloader, self.ctx.net, self.ctx.criterion, tag)

            # Adapter pruning type
            # else:
            #     set_index_switch(self.ctx.net,i)

                loss_tot_val, bpp_loss_val, mse_loss_val, aux_loss_val, psnr_val, ssim_val = test_epoch(epoch, dataloader, self.ctx.net, self.ctx.criterion, tag)

            if self.ctx.log_wandb:
                wandb.log({
                    f"{tag}/epoch":epoch,
                    f"{tag}_{i}/loss": loss_tot_val,
                    f"{tag}_{i}/bpp_loss": bpp_loss_val,
                    f"{tag}_{i}/mse_loss": mse_loss_val,
                    f"{tag}_{i}/aux_loss":aux_loss_val,
                    f"{tag}_{i}/psnr":psnr_val,
                    f"{tag}_{i}/mssim":ssim_val,
                },step = epoch)   

            total_losses.append(loss_tot_val)
        

        avg_loss_tot_val = sum(total_losses) / len(total_losses)

        if tag == 'val':
            avg_loss_tot_val = sum(total_losses) / len(total_losses)
            if isinstance(self.ctx.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.ctx.lr_scheduler.step(avg_loss_tot_val)
            else:
                self.ctx.lr_scheduler.step()
            
            # save checkpoint according to val_loss
            is_best_val = avg_loss_tot_val < self.ctx.best_val_loss
            self.ctx.best_val_loss = min(avg_loss_tot_val, self.ctx.best_val_loss)
        
        elif tag == 'kodak':
            # avg_loss_tot_kodak = sum(total_losses) / len(total_losses)
            is_best_kodak = avg_loss_tot_val < self.ctx.best_kodak_loss
            self.ctx.best_kodak_loss = min(avg_loss_tot_val, self.ctx.best_kodak_loss)
        else:
            # do nothing
            pass 

        if self.args.save and (is_best_kodak or is_best_val):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.ctx.net.state_dict(),
                    "best_val_loss": self.ctx.best_val_loss,
                    "best_kodak_loss":self.ctx.best_kodak_loss,
                    "optimizer": self.ctx.optimizer.state_dict(),
                    "aux_optimizer": self.ctx.aux_optimizer.state_dict() if self.ctx.aux_optimizer is not None else None,
                    "lr_scheduler": self.ctx.lr_scheduler.state_dict(),
                },
                True, # is best
                out_dir=self.ctx.model_dir,
                #filename=f"{str(args.lmbda).replace('0.','')}_checkpoint.pth.tar"
                filename=f"{self.args.nameRun}_checkpoint.pth.tar"
            )
        

    def _get_ref_model(self,quality):
        """
        get reference model with the provided quality
        """
        if self.args.model == "cheng":
            return cheng2020_attn(quality,pretrained = True)
        elif self.args.model == "msh":
            return mbt2018_mean(quality,pretrained = True)
        # elif self.args.model == "cheng":