import torch.nn as nn
import torch.optim as optim

import torch
import shutil

from .engine import test_epoch,train_one_epoch, compress_one_epoch, AverageMeter,pad,crop
from .loss import RateDistortionLoss
from .dataset import TestKodakDataset
from .functions import compute_metrics, compute_msssim, compute_psnr
from .masks import delete_mask, save_mask, generate_mask_from_unstructured,apply_saved_mask,lambda_percentage,count_zeros_ones

#from .chengBA2_old import get_Cheng2020Attention_with_conv_switch, set_cheng2020Attention_index,frozen_cheng2020Attention

import random
import os
import numpy as np

import torch.nn.functional as F





def configure_optimizers(net, args, assert_intersection = True, opt = 'adam'):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())

    if assert_intersection:
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0


    params = [params_dict[n] for n in sorted(parameters)]
    if len(params) > 0:
        print('Creating Optimizer for main network')
        if opt == 'adam':
            print('Adam')
            optimizer = optim.Adam(
                (params_dict[n] for n in sorted(parameters)),
                lr=args.learning_rate,
            )
        else:
            print('SGD')
            optimizer = optim.SGD(
                (params_dict[n] for n in sorted(parameters)),
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=0.0
            )
    else:
        optimizer = None

    aux_params = [params_dict[n] for n in sorted(aux_parameters)]
    if len(aux_params) > 0:
        print('Creating Optimizer for aux network')
        if opt == 'adam':
            print('Adam')
            aux_optimizer = optim.Adam(
                (params_dict[n] for n in sorted(aux_parameters)),
                lr=args.aux_learning_rate,
            )
        else:
            print('SGD')
            aux_optimizer = optim.SGD(
                (params_dict[n] for n in sorted(aux_parameters)),
                lr=args.aux_learning_rate,
                momentum=0.9,
                weight_decay=0.0
            )
    else:
        aux_optimizer = None
    return optimizer, aux_optimizer


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def save_checkpoint(state, is_best, out_dir, filename='last_checkpoint.pth.tar'):
    
    torch.save(state, f'{out_dir}/{filename}')
    if is_best:
        name_best = filename.replace('.pth.tar','') + '_best.pth.tar'
        shutil.copyfile(f'{out_dir}/{filename}', f'{out_dir}/{name_best}')


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)