# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# from custom_compress .models import SymmetricalTransFormer, WACNN
from custom_comp.models import SymmetricalTransFormer, WACNN,QVRFCheng,stfQVRF,GainedMSHyperprior,TCM

from .pretrained import load_pretrained as load_state_dict
from compressai.models.waseda import Cheng2020Attention
from compressai.models.google import MeanScaleHyperprior
from compressai.zoo import cheng2020_attn,mbt2018_mean
# from compressai.zoo.pretrained import load_pretrained

from torch import load,zeros_like


# def STF_loading(state_dict_path):
#     state_dict = load(state_dict_path)
#     #state_dict = load_state_dict(state_dict)
#     state_dict = state_dict['state_dict']

#     net = SymmetricalTransFormer()
#     net.load_state_dict(state_dict)

#     net.g_a = net.layers
#     net.g_s = net.syn_layers
    
#     return net

# def WACNN_loading(state_dict_path):

#     def clean_cnn_checkpoint(state_dict):
#         '''
#             All the key in the checkpoint have a prefix "module."
#             We need to remove it in order to match the keys of the model
#         '''
#         new_state_dict = {}
#         for k,v in state_dict.items():
#             new_key = k.replace("module.","")
#             new_state_dict[new_key] = v
#         return new_state_dict

#     state_dict = load(state_dict_path)
#     state_dict["state_dict"] = clean_cnn_checkpoint( state_dict["state_dict"])
#     state_dict = load_state_dict(state_dict["state_dict"])
#     state_dict = state_dict['state_dict']

#     net = WACNN()
#     net.load_state_dict(state_dict)
    
#     return net

# models = {
#     'stf': lambda:STF_loading("/home/ids/flauron-23/MagV/pretrained_models/stf_0483.pth.tar"),
#     'cnn': lambda:WACNN_loading("/home/ids/flauron-23/MagV/pretrained_models/cnn_025_best.pth.tar"),
#     'cheng': lambda:cheng2020_attn(quality=6,pretrained=True),
#     'chengBA2': Cheng2020Attention_BA2,
#     'msh': lambda:mbt2018_mean(quality=6,pretrained=True), # TODO same as cheng2020
# }


def rewire_g_a_s(model:SymmetricalTransFormer):
    model.g_a = model.layers
    model.g_s = model.syn_layers

def model_loading (model_class,state_dict_path,preprocess=None,postprocess=None,**kwargs):
    state_dict = load(state_dict_path)

    # Only deal with state_dictionnary
    if "state_dict" in state_dict.keys():
        state_dict = state_dict['state_dict']
  
    if preprocess is not None:
        state_dict=preprocess(state_dict,**kwargs)

    net = model_class() 
    net.load_state_dict(state_dict)

    if postprocess is not None: 
        postprocess(net,**kwargs) 
    
    return state_dict,net

def load_model_from_checkpoint(model_class,checkpoint_path):

    # Load the checkpoint
    state_dict = load(checkpoint_path, map_location='cpu')
    # print(state_dict["epoch"])
    #state_dict = load_state_dict(state_dict=state_dict["state_dict"])

    # Create the model
    net = model_class().to("cuda")

    if model_class == SymmetricalTransFormer:
        net.g_a = net.layers
        net.g_s = net.syn_layers


    net.load_state_dict(state_dict["state_dict"])

    # #update entropy model
    net.update(force=True)

    return net

def load_mask(mask_path):
    mask = load(mask_path, map_location='cpu')
    # print(type(mask["g_a"][0])) 
    # mask=mask.to("cuda")
    return mask

models = {
    'stf': lambda:model_loading(SymmetricalTransFormer,"/home/ids/flauron-23/MagV/pretrained_models/stf_0483.pth.tar",postprocess=rewire_g_a_s)[1],
    'cnn': lambda:model_loading(WACNN,"/home/ids/flauron-23/MagV/pretrained_models/cnn_025_best.pth.tar",load_state_dict)[1],
    'qvrf': lambda:model_loading(QVRFCheng,"/home/ids/flauron-23/QRAF/Cheng2020VR.pth.tar")[1],
    'stfqvrf':lambda:model_loading(stfQVRF,"/home/ids/flauron-23/STF-QVRF/STFVRImageNetSTE.pth.tar",postprocess=rewire_g_a_s)[1],
    'cheng': lambda:cheng2020_attn(quality=6,pretrained=True),
    'msh': lambda:mbt2018_mean(quality=6,pretrained=True),
    'gained': lambda:model_loading(GainedMSHyperprior,"/home/ids/flauron-23/MagV/pretrained_models/checkpoint_gainmshp_epoch90.pth")[1],
    'ours': lambda:model_loading(Cheng2020Attention,"/home/ids/flauron-23/MagV/data/magv_ablation_0.4/models/magv_ablation_0.4_checkpoint.pth.tar")[1],
    'tcm': lambda:model_loading(TCM,"/home/ids/flauron-23/MagV/pretrained_models/tcm_0.05.pth.tar")[1]
}