import numpy as np
import torch
from torch.nn.utils import prune
import torch_pruning as tp

from collections import defaultdict

#from torch.utils.data import DataLoader, Subset, random_split

from custom_comp.models import SymmetricalTransFormerEncoder,SymmetricalTransFormerDecoder,STFHypAndDecode,STFEncodeAndHyp
from custom_comp.layers import PatchMerging,PatchSplit
from custom_comp.layers.win_attention import WindowAttention

from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

# def delete_mask(model,parameters_to_prune):
#     """ 
#     Deletes the pruning mask from the model and preserves the pruned weights. 
#     Arguments:
#         model: The model from which to delete the pruning mask.
#         parameters_to_prune: A list of tuples containing the modules and their parameters to prune.
#     Raises:
#         AssertionError: If no parameters to prune are provided.        
#     """        
#     assert parameters_to_prune is not None and len(parameters_to_prune) > 0, "No parameters to prune provided."

#     #Look for any pruned module in the model and modify the mask
#     # to preserve all the weiths of the module
#     # and remove the pruning from the module
#     for module, _ in parameters_to_prune:
#         if hasattr(module, 'weight_orig'):
#             with torch.no_grad():
#                 # create a mask full of ones to preserved the pruned weights
#                 module.weight_mask = torch.ones_like(module.weight_mask)
#             prune.remove(module, 'weight')

def delete_mask(model, mask_dict):
    """ 
    Deletes the pruning masks from the model and restores the original unpruned weights. 
    Arguments:
        model: The model from which to delete the pruning mask.
        mask_dict: A dictionary containing the names of the modules and their masks.
    Raises:
        AssertionError: If no mask_dict is provided.
    """
    assert mask_dict is not None and len(mask_dict) > 0, "Mask dictionary is empty or None."
    for name, module in model.named_modules():
        if name in mask_dict:
            with torch.no_grad():
                # 1. Ripristino dei pesi (weight)
                if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
                    # Riempiamo la maschera di 1 per recuperare i pesi "nascosti"
                    module.weight_mask.fill_(1.0)
                    prune.remove(module, 'weight')
                
                # 2. Ripristino dei bias
                if hasattr(module, 'bias_orig') and hasattr(module, 'bias_mask'):
                    # Riempiamo la maschera di 1 per recuperare i bias "nascosti"
                    module.bias_mask.fill_(1.0)
                    prune.remove(module, 'bias')


def save_mask(model):
    """ 
    Saves the pruning masks from the model into a dictionary. 
    Arguments:
        model: The model from which to save the pruning masks.
    Returns:
        mask_dict: A dictionary containing the names of the modules and their corresponding pruning masks.
    Raises:
        ValueError: If no masks are found in the model.
    """
    mask_dict = {}

    #Look for any pruned module in the model and save the mask into mask_dict
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
            mask_dict[name] = module.weight_mask.detach().clone()

    # if mask_dict is empty, raise an error
    if not mask_dict:
        raise ValueError("No masks found in the model. Ensure that pruning has been applied.")
    
    return mask_dict

def apply_saved_mask(model, mask_dict):
    assert mask_dict is not None and len(mask_dict) > 0, "Mask dictionary is empty or None."

    for name, module in model.named_modules():
        if name in mask_dict:
            layer_masks = mask_dict[name]
            
            if 'weight' in layer_masks and hasattr(module, 'weight') and module.weight is not None:
                prune.custom_from_mask(module, name='weight', mask=layer_masks['weight'])
                
            if 'bias' in layer_masks and hasattr(module, 'bias') and module.bias is not None:
                prune.custom_from_mask(module, name='bias', mask=layer_masks['bias'])

############################################################################################################################################
############################################################UNSTRUCTURED PRUNING############################################################
############################################################################################################################################
def generate_mask_from_unstructured(model,amounts:list):
    """ 
    Generates pruning masks for the model based on the specified amounts.
    Arguments:
        model: The model for which to generate pruning masks.
        amounts: A list of amounts in % specifying the fraction of weights to prune.
        Returns:    
            out_all_mask: A list def adjust_sampling_distribution(bpp,psnr,probs,):
    
    #Compute the squared diff of bpp and psnr
    bpp_diff = (np.array(bpp["ours"],dtype=np.float64)-np.array(bpp["cheng2020"],dtype=np.float64))**2
    psnr_diff = (np.array(psnr["ours"],dtype=np.float64)-np.array(psnr["cheng2020"],dtype=np.float64))**2
    
    #Identify the indices where the diff is greater than a threshold
    #Currently set to 0.1, can be adjusted
    n = np.where(bpp_diff>0.1, 1.0, 0.0)
    m = np.where(psnr_diff>0.1, 1.0, 0.0)

    #Bitwise OR between the two masks    
    f=np.bitwise_or(n.astype(bool), m.astype(bool)).astype(float)

    # Update the probs
    probs += f*0.1*bpp_diff*psnr_diff
    #probs += f * 0.1 * (0.5 * bpp_diff + 0.5 * psnr_diff)
    
    # Normalize the probs
    probs = probs / probs.sum()
    
    print("probs after normalization",probs)
    print("probs sum after normalization",probs.sum())

    return probsof dictionaries containing the pruning masks for each amount.
            parameters_to_prune: A list of tuples containing the modules and their parameters to prune.
    Raises:
        AssertionError: If the model is None or amounts is empty.
    """
    # checks
    assert model is not None
    assert amounts is not None and len(amounts) > 0

    #register all the parameters for the model that are available for pruning
    parameters_to_prune = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear,torch.nn.ConvTranspose2d], model.modules())]
    #parameters_to_prune = [(module, "weight") for name, module in model.named_modules()  if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)) and not name.endswith("attn.qkv")]
    out_all_mask = []

    for index in amounts:

        if index > 0.0: #if not np.isclose(index ,0.0): 
            # generate the pruning masks
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured,amount=index)
            
            # Save pruning masks
            out_all_mask.append(save_mask(model))

            #cleaning the pruning masks from the model
            delete_mask(model, parameters_to_prune)

    return out_all_mask ,parameters_to_prune

############################################################################################################################################
############################################################END UNSTRUCTURED PRUNING########################################################
############################################################################################################################################

def lambda_percentage(alpha,amount,lambda_max=0.0483,lambda_min=0.0018):
    """
    Computes the percentage mapping based on the exponential mapping of lambda values.
    The number of points is determined by the length of the input alpha list.
    Arguments:
        alpha: [list] The input linearly evenly spaced alpha values.
        amount: [float] The max amount for pruning. (ex: 0.6 for 60%)
    Returns:
        lambda_values: The computed lambda values.
        percentage: The computed percentage values."""
    lambda_max = lambda_max
    lambda_min = lambda_min
    # If min pruning is not 0.0
    # if isinstance(alpha,float): #single float
    #     if alpha>0.0:
    #         lambda_max = np.exp(np.log(lambda_max) * (1 - alpha / amount) + np.log(lambda_min) * (alpha / amount))

    # else:
    #     if alpha[-1]>0.0: # list of float
    #         lambda_max = np.exp(np.log(lambda_max) * (1 - alpha[-1] / amount) + np.log(lambda_min) * (alpha[-1] / amount))

    
    lambda_values = np.exp(np.log(lambda_max) * (1 - alpha / amount) + np.log(lambda_min) * (alpha / amount))

    # return lambda_values,amount * (lambda_max - lambda_values) / (lambda_max - lambda_min)
    return lambda_values, amount * (np.log(lambda_max) - np.log(lambda_values)) / (np.log(lambda_max) - np.log(lambda_min))

def check_neuron_sparsity(parameters_to_prune):
    """
    Check neuron sparsity in all pruned layers of the model (g_a or g_s)
    print it and return it
    """
    total_neurons = 0
    zeroed_neurons = 0
    for module, _ in parameters_to_prune:
        W = module.weight.data
        if isinstance(module, torch.nn.Conv2d) or isinstance(module,torch.nn.ConvTranspose2d):
            for i in range(W.size(0)):  # output channels
                total_neurons += 1
                if torch.allclose(W[i], torch.zeros_like(W[i])):
                #if torch.all(W[i] == 0):
                    zeroed_neurons += 1
        elif isinstance(module, torch.nn.Linear):
            for i in range(W.size(0)):  # rows
                total_neurons += 1
                #if torch.all(W[i] == 0):
                if torch.allclose(W[i], torch.zeros_like(W[i])):
                    zeroed_neurons += 1
    
    print(f"Neuron sparsity: {zeroed_neurons/total_neurons:.2%}")
    return torch.tensor([zeroed_neurons/total_neurons],dtype=torch.float64)

def print_layer(model,layer_type):
    [print(m.weight) for _, m in model.named_modules() if isinstance(m, layer_type)]

def make_pruning_permanent(model):
    for name, module in model.named_modules():
        # Rimuove l'hook di pruning dai pesi
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')
        # Rimuove l'hook di pruning dai bias
        if hasattr(module, 'bias_mask'):
            prune.remove(module, 'bias')

def counting_zeros_params(model):
    params = sum(p.numel() for p in model.parameters()) 

    pruned_params = 0.0
    modules = list(filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear], model.modules()))
    for module in modules:
        pruned_params += float(torch.sum(module.weight == 0))
    return pruned_params, params

############################################################################################################################################
##############################################################STRUCTURED PRUNING############################################################
############################################################################################################################################
class RateDistortionLoss(torch.nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, reduction='mean'):
        super().__init__()
        self.lmbda = lmbda
        self.reduction = reduction

    def forward(self, output, target):
        N, _, H, W = target.size() # N è il batch_size
        out = {}
        
        # Pixel totali per SINGOLA immagine
        num_pixels_per_sample = H * W

        # 1. Calcolo BPP Loss per ogni elemento del batch (Shape: [N])
        bpp_loss_per_sample = sum(
            (torch.log(likelihoods).view(N, -1).sum(dim=1) / (-math.log(2) * num_pixels_per_sample))
            for likelihoods in output["likelihoods"].values()
        )

        # 2. Calcolo MSE Loss per ogni elemento del batch (Shape:[N])
        # F.mse_loss con reduction='none' restituisce[N, C, H, W]
        mse_loss_per_sample = F.mse_loss(output["x_hat"], target, reduction='none')
        # Facciamo la media su canali, altezza e larghezza per avere un valore per immagine
        mse_loss_per_sample = mse_loss_per_sample.view(N, -1).mean(dim=1) 

        # 3. Loss Totale per ogni elemento del batch (Shape: [N])
        loss_per_sample = self.lmbda * 255 ** 2 * mse_loss_per_sample + bpp_loss_per_sample

        # 4. Riduciamo in base al parametro
        if self.reduction == 'mean':
            out["bpp_loss"] = bpp_loss_per_sample.mean()
            out["mse_loss"] = mse_loss_per_sample.mean()
            out["loss"] = loss_per_sample.mean() # Ritorna uno scalare
        elif self.reduction == 'none':
            out["bpp_loss"] = bpp_loss_per_sample
            out["mse_loss"] = mse_loss_per_sample
            out["loss"] = loss_per_sample        # Ritorna un vettore [N]
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

        return out

def load_pretrained_state_dict(model, checkpoint):

    # state_dict_full_model = full_model.state_dict()
    state_dict_full_model = torch.load(checkpoint)
    if "state_dict" in state_dict_full_model.keys():
        state_dict_full_model = state_dict_full_model['state_dict']


    state_dict_model = model.state_dict()

    filtered_dict = {
        k: v for k, v in state_dict_full_model.items()
        if k in state_dict_model and v.shape == state_dict_model[k].shape
    }

    state_dict_model.update(filtered_dict)
    model.load_state_dict(state_dict_model, strict=True)

def get_models(encoder, device):
    checkpoint = '/home/ids/gspadaro/repos/MagV/src/draft/checkpoints/stf_0483_best.pth.tar'

    if encoder:
        model = SymmetricalTransFormerEncoder().to(device).eval()
        example_inputs = torch.rand((1,3,256,256)).to(device)
        complementary_model = STFHypAndDecode().to(device).eval()

    else:
        model = SymmetricalTransFormerDecoder().to(device).eval()
        example_inputs = torch.rand((1,384,16,16)).to(device)

        complementary_model = STFEncodeAndHyp().to(device).eval()

    load_pretrained_state_dict(model = model, checkpoint = checkpoint)
    load_pretrained_state_dict(model = complementary_model, checkpoint = checkpoint)

    return model, complementary_model, example_inputs

def get_pruner_elements(encoder, model):
    ignored_layers = []
    if not encoder:
        ignored_layers.append(model.end_conv)
    unwrapped_parameters =[]
    num_heads = {}
    for m in model.modules():
        if isinstance(m, WindowAttention):
            num_heads[m.qkv] = m.num_heads
            unwrapped_parameters.append((m.relative_position_bias_table, 1))

        # if isinstance(m, PatchMerging):
        #     ignored_layers.append(m)
        # if isinstance(m, Mlp):
        #     ignored_layers.append(m.fc2)


        if isinstance(m, PatchMerging) or isinstance(m, PatchSplit):
            if hasattr(m, 'reduction'):
                ignored_layers.append(m.reduction)
            if hasattr(m, 'norm'):
                ignored_layers.append(m.norm)

    return ignored_layers, unwrapped_parameters, num_heads

def generate_pruning_mask_from_pruner(model, pruner):

    # 1. Creiamo una mappa inversa per trovare il nome stringa di ogni modulo
    module_to_name = {module: name for name, module in model.named_modules()}
        
    # 2. init mask
    mask_dict = {}
    for name, module in model.named_modules():
        has_w = hasattr(module, 'weight') and module.weight is not None
        has_b = hasattr(module, 'bias') and module.bias is not None
        
        if has_w or has_b:
            mask_dict[name] = {}
            if has_w:
                mask_dict[name]['weight'] = torch.ones_like(module.weight.data)
            if has_b:
                mask_dict[name]['bias'] = torch.ones_like(module.bias.data)
        
    # 3. pruner
    for i, group in enumerate(pruner.step(interactive=True)):
        for dep, idxs in group:
            target_layer = dep.target.module
            
            # Se il modulo non fa parte dei named_modules, ignoriamolo
            if target_layer not in module_to_name:
                continue
            
            layer_name = module_to_name[target_layer]
            layer_masks = mask_dict[layer_name]
            pruning_fn_name = dep.handler.__name__ # es: 'prune_conv_out_channels'
            
            # --- Modifica In-Channels (Dimensione 1) ---
            if 'in_channels' in pruning_fn_name:
                layer_masks['weight'][:, idxs] = 0
                # target_layer.weight.data[:, idxs] = 0 # Azzeriamo per far procedere il pruner
                
            # --- Modifica Out-Channels (Dimensione 0) ---
            elif 'out_channels' in pruning_fn_name:
                if 'weight' in layer_masks:
                    layer_masks['weight'][idxs] = 0
                    # target_layer.weight.data[idxs] = 0
                    
                if 'bias' in layer_masks and hasattr(target_layer, 'bias') and target_layer.bias is not None:
                    layer_masks['bias'][idxs] = 0
                    # target_layer.bias.data[idxs] = 0
    
    return mask_dict

def convert_soft_to_hard_pruning(model, example_inputs, ignored_layers=None, unwrapped_parameters=None):
    """
    Converte un modello con maschere di pruning (soft) in un modello fisicamente ridotto (hard).
    """
    # 1. Rendi permanenti gli zeri (rimuovi gli hook di soft pruning di PyTorch)
    for name, module in model.named_modules():
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')
        if hasattr(module, 'bias_mask'):
            prune.remove(module, 'bias')

    # 2. Inizializza il Dependency Graph per calcolare le connessioni topologiche
    DG = tp.DependencyGraph().build_dependency(
        model, 
        example_inputs=example_inputs,
        unwrapped_parameters=unwrapped_parameters
    )

    # Raccogliamo i moduli prunabili
    prunable_modules =[m for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    if ignored_layers:
        prunable_modules =[m for m in prunable_modules if m not in ignored_layers]

    # Teniamo traccia dei moduli già processati.
    # Quando pruniamo un layer, torch_pruning riduce automaticamente anche i layer 
    # a esso collegati. Questo set ci evita di cercare di prunarli due volte.
    processed_modules = set()

    print("Conversione da Soft a Hard Pruning in corso...")
    
    for module in prunable_modules:
        if module in processed_modules:
            continue
            
        # Calcoliamo la "magnitudo" (somma assoluta) per ogni canale di output
        weight = module.weight.data
        dims_to_sum = tuple(range(1, weight.dim())) # Somma su tutto tranne la dim 0 (out_channels)
        channel_magnitudes = weight.abs().sum(dim=dims_to_sum)
        
        # Troviamo gli indici dei canali di output che sono ESATTAMENTE zero
        # (Usiamo < 1e-8 per sicurezza contro imprecisioni di floating point)
        zero_idxs = torch.where(channel_magnitudes < 1e-8)[0].tolist()
        
        if len(zero_idxs) > 0:
            # Scegliamo la funzione handler corretta
            prune_fn = tp.prune_conv_out_channels if isinstance(module, torch.nn.Conv2d) else tp.prune_linear_out_channels
            
            # Creiamo un gruppo di pruning strutturato basato su questi zeri
            group = DG.get_pruning_group(module, prune_fn, idxs=zero_idxs)
            
            if DG.check_pruning_group(group):
                group.prune()
                
                # Registriamo tutti i moduli che sono stati modificati da questo gruppo
                for dep, _ in group:
                    processed_modules.add(dep.target.module)

    print("Hard Pruning completato! Il modello è ora fisicamente compresso.")

def prune_encoder_or_decoder(perc = 0.2, 
                             encoder = True, 
                             train_loader = None):

    device = 'cuda'

    model, complementary_model, example_inputs = get_models(encoder, device)
    
    for param in complementary_model.parameters():
        param.requires_grad = False

    imp = tp.importance.GroupMagnitudeImportance(p=2)
    # imp = tp.importance.GroupTaylorImportance()
    
    # imp = tp.importance.GroupHessianImportance()
    
    ignored_layers, unwrapped_parameters, num_heads = get_pruner_elements(encoder, model)
        
    pruner = tp.pruner.BasePruner(
        model, 
        example_inputs, 
        isomorphic=True, # enable isomorphic pruning to improve global ranking
        global_pruning=True, # If False, a uniform pruning ratio will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        pruning_ratio=perc, # target pruning ratio
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters, 
        num_heads=num_heads, # number of heads in self attention
        prune_num_heads=False, # reduce num_heads by pruning entire heads (default: False)
        prune_head_dims=True, # reduce head_dim by pruning featrues dims of each head (default: True)
        head_pruning_ratio=0.5, #args.head_pruning_ratio, # remove 50% heads, only works when prune_num_heads=True (default: 0.0)
        round_to=1
    )

    if isinstance(imp, (tp.importance.GroupTaylorImportance, tp.importance.GroupHessianImportance)):
        taylor_batchs = 10
        
        # criterion = RateDistortionLoss(0.0018)

        model.zero_grad()
        complementary_model.zero_grad()

        if isinstance(imp, tp.importance.GroupHessianImportance):
            criterion = RateDistortionLoss(0.0018, reduction='none')
            imp.zero_grad()
        else:
            criterion = RateDistortionLoss(0.0018, reduction='mean')

        print("Accumulating gradients for pruning...")
        for k, (imgs, lbls) in enumerate(train_loader):
            if k>=taylor_batchs: break
            imgs = imgs.to(device)
            # lbls = lbls.to(device)

            if isinstance(model, SymmetricalTransFormerEncoder):
                assert isinstance(complementary_model, STFHypAndDecode)
                y = model(imgs)
                out = complementary_model(y)  

            elif isinstance(model, SymmetricalTransFormerDecoder):
                assert isinstance(complementary_model, STFEncodeAndHyp)
                y_hat, y_likelihoods, z_likelihoods = complementary_model(imgs)
                x_hat = model(y_hat)

                out = {
                    "x_hat": x_hat,
                    "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                }

            out_criterion = criterion(out, imgs)
            loss = out_criterion["loss"]

            if isinstance(imp, tp.importance.GroupHessianImportance):
                # loss = torch.nn.functional.cross_entropy(output, lbls, reduction='none')
                for l in loss:
                    model.zero_grad()
                    l.backward(retain_graph=True)
                    imp.accumulate_grad(model)
            elif isinstance(imp, tp.importance.GroupTaylorImportance):
                # loss = torch.nn.functional.cross_entropy(output, lbls)
                loss.backward()

    
    mask_dict = generate_pruning_mask_from_pruner(model=model, pruner=pruner)
    return mask_dict




def generate_mask_from_structured(model,amounts:list):
    
    ##WORKS ONLY FOR THE STF MODEL!!!###
    # checks
    assert model is not None
    assert amounts is not None and len(amounts) > 0

    transform = transforms.Compose([transforms.ToTensor()])

    fake_dataset = FakeData(size=1000, image_size=(3, 256, 256), num_classes=10, transform=transform)
    train_loader = DataLoader(fake_dataset, batch_size=2, shuffle=True)

    out_all_mask = {
        'g_a': [],
        'g_s': []
    }
    print("amounts =", amounts)
    for index in amounts:
        if index > 0.0:
            
            #decoder
            enc_mask = prune_encoder_or_decoder(perc = index, encoder=True, train_loader = train_loader)
            enc_mask = {k.replace("layers.", ""): v for k, v in enc_mask.items()}
            #encoder
            dec_mask = prune_encoder_or_decoder(perc = index, encoder=False, train_loader = train_loader)
            dec_mask = {k.replace("syn_layers.", ""): v for k, v in dec_mask.items()}
            
            out_all_mask['g_a'].append(enc_mask)
            out_all_mask['g_s'].append(dec_mask)
    
    print("mask",len(out_all_mask['g_a']))
    print("mask",len(out_all_mask['g_s']))
    return out_all_mask

############################################################################################################################################
##############################################################END STRUCTURED PRUNING########################################################
############################################################################################################################################

def count_zeros_ones(mask: dict):
        zeros = 0
        ones = 0
        for m in mask.values():
            # assicurati che sia su CPU per .item() (opzionale)
            m = m.detach()

            ones += torch.sum(m == 1).item()
            zeros += torch.sum(m == 0).item()

        return zeros/ (zeros+ones)