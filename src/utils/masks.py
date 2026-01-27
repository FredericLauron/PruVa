import torch
from torch.nn.utils import prune
from collections import defaultdict
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split

def delete_mask(model,parameters_to_prune):
    """ 
    Deletes the pruning mask from the model and preserves the pruned weights. 
    Arguments:
        model: The model from which to delete the pruning mask.
        parameters_to_prune: A list of tuples containing the modules and their parameters to prune.
    Raises:
        AssertionError: If no parameters to prune are provided.        
    """        
    assert parameters_to_prune is not None and len(parameters_to_prune) > 0, "No parameters to prune provided."

    #Look for any pruned module in the model and modify the mask
    # to preserve all the weiths of the module
    # and remove the pruning from the module
    for module, _ in parameters_to_prune:
        if hasattr(module, 'weight_orig'):
            with torch.no_grad():
                # create a mask full of ones to preserved the pruned weights
                module.weight_mask = torch.ones_like(module.weight_mask)
            prune.remove(module, 'weight')


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
    """
    Applies saved pruning masks to the model.
    Arguments:
        model: The model to which the masks will be applied.
        mask_dict: A dictionary containing the names of the modules and their corresponding pruning masks.  
    Raises:
        AssertionError: If the mask_dict is empty or None.
    """
    
    assert mask_dict is not None and len(mask_dict) > 0, "Mask dictionary is empty or None."

    for name, module in model.named_modules():
        if name in mask_dict:
            # Ensure the module has 'weight' to prune
            if hasattr(module, 'weight'):
                prune.custom_from_mask(module, name='weight', mask=mask_dict[name])

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

############################################################################################################################################
############################################################STRUCTURED PRUNING##############################################################
############################################################################################################################################

def compute_neuron_norm(parameters_to_prune):
    norms = {}
    nb_neuron = 0
    all_scores = []
    for module, _ in parameters_to_prune:
        W = module.weight.data

        if isinstance(module, torch.nn.Conv2d) or isinstance(module,torch.nn.ConvTranspose2d):
            # Norm per filter (output channel)

            # score of each filter/neuron in the weight matrix of the layer
            neuron_norms = (W.view(W.size(0), -1) ** 2).mean(dim=1)
            #L-2 normalization of the scores
            #neuron_norms = neuron_scores / (torch.norm(neuron_scores, p=2) + 1e-8)

            #neuron_norms = torch.norm(W.view(W.size(0), -1), p=2, dim=1)
            # Nb neuron per filter
            nb_neuron += W.shape[0]

  

        elif isinstance(module, torch.nn.Linear):
            # Norm per neuron (row of weight matrix)
            neuron_norms = (W ** 2).mean(dim=1)
            # Nb neuron 
            nb_neuron += W.shape[0]

        layer_l2 = torch.norm(neuron_norms, p=2)
        neuron_norms = neuron_norms / (layer_l2 + 1e-8)
        norms[module] = neuron_norms
        
    return norms


def get_global_list(norms_dict):
    global_list = []  # list of (module, neuron_index, norm_value)

    for module, neuron_norms in norms_dict.items():
        for idx, norm_value in enumerate(neuron_norms):
            global_list.append((module, idx, norm_value.item()))
    
    return global_list


def get_neuron_to_prune(global_list, amount):
    # global_list.sort(key=lambda x: x[2], reverse=True)
    # total_neurons = len(global_list)
    # num_to_prune = int(amount * total_neurons)

    # neurons_to_prune = global_list[-num_to_prune:]

    global_list.sort(key=lambda x: x[2])
    num_to_prune = int(len(global_list) * amount)
    neurons_to_prune = global_list[:num_to_prune]

    return neurons_to_prune

def group_by_module(neurons_to_prune):
    module_to_indices = defaultdict(list)
    for module, idx, _ in neurons_to_prune:
        module_to_indices[module].append(idx)
    return module_to_indices

def build_and_put_mask_on_module(module_to_indices):
    for module, indices in module_to_indices.items():

        # mask = torch.ones_like(module.weight.data)
        # mask[indices] = 0  # zero out selected neurons
        # prune.custom_from_mask(module, name="weight", mask=mask)
        
        # Number of neuron to prune / total number of neuron of the module
        amount = len(indices) / module.weight.data.size(0)
        #print(f"Pruning {amount:.2%} of neurons in module {module}")
        prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)


def global_structured(parameters_to_prune,amount):
    
    #Compute neurons norms et recover the list of all neurons :(module,neuron_norm),per module neuron_norm is a vector: [norm_neuron_1,norm_neuron_2,...,norm_neuron_n]
    #Structure of the list : (module, neuron_index, norm_value)
    global_list= get_global_list(compute_neuron_norm(parameters_to_prune))

    #Get the list of neurons to prune
    #Group them by module
    #Build the mask and put it on the module
    build_and_put_mask_on_module(group_by_module(get_neuron_to_prune(global_list, amount)))

    #for module, _ in parameters_to_prune:
        #prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)


def generate_mask_from_structured(model,amounts:list):
    """ 
    Generates pruning masks for the model based on the specified amounts.
    Arguments:
        model: The model for which to generate pruning masks.
        amounts: A list of amounts in % specifying the fraction of weights to prune.
        Returns:    
            out_all_mask: A list of dictionaries containing the pruning masks for each amount.
            parameters_to_prune: A list of tuples containing the modules and their parameters to prune.
    Raises:
        AssertionError: If the model is None or amounts is empty.
    """
    # checks
    assert model is not None
    assert amounts is not None and len(amounts) > 0

    #register all the parameters for the model that are available for pruning
    parameters_to_prune = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear, torch.nn.ConvTranspose2d], model.modules())]

    out_all_mask = []

    for index in amounts:

        if not np.isclose(index ,0.0):
            # generate the pruning masks
            #prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured,amount=index)
            global_structured(parameters_to_prune, index)
            
            # Save pruning masks
            out_all_mask.append(save_mask(model))

            #cleaning the pruning masks from the model
            delete_mask(model, parameters_to_prune)

    return out_all_mask ,parameters_to_prune

############################################################################################################################################
############################################################END STRUCTURED PRUNING##########################################################
############################################################################################################################################

def adjust_sampling_distribution(bpp,psnr,probs):
    
    #Compute the squared diff of bpp and psnr
    bpp_diff = np.abs((np.array(bpp["ours"],dtype=np.float64)-np.array(bpp["cheng2020"],dtype=np.float64)))
    psnr_diff = np.abs((np.array(psnr["ours"],dtype=np.float64)-np.array(psnr["cheng2020"],dtype=np.float64)))
    
    #Identify the indices where the diff is greater than a threshold
    #Currently set to 0.1, can be adjusted
    n = np.where(bpp_diff>0.1, 1.0, 0.0)
    m = np.where(psnr_diff>0.9, 1.0, 0.0)

    #Bitwise OR between the two masks    
    f=np.bitwise_or(n.astype(bool), m.astype(bool)).astype(float)

    # Update the probs
    #probs += f*0.1*bpp_diff*psnr_diff
    #probs += f * 0.1 * (0.5 * bpp_diff + 0.5 * psnr_diff)
    probs+=f*0.2
    
    # Clipping for security
    probs = np.clip(probs, 1e-6, 1.0)

    # Normalize the probs
    probs = probs / probs.sum()
    
    print("probs after normalization",probs)
    print("probs sum after normalization",probs.sum())

    return probs



############################################################################################################################################
############################################################STRUCTURED PRUNING FISHER#######################################################
############################################################################################################################################
def compute_fisher_information(parameters_to_prune,model,dataloader,criterion):

    #create a subset of the dataloader
    dataset = dataloader.dataset
    subset_dataset, _ = random_split(dataset, [1000, len(dataset) - 1000])
    subset_loader = DataLoader(subset_dataset, batch_size=16, shuffle=True,pin_memory=True)
    
    model.eval()
    # # Put all the parameters to require grads = False
    # for parameter in model.parameters(): parameter.requires_grad = False
    # # Put only relevant layers to require grads = True
    # for parameter,_ in parameters_to_prune: parameter.requires_grad = True

    device = next(model.parameters()).device

    fisher_info = {}
    # Forward pass
    for i, d in enumerate(subset_loader):
        
        d=d.to(device)

        # Zero gradients
        model.zero_grad()

        output = model(d)
        loss = criterion(output,d)

        # Compute gradients
        loss["loss"].backward()

        # Retrieve gradients
        # Retrieve layers of interest (module,"weight")
        for module,_ in parameters_to_prune:
            # Retrieve gradients (name, parameter (weight/bias))
            for name, p in module.named_parameters():

                if p.requires_grad and name == 'weight':
                    # Compute Fisher Information for the module/parameter
                    module_fisher_grad=(p.grad.detach().clone())**2 # shape (out_c,in_c,k,k) -> (out_c,k,k)
                    # Accumulate Fisher Information
                    if module not in fisher_info:
                        fisher_info[module] = torch.sum((module_fisher_grad + 1e-6),dim = (1,2,3)) # shape (out_c,k,k) -> (out_c)
                    else:
                        fisher_info[module] += torch.sum((module_fisher_grad + 1e-6),dim = (1,2,3)) # shape (out_c,k,k) -> (out_c)
        
    # Compute average Fisher Information over all samples
    fisher_info = {k: (v/len(subset_dataset)) for k, v in fisher_info.items()}

    return fisher_info

def global_structured_fisher(parameters_to_prune,amount,model,dataloader,criterion):
    
    #Compute neurons norms et recover the list of all neurons :(module,neuron_norm),per module neuron_norm is a vector: [norm_neuron_1,norm_neuron_2,...,norm_neuron_n]
    #Structure of the list : (module, neuron_index, norm_value)
    global_list= get_global_list(compute_fisher_information(parameters_to_prune,model,dataloader,criterion))

    #Get the list of neurons to prune
    #Group them by module
    #Build the mask and put it on the module
    build_and_put_mask_on_module(group_by_module(get_neuron_to_prune(global_list, amount)))

    #for module, _ in parameters_to_prune:
        #prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)

def generate_mask_from_structured_fisher(model,amounts:list,dataloader,criterion,submodel):
    """ 
    Generates pruning masks for the model based on the specified amounts.
    Arguments:
        model: The model for which to generate pruning masks.
        amounts: A list of amounts in % specifying the fraction of weights to prune.
        dataloader: The dataloader to use for computing Fisher information.
        criterion: The loss function to use for computing Fisher information.
        submodel: The submodel to which the masks will be applied.
        Returns:    
            out_all_mask: A list of dictionaries containing the pruning masks for each amount.
            parameters_to_prune: A list of tuples containing the modules and their parameters to prune.
    Raises:
        AssertionError: If the model is None or amounts is empty.
    """
    # checks
    assert model is not None
    assert amounts is not None and len(amounts) > 0

    #register all the parameters for the model that are available for pruning
    parameters_to_prune = [(module, "weight") for module in filter(lambda m: type(m) in [torch.nn.Conv2d, torch.nn.Linear], submodel.modules())]

    out_all_mask = []

    for index in amounts:

        # generate the pruning masks
        #prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured,amount=index)
        #global_structured_fisher(parameters_to_prune, index)
        global_structured_fisher(parameters_to_prune,index,model,dataloader,criterion)
        # Save pruning masks
        out_all_mask.append(save_mask(submodel))

        #cleaning the pruning masks from the model
        delete_mask(submodel, parameters_to_prune)

    return out_all_mask ,parameters_to_prune
############################################################################################################################################
############################################################END STRUCTURED PRUNING FISHER###################################################
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