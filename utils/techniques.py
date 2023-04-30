import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.utils.prune as prune
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from models import *
from utils import *

class LASSO(prune.BasePruningMethod):
    """Base pruning method from pytorch designed to compute the mask to prune weights with an absolute value below a given threshold;
        in this case used to prune the arbitrarily small weights after training with LASSO-loss,
        this method was not used during the final experiment runs but rather tested different versions of LASSO-pruning (prune below threshold vs prune till target sparsity is reached)

    Parameters
    ----------
    threshold = float
        threshold below which weights are pruned

    Returns
    -------
    some layer = nn.Module
        the given layer with weight tensor t after applying the computed mask
    """

    PRUNING_TYPE = 'unstructured'

    def __init__(self, threshold):
        self.threshold = threshold
        
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        t = t.cpu().detach().numpy()
        mask = torch.from_numpy(np.where(abs(t) <= self.threshold, 0, 1))
        return mask.cuda()

def prune_resnet(sparsity = 0.8, module_dict = {}, method = '', param = 'weight', LASSO_threshold = 0.0):
    """Prune all layers included in module_dict dictionary
    
    Parameters
    ----------
    resnet = ResNet
        given resnet model instance

    sparsity = float or list
        if float value between 0.0 and 1.0 giving the percentage of sparsity to reach in all layers
        if list provides different pruning ratios for each layer
    
    module_list = list
        list of all layers to prune

    method = str {'LASSO', 'gmp', 'random', 'greedy_layer'}
        Pruning method to use

    to_prune = str {'weight', 'bias'} (only weight was used for my experiment)
        which parameter to be pruned
    """
    if method == "gmp":
        parameters_to_prune = ()

        for layer in module_dict:
            parameters_to_prune += ((module_dict[layer], param),)
        # apply L1Unstructured pruning globally
        prune.global_unstructured(parameters_to_prune, prune.L1Unstructured, amount = sparsity)

    elif method == "random":
        parameters_to_prune = ()

        for layer in module_dict:
            parameters_to_prune += ((module_dict[layer], param),)
        # apply random pruning globally
        prune.global_unstructured(parameters_to_prune, prune.RandomUnstructured, amount = sparsity)

    elif method == "greedy_layer":
        for layer in module_dict:
            prune.l1_unstructured(module_dict[layer], param, sparsity)

    elif method == "LASSO":
        parameters_to_prune = ()

        for layer in module_dict:
            parameters_to_prune += ((module_dict[layer], param),)
        prune.global_unstructured(parameters_to_prune, prune.L1Unstructured, amount = sparsity)

def L1_reg(model, LAMBDA_L1 = 0.001):
    """Calculate LASSO value for given model

    Parameters
    ----------
    model = nn.Module
        given model to calculate LASSO value for

    LAMBDA_L1 = float
        value between 0.0 and 1.0 defining how much the LASSO value is weighted in the LASSO loss function
        0.0 meaning the value doesn't influende the loss at all and 1.0 the LASSO value is used completely
    """
    weights = torch.cat([x.view(-1) for x in model.parameters()])
    L1_reg  = LAMBDA_L1 * torch.norm(weights, 1)
    return L1_reg

def create_ticket(trained_model, initial_model_dict, resnet20 = True):
    """Creates the lottery ticket from a trained and pruned model

    Parameters
    ----------
    trained_model = nn.Module
        trained and pruned model to use as the "mask" for the ticket

    initial_state_dict = state dict
        state dict with weight values from initialization

    model_size = nn.Module
        variable holding the size of all the models, (resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202)

    Returns
    -------
    ticket = nn.Module
        computed ticket with weight values of initialisation but with the same weights removed as the trained and pruned model
    """
    print(f"\nCreating ticket...\n")
    if resnet20:
        ticket = resnet.resnet20()
    else:
        ticket = resnet.resnet56()
    ticket.load_state_dict(initial_model_dict)
    pruned_layers, _ = helper.get_layers(trained_model)
    unpruned_layers, _ = helper.get_layers(ticket)

    for pruned, unpruned in zip(pruned_layers, unpruned_layers):
        prune.custom_from_mask(unpruned_layers[unpruned].cpu(), "weight", pruned_layers[pruned].weight_mask.cpu())
    return ticket