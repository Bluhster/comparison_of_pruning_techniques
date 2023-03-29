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
#import torchvision.datasets as datasets
    
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, lam = 0.5):
        #target = torch.LongTensor(target)
        print(target)
        for q in output:
            print(q)
        mse = nn.MSELoss(reduction = 'sum')
        #mse = nn.CrossEntropyLoss()
        l1 = nn.L1Loss()
        mse_loss = mse(output, target)
        l1_loss = l1(output, target)

        lasso = mse_loss + (lam * l1_loss)
        return lasso

def prune_resnet(sparsity = 0.8, module_dict = {}, method = '', param = 'weight', LASSO_threshold = 0.0):
    """Prune all convolutional layers in given resnet
    
    Parameters
    ----------
    resnet = ResNet
        given resnet model instance

    sparsity = float or list
        if float value between 0.0 and 1.0 giving the percentage of sparsity to reach in all layers
        if list provides different pruning ratios for each layer
    
    module_list = list
        list of all layers to prune

    method = str {'LASSO', 'OBD'}
        Pruning method to use

    to_prune = str {'weight', 'bias'}
        which parameter to be pruned
    """
    if isinstance(sparsity, float):
        sparsities = [sparsity] * len(module_dict)
    elif isinstance(sparsity, list):
        if len(sparsity) != len(module_dict):
            raise ValueError("Incompatible number of sparsities provided for layers")
        sparsities = sparsity
    else:
        raise TypeError

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
        threshold_weights(module_dict, LASSO_threshold)


def threshold_weights(layer_dict, threshold):
    """Set all weights with an absolute value lower than the defined threshold to 0

    Parameters
    ----------
    layer_dict = dict
        dictionary with all layers in which to change the weights

    threshold = float
        float value to cut 
    """
    for layer in layer_dict:
        weights = layer_dict[layer].weight.cpu().detach().numpy()
        #print(f"before thresholding : {weights}")
        weights = torch.from_numpy(np.where(abs(weights) <= threshold, 0, weights))
        #print(f"aaaaaaaaaaaaaaaafter thresholding : {weights}")
        layer_dict[layer].weight = nn.parameter.Parameter(weights)


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

def create_ticket(trained_model, initial_state_dict):
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
    ticket = resnet.resnet20()
    ticket.load_state_dict(initial_state_dict)
    pruned_layers, _ = helper.get_layers(trained_model)
    unpruned_layers, _ = helper.get_layers(ticket)

    for pruned, unpruned in zip(pruned_layers, unpruned_layers):
        pruned_weights = pruned_layers[pruned].weight.cpu().detach().numpy()
        unpruned_weights = unpruned_layers[unpruned].weight.cpu().detach().numpy()
                                                                            # *10 necessary or not?
        unpruned_weights = torch.from_numpy(np.where(pruned_weights == 0, 0, unpruned_weights*10))
        unpruned_layers[unpruned].weight = nn.parameter.Parameter(unpruned_weights)
    return ticket



"""
Save:

torch.save(model.state_dict(), PATH)

Load:

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""
#initial_state_dict = model_gmp.state_dict()

#model_random.load_state_dict(initial_state_dict)

