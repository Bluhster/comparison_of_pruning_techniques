import torch
import json
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from ptflops import get_model_complexity_info

import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def save_stats(stats, name):
    """Saves the given dictionary in folder "stats" as file with given name

    Parameters
    ----------
    stats = dict
        given stats dict with entries for all the computed models and their documented stats

    name = str
        the name of the resulting file in "stats"
    """
    json_object = json.dumps(stats, indent = 3)

    with open("./stats/" + name, "w") as file:
        file.write(json_object)


def get_complexity(model):
    """Computes computational complexity and the number of parameters for a given model

    Parameters
    ----------
    model = nn.Module
        given model to compute complexity and number of parameters for

    Return
    ------
    macs = str
        string containing the calculated computational complexity of the model in Gmac
    """
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                print_per_layer_stat=False, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs

class CIFAR10Dataset(Dataset):
    """CIFAR10 Dataset

    Parameters
    ----------
    root = str
        Directory where the data is located or downloaded to
    
    train = bool
        If True training set it returned, if False test set it returned

    Attributes
    ----------
    dataset = CIFAR10
        instance of torchvision CIFAR10 class
    """
    def __init__(self, root, train = True, download = True):
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                ]
            )
        else: transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                ]
            )

        self.dataset = CIFAR10(
            root = root,
            train = train,
            download = download,
            transform = transform,
        )
    
    def __len__(self):
        """return length of dataset
        """
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get selected sample
        
        Parameters
        ----------
        idx = int
            Index of sample to get

        Returns
        -------
        x = torch.Tensor
            Selected sample at idx
        y = torch.Tensor
            Target label at idx
        """
        return self.dataset[idx]
    
class AverageMeter(object):
    """Computes and stores the average and current value
    
    Parameters
    ----------
    val = float
        current value

    avg = float
        average value

    sum = float
        sum of all values

    count = int
        number of collected values
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    
    Parameters
    ----------
    output = some value of same type as target
        computed output of the network

    target = some value of same type as output
        given correct outputs as they should have been computed

    topk = tuple
        the top accuracy in k

    Returns
    -------
    res = list
        results
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_stats(model):
    """Compute useful information about a given model
    
    Parameters
    ----------
    model = nn.Module
        given model

    Returns
    -------
    stats = dict
        dictionary containing statistics
    """
    stats = {}
    total_params = 0
    total_pruned_params = 0

    module_dict, _ = get_layers(model)

    for layer in module_dict:
        #assert check_pruned(module)

        weight_mask = module_dict[layer].weight
        #bias_mask = module.bias_mask

        params = weight_mask.numel() #+ bias_mask.numel()
        pruned_params = int((weight_mask == 0).sum()) #+ (bias_mask == 0).sum()

        total_params += params
        total_pruned_params += pruned_params

        stats[f"{layer}_total_parameters"] = params
        stats[f"{layer}_pruned_parameters"] = pruned_params
        stats[f"{layer}_prune_ratio"] = float(pruned_params / params)
    
    stats["computational_complexity"] = get_complexity(model)
    stats["total_parameters"] = total_params
    stats["total_pruned_parameters"] = total_pruned_params
    stats["total_prune_ratio"] = total_pruned_params / total_params

    return stats

def get_layers(model):
    """Get all layers which are to prune and all layers in general from a given model
    
    Parameters
    ----------
    model = nn.Module
        given model to get layers from

    Returns
    -------
    layers_to_prune = dict
        dictionary with names and instances of all layers to prune

    all_layers = dict
        dictionary with names and instances of all layers
    """
    layers_to_prune = {}
    all_layers = {}

    for name, module in model.named_modules():

        all_layers[name] = module
        
        # only prune convolutional and linear layers and save names of pruned layers
        if isinstance(module, nn.Conv2d) | isinstance(module, nn.Linear):

            layers_to_prune[name] = module
    return layers_to_prune, all_layers

def print_sparsity(module_dict, model_name = ''):
    """Print sparsity of a model's pruned layers
    
    Parameters
    ----------
    module_dict = dict
        dictionary containing all pruned layers of a model

    model_name = str
        given model name for printing which model's sparsity will be printed
    """
    # print full sparsity
    print("\nPrinting sparsities of all pruned layers for " + model_name + "\n")
    for layer in module_dict:
        print(
        "Sparsity in " + str(layer) + ": {:.2f}%".format(
            100. * float(torch.sum(module_dict[layer].weight == 0))
            / float(module_dict[layer].weight.nelement())
        ))

    sum = 0.0
    sum_e = 0.0
    for layer in module_dict:
        sum += torch.sum(module_dict[layer].weight == 0)
        sum_e += module_dict[layer].weight.nelement()

    print(
        "\nGlobal sparsity for " + model_name + ": {:.2f}%\n".format(
        100. * sum
        / sum_e)
    )

def compare(stats, models, epochs, title = "Comparison of Accuracy and Loss"):
    """Compares training accuracy and loss, test accuracy and loss of given models

    Parameters
    ----------
    stats = dict
        stats dictionary with entries for all models

    models = list
        list with strings of model names that should be compared

    epochs = int
        number of epochs that were run

    title = str
        title of the whole plot
    """
    epo = np.arange(epochs)+1
    x_ticks = np.arange(0, max(epo)+1, 10)
    
    fig, ax = plt.subplots(2, 2, sharex = "col", sharey = "row")
    fig.suptitle(title)
    ax[0,0].set_title('Training loss', loc = 'left')
    #ax[0,0].set_xlabel('Epochs')
    ax[0,0].set_ylabel('Loss')
    ax[0,0].set_xticks(x_ticks)
    ax[0,0].grid()
    
    ax[0,1].set_title('Test loss', loc = 'left')
    #ax[0,1].set_xlabel('Epochs')
    #ax[0,1].set_ylabel('Loss')
    ax[0,1].set_xticks(x_ticks)
    ax[0,1].grid()
    
    ax[1,0].set_title('Training accuracy', loc = 'left')
    ax[1,0].set_xlabel('Epochs')
    ax[1,0].set_ylabel('Accuracy')
    ax[1,0].set_xticks(x_ticks)
    ax[1,0].grid()
    
    ax[1,1].set_title('Test accuracy', loc = 'left')
    ax[1,1].set_xlabel('Epochs')
    #ax[1,1].set_ylabel('Accuracy')
    ax[1,1].set_xticks(x_ticks)
    
    ax[1,1].grid()

    for model in models:
        ax[0,0].plot(epo, stats[model]["train_loss"], label = model)

        ax[0,1].plot(epo, stats[model]["test_loss"], label = model)

        ax[1,0].plot(epo, stats[model]["train_acc"], label = model)
        
        ax[1,1].plot(epo, stats[model]["test_acc"], label = model)
        
    ax[1,1].legend(loc = "center left", bbox_to_anchor = (1,0))    

    plt.tight_layout()
    plt.show()



def plot_loss_acc(stats, epochs, title = "Loss and Accuracy", train_values = True):
    """Plots given losses and accuracies over all epochs

    Parameters
    ----------
    stats = dict
        stats dictionary with all the models as entries
    
    epochs = int
        number of run epochs

    title = str
        title of the whole plot

    train_values = bool
        defines whether to plot training or test values
    """
    epo = np.arange(epochs)+1
    x_ticks = np.arange(0, max(epo)+1, 10)
    
    fig, ax = plt.subplots(1,2)
    fig.suptitle(title)
    ax[0].set_title('Loss', loc = 'left')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_xticks(x_ticks)
    ax[0].grid()

    ax[1].set_title('Accuracy', loc = 'left')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xticks(x_ticks)
    ax[1].grid()

    if train_values:
        for model in stats:
            ax[0].plot(epo, stats[model]["train_loss"], label = model)

            ax[1].plot(epo, stats[model]["train_acc"], label = model)

    else:
        for model in stats:
            ax[0].plot(epo, stats[model]["test_loss"], label = model)

            ax[1].plot(epo, stats[model]["test_acc"], label = model)
    
    ax[1].legend(loc = "center left", bbox_to_anchor = (1,0))    

    plt.tight_layout()
    plt.show()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save the training model
    """
    torch.save(state, filename)