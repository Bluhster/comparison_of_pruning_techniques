import argparse
import os
import time

import torch
import torch.optim
import torch.utils.data

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

from datetime import timedelta
from models import *
from utils import *

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('--arch', '-a', 
                    metavar='ARCH', 
                    default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', 
                    default=4, type=int, 
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', 
                    default=10, type=int, 
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--las-lam', 
                    default=0.0001, type=float, 
                    metavar='N',
                    help='to how much the L1 penalty is weighted when computing the loss for LASSO')
parser.add_argument('--las-thresh', 
                    default=0.001, type=float, 
                    metavar='N',
                    help='threshold for weights to be set to 0 for LASSO pruning')
parser.add_argument('--start-epoch', 
                    default=0, type=int, 
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', 
                    '--batch-size', 
                    default=128, type=int,
                    metavar='N', 
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', 
                    '--learning-rate', 
                    default=0.05, type=float,
                    metavar='LR', 
                    help='initial learning rate')
parser.add_argument('--momentum', 
                    default=0.9, type=float, 
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', 
                    '--wd', 
                    default=1e-4, type=float,
                    metavar='W', 
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', 
                    '-p', 
                    default=100, type=int,
                    metavar='N', 
                    help='print frequency (default: 50)')
parser.add_argument('--resume', 
                    default='', type=str, 
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', 
                    '--evaluate', 
                    dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', 
                    dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', 
                    dest='save_dir',
                    help='The directory used to save the trained models',
                    default='saved_models', type=str)
parser.add_argument('--save-every', 
                    dest='save_every', 
                    type=int, default=10,
                    help='Saves checkpoints at every specified number of epochs',)
parser.add_argument('-s',
                    '--sparsity',
                    help = 'Percentage of weights to remove',
                    type = float,
                    default = 0.8,)

best_prec1 = 0

def main(run = "", tim = ""):
    print(f"\n ----------------------- starting run: {run} ----------------------- \n")
    # initialize identical models
    # initialize model which will not be pruned first
    resnet20 = run.startswith("/0.8_resnet20") or run.startswith("/0.9_resnet20")
    if resnet20:
        model_unpruned              = resnet.resnet20()         # choices: resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
        # and save initial weights for ticket generation later
        initial_model_dict          = model_unpruned.state_dict()
        # instatiate one-shot models without loading weights first, because the weights will be loaded after the unpruned model was trained
        model_greedy_layer          = resnet.resnet20()
        model_gmp                   = resnet.resnet20()
        model_random                = resnet.resnet20()
        
        # then initialize iterative models and LASSO model, which will be trained from scratch, with initial values
        model_greedy_layer_it       = resnet.resnet20()
        model_greedy_layer_it.load_state_dict(model_unpruned.state_dict())
        model_gmp_it                = resnet.resnet20()
        model_gmp_it.load_state_dict(model_unpruned.state_dict())
        model_random_it             = resnet.resnet20()
        model_random_it.load_state_dict(model_unpruned.state_dict())
        model_LASSO                 = resnet.resnet20()
        model_LASSO.load_state_dict(model_unpruned.state_dict())
    else:
        model_unpruned              = resnet.resnet56()         # choices: resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
        # and save initial weights for ticket generation later
        initial_model_dict          = model_unpruned.state_dict()
        # instatiate one-shot models without loading weights first, because the weights will be loaded after the unpruned model was trained
        model_greedy_layer          = resnet.resnet56()
        model_gmp                   = resnet.resnet56()
        model_random                = resnet.resnet56()
        
        # then initialize iterative models and LASSO model, which will be trained from scratch, with initial values
        model_greedy_layer_it       = resnet.resnet56()
        model_greedy_layer_it.load_state_dict(model_unpruned.state_dict())
        model_gmp_it                = resnet.resnet56()
        model_gmp_it.load_state_dict(model_unpruned.state_dict())
        model_random_it             = resnet.resnet56()
        model_random_it.load_state_dict(model_unpruned.state_dict())
        model_LASSO                 = resnet.resnet56()
        model_LASSO.load_state_dict(model_unpruned.state_dict())

    # create stats dictionary to save all models' stats in
    stats_models = {}
    time_stats = {}

    start = time.time()
    moving_start = time.time()

    # first train the initial model without pruning it
    train_loss, train_acc, test_loss, test_acc = compute(model_unpruned, iterative_pruning = False, method = "", model_name = "model_unpruned", save = run)
    # and record the accuracy and loss
    stats_models["model_unpruned"] = helper.get_stats(model_unpruned)
    stats_models["model_unpruned"]["train_loss"], stats_models["model_unpruned"]["train_acc"] = train_loss, train_acc
    stats_models["model_unpruned"]["test_loss"], stats_models["model_unpruned"]["test_acc"] = test_loss, test_acc
    time_stats["unpruned"] = time.time() - moving_start
    moving_start = time.time()

    # firts info block for one-shot models (one-shot greedy layer-wise pruning)
    # load already trained model
    model_greedy_layer.load_state_dict(model_unpruned.state_dict())
    # apply the pruning technique to the model and check the sparsity
    to_prune, _ = helper.get_layers(model_greedy_layer)
    techniques.prune_resnet(sparsity = args.sparsity, module_dict = to_prune, method = "greedy_layer", param = "weight")
    helper.print_sparsity(to_prune, "model_greedy_layer")
    # retrain model for half of training epochs after pruning
    train_loss, train_acc, test_loss, test_acc = finetune(model_greedy_layer, int(0.5*args.epochs), "model_greedy_layer")
    # save the stats in dictionary of stats for all models
    stats_models["model_greedy_layer"] = helper.get_stats(model_greedy_layer)
    stats_models["model_greedy_layer"]["train_loss"], stats_models["model_greedy_layer"]["train_acc"] = train_loss, train_acc
    stats_models["model_greedy_layer"]["test_loss"], stats_models["model_greedy_layer"]["test_acc"] = test_loss, test_acc
    time_stats["greedy_layer"] = time.time() - moving_start
    moving_start = time.time()
    # create ticket from trained network
    ticket_greedy_layer = techniques.create_ticket(model_greedy_layer, initial_model_dict, resnet20)
    # train the ticket without applying any pruning technique
    train_loss, train_acc, test_loss, test_acc = compute(ticket_greedy_layer, iterative_pruning = False, method = '', model_name = "ticket_greedy_layer", save = run)
    # check the ticket's sparsity
    to_prune, _ = helper.get_layers(ticket_greedy_layer)
    helper.print_sparsity(to_prune, "ticket_greedy_layer")
    # save the stats in dictionary of stats for all models
    stats_models["ticket_greedy_layer"] = helper.get_stats(ticket_greedy_layer)
    stats_models["ticket_greedy_layer"]["train_loss"], stats_models["ticket_greedy_layer"]["train_acc"] = train_loss, train_acc
    stats_models["ticket_greedy_layer"]["test_loss"], stats_models["ticket_greedy_layer"]["test_acc"] = test_loss, test_acc
    time_stats["ticket_greedy_layer"] = time.time() - moving_start
    moving_start = time.time()

    # second info block for one-shot models (one-shot global magnitude pruning)
    model_gmp.load_state_dict(model_unpruned.state_dict())
    to_prune, _ = helper.get_layers(model_gmp)
    techniques.prune_resnet(sparsity = args.sparsity, module_dict = to_prune, method = "gmp", param = "weight")
    train_loss, train_acc, test_loss, test_acc = finetune(model_gmp, int(0.5*args.epochs), "model_gmp")
    stats_models["model_gmp"] = helper.get_stats(model_gmp)
    stats_models["model_gmp"]["train_loss"], stats_models["model_gmp"]["train_acc"] = train_loss, train_acc
    stats_models["model_gmp"]["test_loss"], stats_models["model_gmp"]["test_acc"] = test_loss, test_acc
    time_stats["gmp"] = time.time() - moving_start
    moving_start = time.time()
    ticket_gmp = techniques.create_ticket(model_gmp, initial_model_dict, resnet20)
    train_loss, train_acc, test_loss, test_acc = compute(ticket_gmp, iterative_pruning = False, method = '', model_name = "ticket_gmp", save = run)
    to_prune, _ = helper.get_layers(ticket_gmp)
    helper.print_sparsity(to_prune, "ticket_gmp")
    stats_models["ticket_gmp"] = helper.get_stats(ticket_gmp)
    stats_models["ticket_gmp"]["train_loss"], stats_models["ticket_gmp"]["train_acc"] = train_loss, train_acc
    stats_models["ticket_gmp"]["test_loss"], stats_models["ticket_gmp"]["test_acc"] = test_loss, test_acc
    time_stats["ticket_gmp"] = time.time() - moving_start
    moving_start = time.time()

    # third info block for one-shot models (one-shot global random pruning)
    model_random.load_state_dict(model_unpruned.state_dict())
    to_prune, _ = helper.get_layers(model_random)
    techniques.prune_resnet(sparsity = args.sparsity, module_dict = to_prune, method = "random", param = "weight")
    train_loss, train_acc, test_loss, test_acc = finetune(model_random, int(0.5*args.epochs), "model_random")
    stats_models["model_random"] = helper.get_stats(model_random)
    stats_models["model_random"]["train_loss"], stats_models["model_random"]["train_acc"] = train_loss, train_acc
    stats_models["model_random"]["test_loss"], stats_models["model_random"]["test_acc"] = test_loss, test_acc
    time_stats["random"] = time.time() - moving_start
    moving_start = time.time()
    ticket_random = techniques.create_ticket(model_random, initial_model_dict, resnet20)
    train_loss, train_acc, test_loss, test_acc = compute(ticket_random, iterative_pruning = False, method = '', model_name = "ticket_random", save = run)
    to_prune, _ = helper.get_layers(ticket_random)
    helper.print_sparsity(to_prune, "ticket_random")
    stats_models["ticket_random"] = helper.get_stats(ticket_random)
    stats_models["ticket_random"]["train_loss"], stats_models["ticket_random"]["train_acc"] = train_loss, train_acc
    stats_models["ticket_random"]["test_loss"], stats_models["ticket_random"]["test_acc"] = test_loss, test_acc
    time_stats["ticket_random"] = time.time() - moving_start
    moving_start = time.time()

    # first info block for iterative models (iterative greedy layer-wise pruning)
    # prune the network after each epoch except the last one while training it from scratch
    train_loss, train_acc, test_loss, test_acc = compute(model_greedy_layer_it, iterative_pruning = True, method = "greedy_layer", model_name = "model_greedy_layer_it", save = run)
    # check model's sparsity
    to_prune, _ = helper.get_layers(model_greedy_layer_it)
    helper.print_sparsity(to_prune, "model_greedy_layer_it")
    # save the stats in dictionary of stats for all models
    stats_models["model_greedy_layer_it"] = helper.get_stats(model_greedy_layer_it)
    stats_models["model_greedy_layer_it"]["train_loss"], stats_models["model_greedy_layer_it"]["train_acc"] = train_loss, train_acc
    stats_models["model_greedy_layer_it"]["test_loss"], stats_models["model_greedy_layer_it"]["test_acc"] = test_loss, test_acc
    time_stats["greedy_layer_it"] = time.time() - moving_start
    moving_start = time.time()
    # create the ticket from the iteratively pruned model
    ticket_greedy_layer_it = techniques.create_ticket(model_greedy_layer_it, initial_model_dict, resnet20)
    # train the ticket without applying any pruning
    train_loss, train_acc, test_loss, test_acc = compute(ticket_greedy_layer_it, iterative_pruning = False, method = '', model_name = "ticket_greedy_layer_it", save = run)
    # check the ticket's sparsity
    to_prune, _ = helper.get_layers(ticket_greedy_layer_it)
    helper.print_sparsity(to_prune, "ticket_greedy_layer_it")
    # save the stats in dictionary of stats for all models
    stats_models["ticket_greedy_layer_it"] = helper.get_stats(ticket_greedy_layer_it)
    stats_models["ticket_greedy_layer_it"]["train_loss"], stats_models["ticket_greedy_layer_it"]["train_acc"] = train_loss, train_acc
    stats_models["ticket_greedy_layer_it"]["test_loss"], stats_models["ticket_greedy_layer_it"]["test_acc"] = test_loss, test_acc
    time_stats["ticket_greedy_layer_it"] = time.time() - moving_start
    moving_start = time.time()
    
    # second info block for iterative Gmodels (iterative global magnitude pruning)
    train_loss, train_acc, test_loss, test_acc = compute(model_gmp_it, iterative_pruning = True, method = "gmp", model_name = "model_gmp_it", save = run)
    stats_models["model_gmp_it"] = helper.get_stats(model_gmp_it)
    stats_models["model_gmp_it"]["train_loss"], stats_models["model_gmp_it"]["train_acc"] = train_loss, train_acc
    stats_models["model_gmp_it"]["test_loss"], stats_models["model_gmp_it"]["test_acc"] = test_loss, test_acc
    time_stats["gmp_it"] = time.time() - moving_start
    moving_start = time.time()
    ticket_gmp_it = techniques.create_ticket(model_gmp_it, initial_model_dict, resnet20)
    train_loss, train_acc, test_loss, test_acc = compute(ticket_gmp_it, iterative_pruning = False, method = '', model_name = "ticket_gmp_it", save = run)
    to_prune, _ = helper.get_layers(ticket_gmp_it)
    helper.print_sparsity(to_prune, "ticket_gmp_it")
    stats_models["ticket_gmp_it"] = helper.get_stats(ticket_gmp_it)
    stats_models["ticket_gmp_it"]["train_loss"], stats_models["ticket_gmp_it"]["train_acc"] = train_loss, train_acc
    stats_models["ticket_gmp_it"]["test_loss"], stats_models["ticket_gmp_it"]["test_acc"] = test_loss, test_acc
    time_stats["ticket_gmp_it"] = time.time() - moving_start
    moving_start = time.time()
    
    # third info block for iterative models (iterative global random pruning)
    train_loss, train_acc, test_loss, test_acc = compute(model_random_it, iterative_pruning = True, method = "random", model_name = "model_random_it", save = run)
    stats_models["model_random_it"] = helper.get_stats(model_random_it)
    stats_models["model_random_it"]["train_loss"], stats_models["model_random_it"]["train_acc"] = train_loss, train_acc
    stats_models["model_random_it"]["test_loss"], stats_models["model_random_it"]["test_acc"] = test_loss, test_acc
    time_stats["random_it"] = time.time() - moving_start
    moving_start = time.time()
    ticket_random_it = techniques.create_ticket(model_random_it, initial_model_dict, resnet20)
    train_loss, train_acc, test_loss, test_acc = compute(ticket_random_it, iterative_pruning = False, method = '', model_name = "ticket_random_it", save = run)
    to_prune, _ = helper.get_layers(ticket_random_it)
    helper.print_sparsity(to_prune, "ticket_random_it")
    stats_models["ticket_random_it"] = helper.get_stats(ticket_random_it)
    stats_models["ticket_random_it"]["train_loss"], stats_models["ticket_random_it"]["train_acc"] = train_loss, train_acc
    stats_models["ticket_random_it"]["test_loss"], stats_models["ticket_random_it"]["test_acc"] = test_loss, test_acc
    time_stats["ticket_random_it"] = time.time() - moving_start
    moving_start = time.time()

    # info block for LASSO pruning
    # train the model with the LASSO criterion
    train_loss, train_acc, test_loss, test_acc  = compute(model_LASSO, iterative_pruning = False, method = "LASSO", model_name = "model_LASSO", save = run)
    # apply the defined threshold to the trained LASSO model
    to_prune, _ = helper.get_layers(model_LASSO)
    techniques.prune_resnet(sparsity = args.sparsity, module_dict = to_prune, method = "LASSO", param = "weight", LASSO_threshold = args.las_thresh)
    # record stats
    stats_models["model_LASSO"] = helper.get_stats(model_LASSO)
    stats_models["model_LASSO"]["train_loss"], stats_models["model_LASSO"]["train_acc"] = train_loss, train_acc
    stats_models["model_LASSO"]["test_loss"], stats_models["model_LASSO"]["test_acc"] = test_loss, test_acc
    time_stats["LASSO"] = time.time() - moving_start
    moving_start = time.time()
    # create the ticket
    ticket_LASSO = techniques.create_ticket(model_LASSO, initial_model_dict, resnet20)
    # train the ticket (just with the crossentropyloss not LASSO-criterion)
    train_loss, train_acc, test_loss, test_acc = compute(ticket_LASSO, iterative_pruning = False, method = '', model_name = "ticket_LASSO", save = run)
    # check the ticket's sparsity
    to_prune, _ = helper.get_layers(ticket_LASSO)
    helper.print_sparsity(to_prune, "ticket_LASSO")
    # record stats
    stats_models["ticket_LASSO"] = helper.get_stats(ticket_LASSO)
    stats_models["ticket_LASSO"]["train_loss"], stats_models["ticket_LASSO"]["train_acc"] = train_loss, train_acc
    stats_models["ticket_LASSO"]["test_loss"], stats_models["ticket_LASSO"]["test_acc"] = test_loss, test_acc
    time_stats["ticket_LASSO"] = time.time() - moving_start

    # save the stats dictionary
    helper.save_stats(stats_models, name = run)
    helper.save_stats(time_stats, name = tim)

    print(f"Complete training of all models and tickets took {time.time()- start}")

def compute(model, iterative_pruning = False, method = "", model_name = "model", save = ""):
    """Train the given model for a number of Epochs and return the collected losses and accuracies.
        When iterative_pruning is True the in method specified pruning algorithm is applied after each Epoch
        When iterative_pruning is False the pruning algorithm is only applied once over the whole network after training

    Parameters
    ----------
    model = nn.Module
        given model to train and prune

    iterative_pruning = bool
        specifies if the pruning algorithm is applied after every epoch (True) or only once after training (False)

    method = str
        gives the pruning technique to apply to the model

    model_name = str
        name of the training model

    Returns
    -------
    training_loss = list
        computed list with all losses during training

    training_acc = list
        computed list with all accuracies during training

    test_loss = list
        computed list with all test losses, one after each epoch

    test_acc = list
        computed list with all test accuracies, one after each epoch
    """
    global args, best_prec1
    args = parser.parse_args() 
    if save.startswith("/0.8"):
        args.sparsity = 0.8
    else:
        args.sparsity = 0.9
                                                            #  epochs    learning rate   save directory
    args.epochs, args.lr, args.save_dir                     = 10,         0.1,            "./saved_models" + save,
                                                            # lasso lambda      lasso_threshold
    args.las_lam, args.las_thresh                           = 0.0001,           0.001,
     # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # lists for plotting later
    training_loss = []
    training_acc =[]

    test_loss = []
    test_acc = []

    # Check whether to use GPU or CPU
    if torch.cuda.is_available():
        print("GPU is available")
        model.cuda()
    else:
        print("GPU not available, performance may be slow")
        model.cpu()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    train_data = helper.CIFAR10Dataset(root = './data', train = True, download = True)
    test_data = helper.CIFAR10Dataset(root = './data', train = False, download = False)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size = args.batch_size, 
                                               shuffle = True,
                                               num_workers = args.workers, 
                                               pin_memory = True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size, 
                                              shuffle=False,
                                              num_workers=args.workers, 
                                              pin_memory=True)

    # define loss function (criterion) and optimizer
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cpu()

    optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum = args.momentum,
                                weight_decay = args.weight_decay)

    if args.evaluate:
        validate(test_loader, model, criterion)
        return
    
    to_prune, _ = helper.get_layers(model)

    if iterative_pruning:
                                                # epochs -1 because after the last epoch pruning won't be applied
        sparsity = 1 - (1 - args.sparsity) ** (1 / (args.epochs-1))
    else:
        sparsity = args.sparsity

    print("\nStarting training with " + model_name + " for " + str(args.epochs) + " epochs\n")

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        current_loss, current_acc = train(train_loader, model, criterion, optimizer, epoch, sparsity, iterative_pruning, method)

        training_loss.append(current_loss)
        training_acc.append(current_acc)

        # evaluate on validation set
        prec1, loss = validate(test_loader, model, criterion)
        
        test_acc.append(prec1)
        test_loss.append(loss)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            helper.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, model_name + '_checkpoint.th'))

        helper.save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, model_name + '.th'))

    if not iterative_pruning and not ((method == "") | (method == "LASSO")):
        techniques.prune_resnet(sparsity = sparsity, module_dict = to_prune, method = method, param = "weight", LASSO_threshold = args.las_thresh)
        helper.print_sparsity(to_prune, model_name)
        print(f"\nStarting fine-tuning for {int(0.5*args.epochs)} epochs after pruning with {method}\n")
        for epoch in range(int(0.5*args.epochs)):
            current_loss, current_acc = train(train_loader, model, criterion, optimizer, epoch, sparsity, iterative_pruning, method)

            # evaluate on validation set
            prec1, loss = validate(test_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

    return training_loss, training_acc, test_loss, test_acc

def finetune(model, epochs, model_name = 'model'):
    global best_prec1
     # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # lists for plotting later
    training_loss = []
    training_acc =[]

    test_loss = []
    test_acc = []

    # Check whether to use GPU or CPU
    if torch.cuda.is_available():
        print("GPU is available")
        model.cuda()
    else:
        print("GPU not available, performance may be slow")
        model.cpu()

    cudnn.benchmark = True
    
    train_data = helper.CIFAR10Dataset(root = './data', train = True, download = True)
    test_data = helper.CIFAR10Dataset(root = './data', train = False, download = False)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size = args.batch_size, 
                                               shuffle = True,
                                               num_workers = args.workers, 
                                               pin_memory = True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size, 
                                              shuffle=False,
                                              num_workers=args.workers, 
                                              pin_memory=True)

    # define loss function (criterion) and optimizer
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cpu()

    optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum = args.momentum,
                                weight_decay = args.weight_decay)
    
    print(f"\nStarting fine-tuning for {epochs} epochs for {model_name}\n")

    for epoch in range(epochs):
        # train for one epoch
        current_loss, current_acc = train(train_loader, model, criterion, optimizer, epoch)

        training_loss.append(current_loss)
        training_acc.append(current_acc)

        # evaluate on validation set
        prec1, loss = validate(test_loader, model, criterion)
        
        test_acc.append(prec1)
        test_loss.append(loss)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            helper.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, model_name + '_checkpoint.th'))

        helper.save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, model_name + '.th'))

    to_prune, all_layers = helper.get_layers(model)
    helper.print_sparsity(to_prune, model_name)
    return training_loss, training_acc, test_loss, test_acc

def train(train_loader, model, criterion, optimizer, epoch, sparsity = 0.0 , iterative_pruning = False, method = ''):
    """Run one train epoch

    Parameters
    ----------
    train_loader = Dataloader
        Dataloader that yields training samples

    model = nn.Module
        given module to train

    criterion = nn.loss_function
        given loss function

    optimizer = torch.optim.Optimizer
        given optimizer

    epoch = int
        the number of the current epoch

    sparsity = float
        sparsity to apply each pruning iteration (only necessary for iterative pruning)

    iterative_pruning = bool
        determines whether iterative pruning will occur during the training epoch or not

    method = str
        method to use for iterative pruning

    Returns
    -------
    losses.val = float
        current loss in the epoch

    top1.val = float
        current accuracy in the epoch
    """
    batch_time = helper.AverageMeter()
    data_time = helper.AverageMeter()
    losses = helper.AverageMeter()
    top1 = helper.AverageMeter()

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    
    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            target = target.cuda()
            input_var = input.cuda()
        else:
            target = target.cpu()
            input_var = input.cpu()
        target_var = target

        # compute output
        output = model(input_var)

        if method == 'LASSO':
            loss = criterion(output, target_var)
            loss_lass = techniques.L1_reg(model, args.las_lam)
            loss = loss + loss_lass
        else:
            loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = helper.accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
        
    if iterative_pruning and epoch < args.epochs:
        to_prune, _ = helper.get_layers(model)
        techniques.prune_resnet(sparsity, to_prune, method, "weight")

    return losses.val, top1.val

def validate(val_loader, model, criterion):
    """Run evaluation on test dataset

    Parameters
    ----------
    val_loader = Dataloader
        Dataloader that yields test samples

    model = nn.Module
        given model to evaluate

    criterion = nn.loss_function
        given loss function

    Returns
    -------
    top1.avg = float
        average accuracy value during test

    losses.avg = float
        average loss during test
    """
    batch_time = helper.AverageMeter()
    losses = helper.AverageMeter()
    top1 = helper.AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()
            else:
                target = target.cpu()
                input_var = input.cpu()
                target_var = target.cpu()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = helper.accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Acc {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg

if __name__ == '__main__':
    resnet20_80_runs = [["/0.8_resnet20_runs/0.8_resnet20_1st_run", 
                         "/0.8_resnet20_runs/0.8_resnet20_2nd_run", 
                         "/0.8_resnet20_runs/0.8_resnet20_3rd_run", 
                         "/0.8_resnet20_runs/0.8_resnet20_4th_run", 
                         "/0.8_resnet20_runs/0.8_resnet20_5th_run"], 
                        ["/0.8_resnet20_runs/training_times_0.8_resnet20_1",
                         "/0.8_resnet20_runs/training_times_0.8_resnet20_2",
                         "/0.8_resnet20_runs/training_times_0.8_resnet20_3", 
                         "/0.8_resnet20_runs/training_times_0.8_resnet20_4", 
                         "/0.8_resnet20_runs/training_times_0.8_resnet20_5"]]
    resnet20_90_runs = [["/0.9_resnet20_runs/0.9_resnet20_1st_run", 
                         "/0.9_resnet20_runs/0.9_resnet20_2nd_run", 
                         "/0.9_resnet20_runs/0.9_resnet20_3rd_run", 
                         "/0.9_resnet20_runs/0.9_resnet20_4th_run", 
                         "/0.9_resnet20_runs/0.9_resnet20_5th_run"], 
                        ["/0.9_resnet20_runs/training_times_0.9_resnet20_1",
                         "/0.9_resnet20_runs/training_times_0.9_resnet20_2",
                         "/0.9_resnet20_runs/training_times_0.9_resnet20_3", 
                         "/0.9_resnet20_runs/training_times_0.9_resnet20_4", 
                         "/0.9_resnet20_runs/training_times_0.9_resnet20_5"]]
    resnet56_80_runs = [["/0.8_resnet56_runs/0.8_resnet56_1st_run", 
                         "/0.8_resnet56_runs/0.8_resnet56_2nd_run", 
                         "/0.8_resnet56_runs/0.8_resnet56_3rd_run", 
                         "/0.8_resnet56_runs/0.8_resnet56_4th_run", 
                         "/0.8_resnet56_runs/0.8_resnet56_5th_run"], 
                        ["/0.8_resnet56_runs/training_times_0.8_resnet56_1",
                         "/0.8_resnet56_runs/training_times_0.8_resnet56_2",
                         "/0.8_resnet56_runs/training_times_0.8_resnet56_3", 
                         "/0.8_resnet56_runs/training_times_0.8_resnet56_4", 
                         "/0.8_resnet56_runs/training_times_0.8_resnet56_5"]]
    resnet56_90_runs = [["/0.9_resnet56_runs/0.9_resnet56_1st_run", 
                         "/0.9_resnet56_runs/0.9_resnet56_2nd_run", 
                         "/0.9_resnet56_runs/0.9_resnet56_3rd_run", 
                         "/0.9_resnet56_runs/0.9_resnet56_4th_run", 
                         "/0.9_resnet56_runs/0.9_resnet56_5th_run"], 
                        ["/0.9_resnet56_runs/training_times_0.9_resnet56_1",
                         "/0.9_resnet56_runs/training_times_0.9_resnet56_2",
                         "/0.9_resnet56_runs/training_times_0.9_resnet56_3", 
                         "/0.9_resnet56_runs/training_times_0.9_resnet56_4", 
                         "/0.9_resnet56_runs/training_times_0.9_resnet56_5"]]
    full = [resnet20_80_runs, resnet20_90_runs, resnet56_80_runs, resnet56_90_runs]
    test_runs = [[["/0.8_resnet20_new_run"],["/0.8_resnet20_times_new_run"]],
                 [["/0.9_resnet20_new_run"],["/0.9_resnet20_times_new_run"]],
                 [["/0.8_resnet56_new_run"],["/0.8_resnet56_times_new_run"]],
                 [["/0.9_resnet56_new_run"],["/0.9_resnet56_times_new_run"]]]
    # "test_runs" was "full" when conducting the experiment mentioned in the paper
    for runs in test_runs:                           
        for run,times in zip(runs[0],runs[1]):
            main(run,times)
