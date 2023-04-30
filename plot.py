import torch
import json

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def main():
    with open("./stats/0.8_resnet20_runs/0.8_resnet20_1st_run", "r") as file:
        resnet20_80_1 = json.load(file)
    with open("./stats/0.8_resnet20_runs/0.8_resnet20_2nd_run", "r") as file:
        resnet20_80_2 = json.load(file)
    with open("./stats/0.8_resnet20_runs/0.8_resnet20_3rd_run", "r") as file:
        resnet20_80_3 = json.load(file)
    with open("./stats/0.8_resnet20_runs/0.8_resnet20_4th_run", "r") as file:
        resnet20_80_4 = json.load(file)
    with open("./stats/0.8_resnet20_runs/0.8_resnet20_5th_run", "r") as file:
        resnet20_80_5 = json.load(file)

    with open("./stats/0.9_resnet20_runs/0.9_resnet20_1st_run", "r") as file:
        resnet20_90_1 = json.load(file)
    with open("./stats/0.9_resnet20_runs/0.9_resnet20_2nd_run", "r") as file:
        resnet20_90_2 = json.load(file)
    with open("./stats/0.9_resnet20_runs/0.9_resnet20_3rd_run", "r") as file:
        resnet20_90_3 = json.load(file)
    with open("./stats/0.9_resnet20_runs/0.9_resnet20_4th_run", "r") as file:
        resnet20_90_4 = json.load(file)
    with open("./stats/0.9_resnet20_runs/0.9_resnet20_5th_run", "r") as file:
        resnet20_90_5 = json.load(file)

    with open("./stats/0.8_resnet56_runs/0.8_resnet56_1st_run", "r") as file:
        resnet56_80_1 = json.load(file)
    with open("./stats/0.8_resnet56_runs/0.8_resnet56_2nd_run", "r") as file:
        resnet56_80_2 = json.load(file)
    with open("./stats/0.8_resnet56_runs/0.8_resnet56_3rd_run", "r") as file:
        resnet56_80_3 = json.load(file)
    with open("./stats/0.8_resnet56_runs/0.8_resnet56_4th_run", "r") as file:
        resnet56_80_4 = json.load(file)
    with open("./stats/0.8_resnet56_runs/0.8_resnet56_5th_run", "r") as file:
        resnet56_80_5 = json.load(file)

    with open("./stats/0.9_resnet56_runs/0.9_resnet56_1st_run", "r") as file:
        resnet56_90_1 = json.load(file)
    with open("./stats/0.9_resnet56_runs/0.9_resnet56_2nd_run", "r") as file:
        resnet56_90_2 = json.load(file)
    with open("./stats/0.9_resnet56_runs/0.9_resnet56_3rd_run", "r") as file:
        resnet56_90_3 = json.load(file)
    with open("./stats/0.9_resnet56_runs/0.9_resnet56_4th_run", "r") as file:
        resnet56_90_4 = json.load(file)
    with open("./stats/0.9_resnet56_runs/0.9_resnet56_5th_run", "r") as file:
        resnet56_90_5 = json.load(file)        

    def fill_full(to_fill, name = "", values = []):
        to_fill["resnet20_80_avg"]  = average(resnet20_80_runs, name, values)
        to_fill["resnet20_90_avg"]  = average(resnet20_90_runs, name, values)
        to_fill["resnet56_80_avg"]  = average(resnet56_80_runs, name, values)
        to_fill["resnet56_90_avg"]  = average(resnet56_90_runs, name, values)
        temp_dict20_80_1, temp_dict20_80_2, temp_dict20_80_3, temp_dict20_80_4, temp_dict20_80_5 = {}, {}, {}, {}, {}
        temp_dict20_90_1, temp_dict20_90_2, temp_dict20_90_3, temp_dict20_90_4, temp_dict20_90_5 = {}, {}, {}, {}, {}
        temp_dict56_80_1, temp_dict56_80_2, temp_dict56_80_3, temp_dict56_80_4, temp_dict56_80_5 = {}, {}, {}, {}, {}
        temp_dict56_90_1, temp_dict56_90_2, temp_dict56_90_3, temp_dict56_90_4, temp_dict56_90_5 = {}, {}, {}, {}, {}

        for value in values:
            temp_dict20_80_1[value] = resnet20_80_1[name][value]
            temp_dict20_80_2[value] = resnet20_80_2[name][value]
            temp_dict20_80_3[value] = resnet20_80_3[name][value]
            temp_dict20_80_4[value] = resnet20_80_4[name][value]
            temp_dict20_80_5[value] = resnet20_80_5[name][value]
            
            temp_dict20_90_1[value] = resnet20_90_1[name][value]
            temp_dict20_90_2[value] = resnet20_90_2[name][value]
            temp_dict20_90_3[value] = resnet20_90_3[name][value]
            temp_dict20_90_4[value] = resnet20_90_4[name][value]
            temp_dict20_90_5[value] = resnet20_90_5[name][value]

            temp_dict56_80_1[value] = resnet56_80_1[name][value]
            temp_dict56_80_2[value] = resnet56_80_2[name][value]
            temp_dict56_80_3[value] = resnet56_80_3[name][value]
            temp_dict56_80_4[value] = resnet56_80_4[name][value]
            temp_dict56_80_5[value] = resnet56_80_5[name][value]

            temp_dict56_90_1[value] = resnet56_90_1[name][value]
            temp_dict56_90_2[value] = resnet56_90_2[name][value]
            temp_dict56_90_3[value] = resnet56_90_3[name][value]
            temp_dict56_90_4[value] = resnet56_90_4[name][value]
            temp_dict56_90_5[value] = resnet56_90_5[name][value]
        
        to_fill["resnet20_80_1"] = temp_dict20_80_1
        to_fill["resnet20_80_2"] = temp_dict20_80_2
        to_fill["resnet20_80_3"] = temp_dict20_80_3
        to_fill["resnet20_80_4"] = temp_dict20_80_4
        to_fill["resnet20_80_5"] = temp_dict20_80_5

        to_fill["resnet20_90_1"] = temp_dict20_90_1
        to_fill["resnet20_90_2"] = temp_dict20_90_2
        to_fill["resnet20_90_3"] = temp_dict20_90_3
        to_fill["resnet20_90_4"] = temp_dict20_90_4
        to_fill["resnet20_90_5"] = temp_dict20_90_5

        to_fill["resnet56_80_1"] = temp_dict56_80_1
        to_fill["resnet56_80_2"] = temp_dict56_80_2
        to_fill["resnet56_80_3"] = temp_dict56_80_3
        to_fill["resnet56_80_4"] = temp_dict56_80_4
        to_fill["resnet56_80_5"] = temp_dict56_80_5

        to_fill["resnet56_90_1"] = temp_dict56_90_1
        to_fill["resnet56_90_2"] = temp_dict56_90_2
        to_fill["resnet56_90_3"] = temp_dict56_90_3
        to_fill["resnet56_90_4"] = temp_dict56_90_4
        to_fill["resnet56_90_5"] = temp_dict56_90_5

        return to_fill

    all_models = ["model_unpruned", "model_greedy_layer", "model_gmp", "model_random",
                "model_greedy_layer_it", "model_gmp_it", "model_random_it",
                "ticket_greedy_layer", "ticket_gmp", "ticket_random",
                "ticket_greedy_layer_it", "ticket_gmp_it", "ticket_random_it"]
    one_models = ["model_greedy_layer", "model_gmp", "model_random"]
    it_models = ["model_greedy_layer_it", "model_gmp_it", "model_random_it"]
    one_tickets = ["model_unpruned", "ticket_greedy_layer", "ticket_gmp", "ticket_random"]
    it_tickets = ["model_unpruned", "ticket_greedy_layer_it", "ticket_gmp_it", "ticket_random_it"]
    all_tickets = ["model_unpruned", "ticket_greedy_layer", "ticket_gmp", "ticket_random", "ticket_greedy_layer_it", "ticket_gmp_it", "ticket_random_it"]

    values = ["train_loss","test_loss","train_acc", "test_acc"]

    all_runs         = [resnet20_80_1, resnet20_80_2, resnet20_80_3, resnet20_80_4, resnet20_80_5,
                        resnet20_90_1, resnet20_90_2, resnet20_90_3, resnet20_90_4, resnet20_90_5,
                        resnet56_80_1, resnet56_80_2, resnet56_80_3, resnet56_80_4, resnet56_80_5,
                        resnet56_90_1, resnet56_90_2, resnet56_90_3, resnet56_90_4, resnet56_90_5]
    resnet20_80_runs = [resnet20_80_1, resnet20_80_2, resnet20_80_3, resnet20_80_4, resnet20_80_5]
    resnet20_90_runs = [resnet20_90_1, resnet20_90_2, resnet20_90_3, resnet20_90_4, resnet20_90_5]
    resnet56_80_runs = [resnet56_80_1, resnet56_80_2, resnet56_80_3, resnet56_80_4, resnet56_80_5]
    resnet56_90_runs = [resnet56_90_1, resnet56_90_2, resnet56_90_3, resnet56_90_4, resnet56_90_5]

    all_in_one              = {}
    unpruned_full           = {}
    greedy_layer_full       = {}
    greedy_layer_it_full    = {}
    gmp_full                = {}
    gmp_it_full             = {}
    random_full             = {}
    random_it_full          = {}
    LASSO_full              = {}

    greedy_layer_ticket_full    = {}
    greedy_layer_it_ticket_full = {}
    gmp_ticket_full             = {}
    gmp_it_ticket_full          = {}
    random_ticket_full          = {}
    random_it_ticket_full       = {}
    LASSO_ticket_full           = {}


    all_in_one["model_unpruned"]            = fill_full(unpruned_full, "model_unpruned", values)
    remove_unpruned(unpruned_full)
    all_in_one["model_greedy_layer"]        = fill_full(greedy_layer_full, "model_greedy_layer", values)
    all_in_one["model_greedy_layer_it"]     = fill_full(greedy_layer_it_full, "model_greedy_layer_it", values)
    all_in_one["model_gmp"]                 = fill_full(gmp_full, "model_gmp", values)
    all_in_one["model_gmp_it"]              = fill_full(gmp_it_full, "model_gmp_it", values)
    all_in_one["model_random"]              = fill_full(random_full, "model_random", values)
    all_in_one["model_random_it"]           = fill_full(random_it_full, "model_random_it", values)
    all_in_one["model_LASSO"]               = fill_full(LASSO_full, "model_LASSO", values)

    all_in_one["ticket_greedy_layer"]       = fill_full(greedy_layer_ticket_full, "ticket_greedy_layer", values)
    all_in_one["ticket_greedy_layer_it"]    = fill_full(greedy_layer_it_ticket_full, "ticket_greedy_layer_it", values)
    all_in_one["ticket_gmp"]                = fill_full(gmp_ticket_full, "ticket_gmp", values)
    all_in_one["ticket_gmp_it"]             = fill_full(gmp_it_ticket_full, "ticket_gmp_it", values)
    all_in_one["ticket_random"]             = fill_full(random_ticket_full, "ticket_random", values)
    all_in_one["ticket_random_it"]          = fill_full(random_it_ticket_full, "ticket_random_it", values)
    all_in_one["ticket_LASSO"]              = fill_full(LASSO_ticket_full, "ticket_LASSO", values)

    mean_std(all_in_one)
    
    plot_test_averages(all_in_one)
    plot_tickets_vs_unpruned(all_in_one, all_tickets)
    helper.save_stats(all_in_one, "full_loss_accuracy")

    resnet20_80_runs = ["resnet20_80_1", "resnet20_80_2", "resnet20_80_3", "resnet20_80_4", "resnet20_80_5"]
    resnet20_90_runs = ["resnet20_90_1", "resnet20_90_2", "resnet20_90_3", "resnet20_90_4", "resnet20_90_5"]
    resnet56_80_runs = ["resnet56_80_1", "resnet56_80_2", "resnet56_80_3", "resnet56_80_4", "resnet56_80_5"]
    resnet56_90_runs = ["resnet56_90_1", "resnet56_90_2", "resnet56_90_3", "resnet56_90_4", "resnet56_90_5"]

    print_info(all_in_one, ["ticket_greedy_layer", "ticket_greedy_layer_it", "ticket_LASSO"], resnet56_80_runs, test = True)

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
    ax[0,0].set_ylabel('Loss')
    ax[0,0].set_xticks(x_ticks)
    ax[0,0].grid()
    
    ax[0,1].set_title('Test loss', loc = 'left')
    ax[0,1].set_xticks(x_ticks)
    ax[0,1].grid()
    
    ax[1,0].set_title('Training accuracy', loc = 'left')
    ax[1,0].set_xlabel('Epochs')
    ax[1,0].set_ylabel('Accuracy')
    ax[1,0].set_xticks(x_ticks)
    ax[1,0].grid()
    
    ax[1,1].set_title('Test accuracy', loc = 'left')
    ax[1,1].set_xlabel('Epochs')
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

def average(runs, model, values):
    out_dict = {}
    for value in values:
        

        list0 = runs[0][model][value]
        list1 = runs[1][model][value]
        list2 = runs[2][model][value]
        list3 = runs[3][model][value]
        list4 = runs[4][model][value]
        sums = [sum(x) for x in zip(list0,list1,list2,list3,list4)]
        out_dict[value] = [x/5 for x in sums]

    return out_dict

def plot_average_test(dict, epochs, title = ""):
    epo = np.arange(epochs)+1
    x_ticks = np.arange(0, max(epo)+1, 10)
    
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(title)
    ax[0].set_title('Test loss', loc = 'left')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_xticks(x_ticks)
    ax[0].grid()

    ax[1].set_title('Test Accuracy', loc = 'left')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xticks(x_ticks)
    ax[1].grid()

    for run in dict:
        if run == "resnet20_80_avg" or run == "resnet20_90_avg" or run == "resnet56_80_avg" or run == "resnet56_90_avg":
            if run == "resnet20_80_avg":
                col = "g"
            elif run == "resnet20_90_avg":
                col = "r"
            elif run == "resnet56_80_avg":
                col = "b"
            elif run == "resnet56_90_avg":
                col = "k"
            ax[0].plot(epo, dict[run]["test_loss"], label = run, color = col)
            ax[1].plot(epo, dict[run]["test_acc"], label = run, color = col)
        elif run == "resnet20_80_1" or run == "resnet20_80_2" or run == "resnet20_80_3" or run == "resnet20_80_4" or run == "resnet20_80_5":
            col = "g"
            ax[0].plot(epo, dict[run]["test_loss"], alpha = 0.15, color = col)
            ax[1].plot(epo, dict[run]["test_acc"], alpha = 0.15, color = col)
        elif run == "resnet20_90_1" or run == "resnet20_90_2" or run == "resnet20_90_3" or run == "resnet20_90_4" or run == "resnet20_90_5":
            col = "r"
            ax[0].plot(epo, dict[run]["test_loss"], alpha = 0.15, color = col)
            ax[1].plot(epo, dict[run]["test_acc"], alpha = 0.15, color = col)
        elif run == "resnet56_80_1" or run == "resnet56_80_2" or run == "resnet56_80_3" or run == "resnet56_80_4" or run == "resnet56_80_5":
            col = "b"
            ax[0].plot(epo, dict[run]["test_loss"], alpha = 0.15, color = col)
            ax[1].plot(epo, dict[run]["test_acc"], alpha = 0.15, color = col)
        elif run == "resnet56_90_1" or run == "resnet56_90_2" or run == "resnet56_90_3" or run == "resnet56_90_4" or run == "resnet56_90_5":
            col = "k"
            ax[0].plot(epo, dict[run]["test_loss"], alpha = 0.15, color = col)
            ax[1].plot(epo, dict[run]["test_acc"], alpha = 0.15, color = col)
    
    ax[1].legend(loc = "lower right", framealpha=1)   

    plt.tight_layout()
    plt.show()

def plot_single(model_dict, epochs, model_name = ""):
    epo = np.arange(epochs)+1
    x_ticks = np.arange(0, max(epo)+1, 10)
    
    fig, ax = plt.subplots(2, 2, sharex = "col", sharey = "row")
    fig.suptitle(title)
    ax[0,0].set_title('Test loss', loc = 'left')
    ax[0,0].set_ylabel('Loss')
    ax[0,0].set_xticks(x_ticks)
    ax[0,0].grid()
    
    ax[0,1].set_title('Training loss', loc = 'left')
    ax[0,1].set_xticks(x_ticks)
    ax[0,1].grid()
    
    ax[1,0].set_title('Test accuracy', loc = 'left')
    ax[1,0].set_xlabel('Epochs')
    ax[1,0].set_ylabel('Accuracy')
    ax[1,0].set_xticks(x_ticks)
    ax[1,0].grid()
    
    ax[1,1].set_title('Training accuracy', loc = 'left')
    ax[1,1].set_xlabel('Epochs')
    ax[1,1].set_xticks(x_ticks)
    
    ax[1,1].grid()



    for model in models:
        ax[0,0].plot(epo, stats[model]["test_loss"], label = model)

        ax[0,1].plot(epo, stats[model]["train_loss"], label = model)

        ax[1,0].plot(epo, stats[model]["test_acc"], label = model)
        
        ax[1,1].plot(epo, stats[model]["train_acc"], label = model)

def plot(all, epochs, title = "", models = [], data = [], test = True):
    epo = np.arange(epochs)+1
    x_ticks = np.arange(0, max(epo)+1, 10)
    
    if test:
        fig, ax = plt.subplots(1, 2)
        fig.suptitle(title)
        ax[0].set_title('Test loss', loc = 'left')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_xticks(x_ticks)
        ax[0].grid()

        ax[1].set_title('Test Accuracy', loc = 'left')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('Epochs')
        ax[1].set_xticks(x_ticks)
        ax[1].grid()

        for model in models:
            for dat in data:
                ax[0].plot(epo, all[model][dat]["test_loss"], label = model)
                ax[1].plot(epo, all[model][dat]["test_acc"], label = model)
        
        ax[1].legend(loc = "lower right", framealpha=1)     

        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(2, 2, sharex = "col", sharey = "row")
        fig.suptitle(title)
        ax[0,0].set_title('Test Accuracy ResNet20/80%', loc = 'left')
        ax[0,0].set_ylabel('Accuracy')
        ax[0,0].set_xticks(x_ticks)
        ax[0,0].grid()
        
        ax[0,1].set_title('Test Accuracy ResNet20/90%', loc = 'left')
        ax[0,1].set_xticks(x_ticks)
        ax[0,1].grid()
        
        ax[1,0].set_title('Test Accuracy ResNet56/80%', loc = 'left')
        ax[1,0].set_xlabel('Epochs')
        ax[1,0].set_ylabel('Accuracy')
        ax[1,0].set_xticks(x_ticks)
        ax[1,0].grid()
        
        ax[1,1].set_title('Test Accuracy ResNet56/90%', loc = 'left')
        ax[1,1].set_xlabel('Epochs')
        ax[1,1].set_xticks(x_ticks)
        ax[1,1].grid()

        for model in models:
            for dat in data:
                if model == "model_unpruned":
                    if dat == "resnet20_80_avg":
                        ax[0,0].plot(epo, all[model][dat]["test_acc"], label = model)
                    elif dat == "resnet20_90_avg":
                        ax[0,1].plot(epo, all[model]["resnet20_80_avg"]["test_acc"], label = model)
                    elif dat == "resnet56_80_avg":
                        ax[1,0].plot(epo, all[model][dat]["test_acc"], label = model)
                    elif dat == "resnet56_90_avg":
                        ax[1,1].plot(epo, all[model]["resnet56_80_avg"]["test_acc"], label = model)
                else:
                    if dat == "resnet20_80_avg":
                        ax[0,0].plot(epo, all[model][dat]["test_acc"], label = model)
                    elif dat == "resnet20_90_avg":
                        ax[0,1].plot(epo, all[model][dat]["test_acc"], label = model)
                    elif dat == "resnet56_80_avg":
                        ax[1,0].plot(epo, all[model][dat]["test_acc"], label = model)
                    elif dat == "resnet56_90_avg":
                        ax[1,1].plot(epo, all[model][dat]["test_acc"], label = model)
        ax[1,1].legend(loc = "lower right", framealpha=1)   

        plt.tight_layout()
        plt.show()

def plot_average(dict, epochs, models = [], title= ""):
    epo = np.arange(epochs)+1
    x_ticks = np.arange(0, max(epo)+1, 10)
    
    fig, ax = plt.subplots(2, 2, sharex = "col", sharey = "row")
    fig.suptitle(title)
    ax[0,0].set_title('Test Loss', loc = 'left')
    ax[0,0].set_ylabel('Loss')
    ax[0,0].set_xticks(x_ticks)
    ax[0,0].grid()
    
    ax[0,1].set_title('Train Loss', loc = 'left')
    ax[0,1].set_xticks(x_ticks)
    ax[0,1].grid()
    
    ax[1,0].set_title('Test Accuracy', loc = 'left')
    ax[1,0].set_xlabel('Epochs')
    ax[1,0].set_ylabel('Accuracy')
    ax[1,0].set_xticks(x_ticks)
    ax[1,0].grid()
    
    ax[1,1].set_title('Training Accuracy', loc = 'left')
    ax[1,1].set_xlabel('Epochs')
    ax[1,1].set_xticks(x_ticks)
    
    ax[1,1].grid()

    for model in models:
        for run in dict[model]:
            if run == "resnet20_80_avg" or run == "resnet20_90_avg" or run == "resnet56_80_avg" or run == "resnet56_90_avg":
                if run == "resnet20_80_avg":
                    col = "g"
                elif run == "resnet20_90_avg":
                    col = "r"
                elif run == "resnet56_80_avg":
                    col = "b"
                elif run == "resnet56_90_avg":
                    col = "k"
                ax[0,0].plot(epo, dict[model][run]["test_loss"], label = model, color = col)
                ax[0,1].plot(epo, dict[model][run]["train_loss"], label = model, color = col)
                ax[1,0].plot(epo, dict[model][run]["test_acc"], label = model, color = col)
                ax[1,1].plot(epo, dict[model][run]["train_acc"], label = model, color = col)
            elif run == "resnet20_80_1" or run == "resnet20_80_2" or run == "resnet20_80_3" or run == "resnet20_80_4" or run == "resnet20_80_5":
                col = "g"
                ax[0,0].plot(epo, dict[model][run]["test_loss"], alpha = 0.15, color = col)
                ax[0,1].plot(epo, dict[model][run]["train_loss"], alpha = 0.15, color = col)
                ax[1,0].plot(epo, dict[model][run]["test_acc"], alpha = 0.15, color = col)
                ax[1,1].plot(epo, dict[model][run]["train_acc"], alpha = 0.15, color = col)
            elif run == "resnet20_90_1" or run == "resnet20_90_2" or run == "resnet20_90_3" or run == "resnet20_90_4" or run == "resnet20_90_5":
                col = "r"
                ax[0,0].plot(epo, dict[model][run]["test_loss"], alpha = 0.15, color = col)
                ax[0,1].plot(epo, dict[model][run]["train_loss"], alpha = 0.15, color = col)
                ax[1,0].plot(epo, dict[model][run]["test_acc"], alpha = 0.15, color = col)
                ax[1,1].plot(epo, dict[model][run]["train_acc"], alpha = 0.15, color = col)
            elif run == "resnet56_80_1" or run == "resnet56_80_2" or run == "resnet56_80_3" or run == "resnet56_80_4" or run == "resnet56_80_5":
                col = "b"
                ax[0,0].plot(epo, dict[model][run]["test_loss"], alpha = 0.15, color = col)
                ax[0,1].plot(epo, dict[model][run]["train_loss"], alpha = 0.15, color = col)
                ax[1,0].plot(epo, dict[model][run]["test_acc"], alpha = 0.15, color = col)
                ax[1,1].plot(epo, dict[model][run]["train_acc"], alpha = 0.15, color = col)
            elif run == "resnet56_90_1" or run == "resnet56_90_2" or run == "resnet56_90_3" or run == "resnet56_90_4" or run == "resnet56_90_5":
                col = "k"
                ax[0,0].plot(epo, dict[model][run]["test_loss"], alpha = 0.15, color = col)
                ax[0,1].plot(epo, dict[model][run]["train_loss"], alpha = 0.15, color = col)
                ax[1,0].plot(epo, dict[model][run]["test_acc"], alpha = 0.15, color = col)
                ax[1,1].plot(epo, dict[model][run]["train_acc"], alpha = 0.15, color = col)
        
    ax[1,1].legend(loc = "center left", bbox_to_anchor = (1,0))    

    plt.tight_layout()
    plt.show()

def plot_unpruned(unpruned_full):
    del unpruned_full["resnet20_90_avg"]
    del unpruned_full["resnet20_90_1"]
    del unpruned_full["resnet20_90_2"]
    del unpruned_full["resnet20_90_3"]
    del unpruned_full["resnet20_90_4"]
    del unpruned_full["resnet20_90_5"]
    del unpruned_full["resnet56_90_avg"]
    del unpruned_full["resnet56_90_1"]
    del unpruned_full["resnet56_90_2"]
    del unpruned_full["resnet56_90_3"]
    del unpruned_full["resnet56_90_4"]
    del unpruned_full["resnet56_90_5"]
    plot_average_test(unpruned_full, 50, "unpruned Network")

def plot_test_averages(all_in_one):
    plot_average_test(all_in_one["model_unpruned"],         50, "Unpruned network")
    plot_average_test(all_in_one["model_greedy_layer"],     25, "Greedy layer-wise pruning - fine-tuning after one-shot")
    plot_average_test(all_in_one["model_greedy_layer_it"],  50, "Greedy layer-wise pruning - iterative training")
    plot_average_test(all_in_one["model_gmp"] ,             25, "Global magnitude pruning - fine-tuning after one-shot")
    plot_average_test(all_in_one["model_gmp_it"] ,          50, "Global magnitude pruning - iterative training")
    plot_average_test(all_in_one["model_random"],           25, "Random pruning - fine-tuning after one-shot")
    plot_average_test(all_in_one["model_random_it"] ,       50, "Random pruning - iterative training")
    plot_average_test(all_in_one["model_LASSO"],            50, "LASSO pruning")

    plot_average_test(all_in_one["ticket_greedy_layer"],    50, "Ticket greedy layer-wise pruning - one-shot")
    plot_average_test(all_in_one["ticket_greedy_layer_it"], 50, "Ticket greedy layer-wise pruning - iterative")
    plot_average_test(all_in_one["ticket_gmp"],             50, "Ticket global magnitude pruning - one-shot")
    plot_average_test(all_in_one["ticket_gmp_it"],          50, "Ticket global magnitude pruning - iterative")
    plot_average_test(all_in_one["ticket_random"],          50, "Ticket global random pruning - one-shot")
    plot_average_test(all_in_one["ticket_random_it"],       50, "Ticket global random pruning - iterative")
    plot_average_test(all_in_one["ticket_LASSO"],           50, "Ticket LASSO pruning")

def plot2(all, epochs, title = "", models = [], data = [], test = True):
    epo = np.arange(epochs)+1
    x_ticks = np.arange(0, max(epo)+1, 10)
    fig, ax = plt.subplots(2, 3, sharey = "row")
    fig.suptitle(title)
    ax[0,0].set_title('First run', loc = 'left')
    ax[0,0].set_ylabel('Accuracy')
    ax[0,0].set_xticks(x_ticks)
    ax[0,0].grid()
    
    ax[0,1].set_title('Second run', loc = 'left')
    ax[0,1].set_xticks(x_ticks)
    ax[0,1].grid()

    ax[0,2].set_title('Third run', loc = 'left')
    ax[0,2].set_xlabel('Epochs')
    ax[0,2].set_xticks(x_ticks)
    ax[0,2].grid()
    
    ax[1,0].set_title('Forth run', loc = 'left')
    ax[1,0].set_xlabel('Epochs')
    ax[1,0].set_ylabel('Accuracy')
    ax[1,0].set_xticks(x_ticks)
    ax[1,0].grid()
    
    ax[1,1].set_title('Fifth run', loc = 'left')
    ax[1,1].set_xlabel('Epochs')
    ax[1,1].set_xticks(x_ticks)
    ax[1,1].grid()

    ax[1,2].remove()

    for model in models:
        if model == "model_unpruned":
            al = 1
        else: 
            al = 0.6
        ax[0,0].plot(epo, all[model]["resnet20_80_1"]["test_acc"], alpha = al, label = model)
        ax[0,1].plot(epo, all[model]["resnet20_80_2"]["test_acc"], alpha = al, label = model)
        ax[0,2].plot(epo, all[model]["resnet20_80_3"]["test_acc"], alpha = al, label = model)
        ax[1,0].plot(epo, all[model]["resnet20_80_4"]["test_acc"], alpha = al, label = model)
        ax[1,1].plot(epo, all[model]["resnet20_80_5"]["test_acc"], alpha = al, label = model)
                
    ax[1,1].legend(loc = "lower right", framealpha=1)   

    plt.tight_layout()
    plt.show()

def plot_tickets_vs_unpruned(all_in_one, all_tickets):
    plot(all_in_one, 50, "Comparison of tickets to the unpruned network", all_tickets, ["resnet20_80_avg","resnet20_90_avg","resnet56_80_avg","resnet56_90_avg"], False)
    #plot2(all_in_one, 50, "Searching for winning tickets: ResNet20 with {80%} sparsity",all_tickets, ["resnet20_80_1","resnet20_80_2","resnet20_80_3","resnet20_80_4", "resnet20_80_5"], False)
    #plot(all_in_one, 50, "Comparison of tickets for ResNet20 with {90%} sparsity", all_tickets, ["resnet20_90_avg"], False)
    #plot(all_in_one, 50, "Comparison of tickets for ResNet56 with {80%} sparsity", all_tickets, ["resnet56_80_avg"], False)
    #plot(all_in_one, 50, "Comparison of tickets for ResNet56 with {90%} sparsity", all_tickets, ["resnet56_90_avg"], False)

def mean_std(all_in_one):
    for model in all_in_one:
        for structure in all_in_one[model]:
            all_in_one[model][structure]["test_loss_mean"] = np.mean(all_in_one[model][structure]["test_loss"])
            all_in_one[model][structure]["test_loss_std"] = np.std(all_in_one[model][structure]["test_loss"])
            all_in_one[model][structure]["test_loss_max"] = [np.amax(all_in_one[model][structure]["test_loss"]), float(np.argmax(all_in_one[model][structure]["test_loss"]))]
            all_in_one[model][structure]["test_loss_min"] = [np.amin(all_in_one[model][structure]["test_loss"]), float(np.argmin(all_in_one[model][structure]["test_loss"]))]

            all_in_one[model][structure]["test_acc_mean"] = np.mean(all_in_one[model][structure]["test_acc"])
            all_in_one[model][structure]["test_acc_std"] = np.std(all_in_one[model][structure]["test_acc"])
            all_in_one[model][structure]["test_acc_max"] = [np.amax(all_in_one[model][structure]["test_acc"]), float(np.argmax(all_in_one[model][structure]["test_acc"]))]
            all_in_one[model][structure]["test_acc_min"] = [np.amin(all_in_one[model][structure]["test_acc"]), float(np.argmin(all_in_one[model][structure]["test_acc"]))]

            all_in_one[model][structure]["train_loss_mean"] = np.mean(all_in_one[model][structure]["train_loss"])
            all_in_one[model][structure]["train_loss_std"] = np.std(all_in_one[model][structure]["train_loss"])
            all_in_one[model][structure]["train_loss_max"] = [np.amax(all_in_one[model][structure]["train_loss"]), float(np.argmax(all_in_one[model][structure]["train_loss"]))]
            all_in_one[model][structure]["train_loss_min"] = [np.amin(all_in_one[model][structure]["train_loss"]), float(np.argmin(all_in_one[model][structure]["train_loss"]))]

            all_in_one[model][structure]["train_acc_mean"] = np.mean(all_in_one[model][structure]["train_acc"])
            all_in_one[model][structure]["train_acc_std"] = np.std(all_in_one[model][structure]["train_acc"])
            all_in_one[model][structure]["train_acc_max"] = [np.amax(all_in_one[model][structure]["train_acc"]), float(np.argmax(all_in_one[model][structure]["train_acc"]))]
            all_in_one[model][structure]["train_acc_min"] = [np.amin(all_in_one[model][structure]["train_acc"]), float(np.argmin(all_in_one[model][structure]["train_acc"]))]

def print_info(all_in_one, models = [], runs = [], test = True):
    for model in models:
        for run in runs:
            print("----------------------------------------------------------------------------------------")
            test_acc_mean = round(all_in_one[model][run]["test_acc_mean"],3)
            print(f"test_acc_mean for {model} in run {run} = {test_acc_mean}")

            test_acc_std = round(all_in_one[model][run]["test_acc_std"],3)
            print(f"test_acc_std for {model} in run {run} = {test_acc_std}")

            test_acc_max = [round(all_in_one[model][run]["test_acc_max"][0],3), all_in_one[model][run]["test_acc_max"][1]+1]
            print(f"test_acc_max for {model} in run {run} = {test_acc_max}")

            test_acc_min = all_in_one[model][run]["test_acc_min"]
            print(f"test_acc_min for {model} in run {run} = {test_acc_min}")

            final_test_acc = round(all_in_one[model][run]["test_acc"][(len(all_in_one[model][run]["test_acc"])-1)],3)
            print(f"final_test_acc for {model} in run {run} = {final_test_acc}\n")

            test_loss_mean = round(all_in_one[model][run]["test_loss_mean"],3)
            print(f"test_loss_mean for {model} in run {run} = {test_loss_mean}")

            test_loss_std = round(all_in_one[model][run]["test_loss_std"],3)
            print(f"test_loss_std for {model} in run {run} = {test_loss_std}")

            test_loss_max = all_in_one[model][run]["test_loss_max"]
            print(f"test_loss_max for {model} in run {run} = {test_loss_max}")

            test_loss_min = [round(all_in_one[model][run]["test_loss_min"][0],3), all_in_one[model][run]["test_loss_min"][1]+1]
            print(f"test_loss_min for {model} in run {run} = {test_loss_min}")

            final_test_loss = round(all_in_one[model][run]["test_loss"][(len(all_in_one[model][run]["test_loss"])-1)],3)
            print(f"final_test_loss for {model} in run {run} = {final_test_loss}\n")
            
            if test == False:
                train_acc_mean = all_in_one[model][run]["train_acc_mean"]
                print(f"train_acc_mean for {model} in run {run} = {train_acc_mean}")

                train_acc_std = all_in_one[model][run]["train_acc_std"]
                print(f"train_acc_std for {model} in run {run} = {train_acc_std}")

                train_acc_max = all_in_one[model][run]["train_acc_max"]
                print(f"train_acc_max for {model} in run {run} = {train_acc_max}")

                train_acc_min = all_in_one[model][run]["train_acc_min"]
                print(f"train_acc_min for {model} in run {run} = {train_acc_min}")

                final_train_acc = all_in_one[model][run]["train_acc"][(len(all_in_one[model][run]["train_acc"])-1)]
                print(f"final_train_acc for {model} in run {run} = {final_train_acc}\n")

                train_loss_mean = all_in_one[model][run]["train_loss_mean"]
                print(f"train_loss_mean for {model} in run {run} = {train_loss_mean}")

                train_loss_std = all_in_one[model][run]["train_loss_std"]
                print(f"train_loss_std for {model} in run {run} = {train_loss_std}")

                train_loss_max = all_in_one[model][run]["train_loss_max"]
                print(f"train_loss_max for {model} in run {run} = {train_loss_max}")

                train_loss_min = all_in_one[model][run]["train_loss_min"]
                print(f"train_loss_min for {model} in run {run} = {train_loss_min}")

                final_train_loss = all_in_one[model][run]["train_loss"][(len(all_in_one[model][run]["train_loss"])-1)]
                print(f"final_train_loss for {model} in run {run} = {final_train_loss}\n")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

def remove_unpruned(unpruned):
    del unpruned["resnet20_90_avg"]
    #del unpruned["resnet20_90_1"]
    #del unpruned["resnet20_90_2"]
    #del unpruned["resnet20_90_3"]
    #del unpruned["resnet20_90_4"]
    #del unpruned["resnet20_90_5"]
    del unpruned["resnet56_90_avg"]
    #del unpruned["resnet56_90_1"]
    #del unpruned["resnet56_90_2"]
    #del unpruned["resnet56_90_3"]
    #del unpruned["resnet56_90_4"]
    #del unpruned["resnet56_90_5"]

if __name__ == '__main__':
    main()