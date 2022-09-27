import torch
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

import os

import sys
sys.path.append("../code")
from experiments.functions_load_model_and_data import load_data_and_model

from functions_analysis import calc_L1_norm, calc_cos_dist, create_df_for_seaborn_g2, calc_metric_dict_g2

import argparse

parser = argparse.ArgumentParser(description='exp1')
parser.add_argument('--exp_folder', type = str, help='path to experiment folder')
parser.add_argument('--data_name', type = str, help='dataset name')
parser.add_argument('--model_name', type = str, help='model name')
parser.add_argument('--task', type = str, choices=['regr', 'classif'], help='prediction task (regression or classification)')
parser.add_argument('--top_or_bottom_k', type = str, choices=['topk', 'bottomk'], help='whether to perturb top k or bottom k features in perturbation test')


#parse arguments
args = parser.parse_args()

exp_folder = args.exp_folder
data_name = args.data_name
model_name = args.model_name
task = args.task
top_or_bottom_k = args.top_or_bottom_k

results_folder = f'explanations_{task}_{data_name}_{model_name}_n100'



# #regression setup
# exp_folder = './experiments/results_saved'
# data_name = 'who'
# model_name = 'ffnnA' #regr: linear, ffnnA, ffnnB, ffnnC
# task = 'regr'
# results_folder = f'explanations_{task}_{data_name}_{model_name}_n100'

# #classification setup
# exp_folder = '../experiments/results_saved/'
# data_name= 'heloc'
# model_name = 'ffnnA' #classif: logistic, ffnnA, ffnnB, ffnnC
# task = 'classif'
# results_folder = f'explanations_{task}_{data_name}_{model_name}_n100'



#####load model_f, X, and y
X, y, model_f = load_data_and_model(data_name, model_name, task)


#####load attributions
#use captum methods (same as lfa)

####methods based on n_perturb
method_names = [
    'captum_lime',
    'captum_ks',
    'captum_sg',
    'captum_ig',
]
n_perturb = 1000

#load attributions
attr_dict = {}
for method_name in method_names:
    #load feature attributions generated based on each n_perturb
    attr_tensor = torch.load(f'{exp_folder}/{results_folder}/{method_name}/{method_name}_{n_perturb}/attr_{method_name}_{n_perturb}.pth')
    #save dictionary
    attr_dict[method_name] = attr_tensor
    

###methods not based on n_perturb
method_names = [
    'captum_vg',
    'captum_gxi',
    'captum_oc',
]

#load attributions
for method_name in method_names:
    #load feature attributions generated based on each n_perturb
    attr_tensor = torch.load(f'{exp_folder}/{results_folder}/{method_name}/attr_{method_name}.pth').detach()
    #save dictionary
    attr_dict[method_name] = attr_tensor



#####perturbation test calculations

#perturbation test for one method and one k value
#generates ONE perturbation for binary perturbations and MANY perturbation for continuous perturbations

def perturb_test_1method_1k(model_f, g_weights, X, k, perturb_type=['binary', 'continuous'], n_perturb_continuous=100, top_or_bottom_k=['topk', 'bottomk']):
    #get bottom-k or top-k features predicted by g_weights
    if top_or_bottom_k == 'bottomk':
        bottomk_id = torch.argsort(g_weights.abs(), descending=False, dim=1)[:, 0:k]
    if top_or_bottom_k == 'topk':
        bottomk_id = torch.argsort(g_weights.abs(), descending=True, dim=1)[:, 0:k]  #'bottomk_id' refers to 'topk' (re-using code)

    ###create perturbations and calculate change in model predictions
    n_data_points = X.size(0)
    row_idx = torch.arange(n_data_points).unsqueeze(1) #TODO: may differ for images
    X_bottomk_perturb = X.detach().clone() #get a copy of X

    #binary perturbations: set bottom-k features equal to zero --> one perturbation
    if perturb_type=='binary':
        X_bottomk_perturb[row_idx, bottomk_id] = 0

        #get model predictions for X (original data points)
        #get model predictions for X_bottomk_perturb (data points with bottom-k features turned off)
        #calculate magnitude of prediction difference
        pred_delta = (model_f(X) - model_f(X_bottomk_perturb)).abs() #[n_data_points in g_weights, 1] ([100, 1])
        pred_delta = pred_delta.detach().numpy()

    #continuous perturbations: add gaussian noise to bottom-k features --> many perturbations
    if perturb_type=='continuous':

        pred_delta = torch.zeros((n_data_points, k)) #store change in model predictions for each data point

        for i in range(n_perturb_continuous):
            #generate one perturbation for each data point 
            noise = torch.randn_like(X[row_idx, bottomk_id]) * 0.1 #perturbation = x + Normal(0, epsilon^2), epsilon=0.1
            X_bottomk_perturb[row_idx, bottomk_id] = X[row_idx, bottomk_id] + noise #[n_data_points, k]

            #get model predictions for X (original data points)
            #get model predictions for X_bottomk_perturb (data points with bottom-k features turned off)
            #calculate magnitude of prediction difference
            pred_delta_one_perturb = (model_f(X) - model_f(X_bottomk_perturb)).abs() #[n_data_points, 1] ([100, 1])
            pred_delta += pred_delta_one_perturb.detach().numpy()

        #for each data point, take the average of the perturbations
        pred_delta = pred_delta/n_perturb_continuous #mean over n_perturb_continuous

    #return metric mean, sem, distr 
    return pred_delta.mean(), sem(pred_delta)[0], pred_delta #mean over n_data_points



def perturb_test(attr_dict, n_features, model_f, X, perturb_type=['binary', 'continuous']):
    perturb_test_dict = {}

    for method_name, attr_tensor in attr_dict.items():
        perturb_test_dict_method = {}
        for k in range(1, n_features+1):
            metric_mean, metric_sem, metric_distr = perturb_test_1method_1k(model_f=model_f, g_weights=attr_tensor, X=X, k=k, perturb_type=perturb_type, top_or_bottom_k=top_or_bottom_k)
            perturb_test_dict_method[k] = metric_mean, metric_sem

        perturb_test_dict[method_name] = perturb_test_dict_method
    
    return perturb_test_dict


results_dict_binary = perturb_test(attr_dict=attr_dict, 
                                        n_features=11, 
                                        model_f=model_f,
                                        X=X, 
                                        perturb_type='binary')

results_dict_continuous = perturb_test(attr_dict=attr_dict, 
                                        n_features=11, 
                                        model_f=model_f,
                                        X=X, 
                                        perturb_type='continuous')


#####perturbation tests, plots

methods_subset = ['captum_lime', 'captum_ks', 'captum_oc', 'captum_ig', 'captum_gxi',  'captum_sg', 'captum_vg'] #all

legend_label_dict = {'captum_lime': 'LIME',
                     'captum_ks': 'KernelSHAP',
                     'captum_oc': 'Occlusion',
                     'captum_ig': 'Integrated Gradients',
                     'captum_gxi': 'Gradient x Input',
                     'captum_sg': 'SmoothGrad',
                     'captum_vg': 'Vanilla Gradients',
                    }

#color and shape
point_dict = {'captum_lime': ['o', 'cornflowerblue'],
              'captum_ks': ['^', 'cornflowerblue'],
              'captum_oc': ['s', 'cornflowerblue'],
              'captum_ig': ['p', 'cornflowerblue'],
              'captum_gxi': ['*', 'cornflowerblue'],
              'captum_sg': ['o', 'lightsalmon'],
              'captum_vg': ['^', 'lightsalmon'],
             }


plot_folder = f'analysis/figures/exp3_{top_or_bottom_k}'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)


#plot
torch.manual_seed(12345)
np.random.seed(12345)

fig, axes = plt.subplots(1, 2, figsize =(13, 4))

###plot binary perturbations
for method_name in methods_subset:
    
    metric_dict_method = results_dict_binary[method_name] #use results_dict_binary
    
    ks = list(metric_dict_method.keys())
    mean_sem_tup_list = list(metric_dict_method.values())
    means, sems = zip(*mean_sem_tup_list)
    
    axes[0].errorbar(ks, means, yerr=sems, alpha=0.5,
                     fmt=point_dict[method_name][0], color=point_dict[method_name][1], label=f'{legend_label_dict[method_name]}')
    axes[0].legend(loc='best')
    axes[0].set(xlabel='$k$', ylabel='Absolute Difference of Model Prediction', title='Binary Perturbations')
    
###remove error bar in legend
# get handles
handles, labels = axes[0].get_legend_handles_labels()
# remove the errorbars
handles = [h[0] for h in handles]
# use them in the legend
axes[0].legend(handles, labels, loc='best')

###plot continuous perturbations
for method_name in methods_subset:
    
    metric_dict_method = results_dict_continuous[method_name] #use results_dict_continuous
    
    ks = list(metric_dict_method.keys())
    mean_sem_tup_list = list(metric_dict_method.values())
    means, sems = zip(*mean_sem_tup_list)
    
    axes[1].errorbar(ks, means, yerr=sems, alpha=0.5,
                     fmt=point_dict[method_name][0], color=point_dict[method_name][1], label=f'{legend_label_dict[method_name]}')
    axes[1].legend(loc='best') #turn off legend
    axes[1].get_legend().remove()
    axes[1].set(xlabel='$k$', ylabel='Absolute Difference of Model Prediction', title='Continuous Perturbations')
    
    plot_path = f'{plot_folder}/exp3_{data_name}_{model_name}.png'
    plt.savefig(plot_path, facecolor='white', transparent=False, dpi=1000);

