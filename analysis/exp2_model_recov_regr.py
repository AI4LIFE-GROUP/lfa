#!/usr/bin/python

import torch
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

import sys
sys.path.append("../code")
from experiments.functions_load_model_and_data import load_data_and_model

from functions_analysis import calc_L1_norm, calc_cos_dist, create_df_for_seaborn_g2, calc_metric_dict_g2


#####regression setup
exp_folder = './experiments/results'
data_name = 'who'
model_name = 'linear' #only this model
task = 'regr'
results_folder = f'explanations_{task}_{data_name}_{model_name}_n100'


#####load model and data
#load model_f, X, and y
X, y, model_f = load_data_and_model(data_name, model_name, task)


#####load attributions

###methods with one 'n_perturb' value

#set arguments
method_names_g2 = [
    'captum_lime',
    'captum_ks',
    'captum_sg',
    'captum_ig',

    'meta_lime',
    'meta_ks',
    
    'meta_sg_mse',
    'meta_sg_gm',
    
    'meta_ig',
    
    'meta_sg_gm', 
    'meta_sg_gm2',
    'meta_sg_gm3',
    'meta_sg_gm4',
    'meta_sg_gm5',
    
    'meta_ig',
    'meta_ig2',
    'meta_ig3',
    'meta_ig4',
    'meta_ig5',
    
    'meta_oc',
]

n_perturb = 1000

#load attributions
attr_g2 = {}
for method_name in method_names_g2:
    #load feature attributions generated based on each n_perturb
    attr_tensor = torch.load(f'{exp_folder}/{results_folder}/{method_name}/{method_name}_{n_perturb}/attr_{method_name}_{n_perturb}.pth')
    #save dictionary
    attr_g2[method_name] = attr_tensor

#attr_g2 = {'method_name': attr_tensor}


###methods with no 'n_perturb' value

#set arguments
method_names_g3 = [
    'captum_vg',
    'captum_gxi',
    'captum_oc',
]

#load attributions
for method_name in method_names_g3:
    #load feature attributions generated based on each n_perturb
    attr_tensor = torch.load(f'{exp_folder}/{results_folder}/{method_name}/attr_{method_name}.pth').detach()
    #save dictionary
    attr_g2[method_name] = attr_tensor
    

#####boxplot calculations

#create matrix with model weights (use as attrB argument for calculate_L1_norm())
model_f_weights = model_f.linear.weight.detach() #[1, n_features]
n_points = X.size(0) 

model_f_weights_matrix = model_f_weights.repeat(n_points, 1) #[n_points, n_features]
model_f_weights_x_input_matrix = model_f_weights_matrix * X #[n_points, n_features]


def calc_metric_dict_modelrec_g2(attr_g2, recover_matrix, calc_metric_fn):
    
    metric_dict = {}

    for method_name, attr_tensor in attr_g2.items():
        attrA = attr_tensor #attributions
        attrB = recover_matrix #model weights
        metric_distr = calc_metric_fn(attrA, attrB)
        metric_dict[method_name] = metric_distr
    
    return metric_dict

#format
#attr_g2 = {method_name: attr_tensor} 
#model_f_weights_matrix = stacked rows of model weights, [n_points, n_features]
#metric_dict = {method_name: metric_distr} 



###methods with one/no 'n_perturb' values

g2_L1_w = calc_metric_dict_modelrec_g2(attr_g2, model_f_weights_matrix, calc_metric_fn=calc_L1_norm) #L1 norm
g2_cd_w = calc_metric_dict_modelrec_g2(attr_g2, model_f_weights_matrix, calc_metric_fn=calc_cos_dist) #cosine distance

g2_L1_wxi = calc_metric_dict_modelrec_g2(attr_g2, model_f_weights_x_input_matrix, calc_metric_fn=calc_L1_norm) #L1 norm
g2_cd_wxi = calc_metric_dict_modelrec_g2(attr_g2, model_f_weights_x_input_matrix, calc_metric_fn=calc_cos_dist) #cosine distance


#####plot boxplots

plot_folder = 'analysis/figures/exp2'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)


def boxplot_recovery(g2_L1, g2_cd, method_subset, recovery_quantity, xlabel_dict, approach):
    #format data for plotting
    L1_df = create_df_for_seaborn_g2(g2_L1)
    cd_df = create_df_for_seaborn_g2(g2_cd)

    ###plot metric distributions
    fig, axes = plt.subplots(1, 2, figsize = (13, 3))

    #L1
    sns.boxplot(x='method', y='metric', order=method_subset, data=L1_df[L1_df['method'].isin(method_subset)], ax=axes[0], color='cornflowerblue')
    axes[0].set_title(f'Weights of $g$ vs. {recovery_quantity}')
    axes[0].set_xlabel('Explanation Method')
    axes[0].set_ylabel('L1 Distance')

    #cosine distance
    sns.boxplot(x='method', y='metric', order=method_subset, data=cd_df[cd_df['method'].isin(method_subset)], color='cornflowerblue')
    axes[1].set_title(f'Weights of $g$ vs. {recovery_quantity}')
    axes[1].set_xlabel('Explanation Method')
    axes[1].set_ylabel('Cosine Distance')

    for ax in axes: 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xticklabels([xlabel_dict[method] for method in method_subset])
        
    recovery_quantity_short='w' if recovery_quantity=='Weights of $f$' else 'wxi'
    plot_path = f'{plot_folder}/exp2_{data_name}_{model_name}_{recovery_quantity_short}_{approach}.png'
    plt.savefig(plot_path, facecolor='white', transparent=False, dpi=1000, bbox_inches='tight');


###captum implementation
#plot recovery of weights

method_subset = ['captum_lime', 'captum_ks', 'captum_oc', 'captum_ig', 'captum_gxi',  'captum_sg', 'captum_vg'] #captum methods
xlabel_dict = {'captum_lime': 'LIME', 
               'captum_ks': 'KernelSHAP', 
               'captum_oc': 'Occlusion', 
               'captum_ig': 'Integrated\nGradients', 
               'captum_gxi': 'Gradient\nx Input',  
               'captum_sg': 'SmoothGrad', 
               'captum_vg': 'Vanilla\nGradients'}
approach = 'existing'
recovery_quantity='Weights of $f$'
boxplot_recovery(g2_L1_w, g2_cd_w, method_subset, recovery_quantity, xlabel_dict, approach)

#plot recovery of weight*input
recovery_quantity='Weights of $f$ x Input'
boxplot_recovery(g2_L1_wxi, g2_cd_wxi, method_subset, recovery_quantity, xlabel_dict, approach)


###lfa implementation

#plot recovery of weights

method_subset = ['meta_lime', 'meta_ks', 'meta_oc', 'meta_ig', 'meta_ig5',  'meta_sg_gm', 'meta_sg_gm3'] #meta-algo methods
xlabel_dict = {'meta_lime': 'LIME', 
               'meta_ks': 'KernelSHAP', 
               'meta_oc': 'Occlusion', 
               'meta_ig': 'Integrated\nGradients', 
               'meta_ig5': 'Gradient\nx Input',  
               'meta_sg_gm': 'SmoothGrad', 
               'meta_sg_gm3': 'Vanilla\nGradients'}

approach = 'lfa'
recovery_quantity='Weights of $f$'
boxplot_recovery(g2_L1_w, g2_cd_w, method_subset, recovery_quantity, xlabel_dict, approach)

#plot recovery of weight*input
recovery_quantity='Weights of $f$ x Input'
boxplot_recovery(g2_L1_wxi, g2_cd_wxi, method_subset, recovery_quantity, xlabel_dict, approach)



