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

import argparse

parser = argparse.ArgumentParser(description='exp1')
parser.add_argument('--exp_folder', type = str, help='path to experiment folder')
parser.add_argument('--data_name', type = str, help='dataset name')
parser.add_argument('--model_name', type = str, help='model name')
parser.add_argument('--task', type = str, choices=['regr', 'classif'], help='prediction task (regression or classification)')


#parse arguments
args = parser.parse_args()

exp_folder = args.exp_folder
data_name = args.data_name
model_name = args.model_name
task = args.task

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
    




#####heatmap calculations

def calculate_heatmap_matrix(captum_attr_dict, meta_attr_dict, method_idx_dict, calc_metric_fn):
    
    #function
    n_methods = len(method_idx_dict)
    matrix = torch.zeros([n_methods, n_methods])


    #fill in matrix by calculating average metric value
    for method_name_captum, attr_captum in captum_attr_dict.items():
        for method_name_meta, attr_meta in meta_attr_dict.items():
            #calculate average metric value
            avg_metric = calc_metric_fn(attr_captum, attr_meta).mean()

            #save value to matrix
            row_idx = method_idx_dict[method_name_captum] #row = captum methods
            col_idx = method_idx_dict[method_name_meta] #column = meta methods

            matrix[row_idx, col_idx] = avg_metric
            
    return matrix


captum_attr_dict = {'LIME': attr_g2['captum_lime'],
                    'KernelSHAP': attr_g2['captum_ks'],
                    'Occlusion': attr_g2['captum_oc'], 
                    'Integrated Gradients': attr_g2['captum_ig'],
                    'Gradient x Input': attr_g2['captum_gxi'],
                    'Vanilla Gradient': attr_g2['captum_vg'],
                    'SmoothGrad': attr_g2['captum_sg'],
                   }

meta_attr_dict = {'LIME': attr_g2['meta_lime'],
                  'KernelSHAP': attr_g2['meta_ks'],
                  'Occlusion': attr_g2['meta_oc'], 
                  'Integrated Gradients': attr_g2['meta_ig'],
                  'Gradient x Input': attr_g2['meta_ig5'], #ig --> gxi, meta_ig5 has alpha=0.95 (largest alpha)
                  'Vanilla Gradient': attr_g2['meta_sg_gm3'], #sg --> vg, meta_sg_gm3 has sigma=0.01 (smallest sigma)
                  'SmoothGrad': attr_g2['meta_sg_gm'],
                 }

method_idx_dict = {'LIME': 0,
                   'KernelSHAP': 1,
                   'Occlusion': 2,
                   'Integrated Gradients': 3,
                   'Gradient x Input': 4,
                   'Vanilla Gradient': 5,
                   'SmoothGrad': 6,
                  }

#calculate matrix for L1
matrix_L1 = calculate_heatmap_matrix(captum_attr_dict, meta_attr_dict, method_idx_dict, calc_metric_fn=calc_L1_norm)
#calculate matrix for cosine distance
matrix_cd = calculate_heatmap_matrix(captum_attr_dict, meta_attr_dict, method_idx_dict, calc_metric_fn=calc_cos_dist)


#####plot heatmap

plot_folder = 'analysis/figures/exp1'
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

cmap = sns.color_palette('vlag', as_cmap=True) #diverging colormap
# labels = list(method_idx_dict.keys())
labels = ['LIME', 'KernelSHAP', 'Occlusion', ' Integrated\nGradients', 'Gradient\nx Input', 'Vanilla\nGradients', 'SmoothGrad']


#heatmap
fig, axes = plt.subplots(1, 2, figsize = (15, 7))
sns.heatmap(matrix_L1, cmap=cmap, 
            xticklabels=labels, yticklabels=labels, annot=True, fmt='.4f',
            square=True, linewidths=.5, cbar_kws={'shrink': 0.83}, ax=axes[0])
axes[0].set_title('L1 Distance')

sns.heatmap(matrix_cd, cmap=cmap, 
            xticklabels=labels, yticklabels=labels, annot=True, fmt='.4f',
            square=True, linewidths=.5, cbar_kws={'shrink': 0.83}, ax=axes[1])
axes[1].set_title('Cosine Distance')

for ax in axes: 
    ax.set_xlabel('Local Function Approximation (LFA) Framework')
    ax.set_ylabel('Existing Method')
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# fig.suptitle('Average metric')
fig.tight_layout()
plot_path = f'{plot_folder}/exp1_{data_name}_{model_name}_heatmap.png'
plt.savefig(plot_path, facecolor='white', transparent=False, dpi=1000)


#####boxplot calculations


###calculate metrics for g2: methods with one/no 'n_perturb' values

method_pairs_g2 = {'0.3': ('captum_vg', 'meta_sg_gm5'), #sg --> vg: note the change in order
                   '0.2': ('captum_vg', 'meta_sg_gm4'),
                   '0.1': ('captum_vg', 'meta_sg_gm'),
                   '0.05': ('captum_vg', 'meta_sg_gm2'),
                   '0.01': ('captum_vg', 'meta_sg_gm3'),
                   
                   '0': ('captum_gxi', 'meta_ig'), #ig --> gxi
                   '0.50': ('captum_gxi', 'meta_ig2'),
                   '0.65': ('captum_gxi', 'meta_ig3'),
                   '0.80': ('captum_gxi', 'meta_ig4'),
                   '0.95': ('captum_gxi', 'meta_ig5'),
                  }

g2_L1 = calc_metric_dict_g2(method_pairs_g2, attr_dict=attr_g2, calc_metric_fn=calc_L1_norm) #L1 norm
g2_cd = calc_metric_dict_g2(method_pairs_g2, attr_dict=attr_g2, calc_metric_fn=calc_cos_dist) #cosine distance


#####plot boxplots

def boxplot_g2(L1_dict, cd_dict, method_subset, conv_method=['vg', 'gxi']):
    L1_df = create_df_for_seaborn_g2(L1_dict)
    cd_df = create_df_for_seaborn_g2(cd_dict)
    
    fig, axes = plt.subplots(1, 2, figsize = (7, 5.5))
    
    #L1 norm 
    sns.boxplot(x='method', y='metric', data=L1_df[L1_df['method'].isin(method_subset)], ax=axes[0], color='cornflowerblue')
    axes[0].set_ylabel('L1 Distance')        
    
    #cosine distance
    sns.boxplot(x='method', y='metric', data=cd_df[cd_df['method'].isin(method_subset)], ax=axes[1], color='cornflowerblue')
    axes[1].set_ylabel('Cosine Distance')
    
    if conv_method == 'vg':
        axes[0].set_title('SmoothGrad vs.\nVanilla Gradients')
        axes[1].set_title('SmoothGrad vs.\nVanilla Gradients')
        for ax in axes: 
            ax.set_xlabel('Standard Deviation ($\sigma$) of Noise (N(0, $\sigma^2$))\nfor SmoothGrad')
    if conv_method == 'gxi':
        axes[0].set_title('Integrated Gradients vs.\nGradient x Input')
        axes[1].set_title('Integrated Gradients vs.\nGradient x Input')
        for ax in axes: 
            ax.set_xlabel('Lowerbound ($a$) of Noise (Uniform($a$, 1))\nfor Integrated Gradients')
    
    fig.tight_layout()
    plot_path = f'{plot_folder}/exp1_{data_name}_{model_name}_{conv_method}.png'
    plt.savefig(plot_path, facecolor='white', transparent=False, dpi=1000);


#sg -> vg
method_subset = ['0.3', '0.2', '0.1', '0.05', '0.01'] 
boxplot_g2(g2_L1, g2_cd, method_subset, conv_method='vg')

#ig -> gxi
method_subset = ['0', '0.50', '0.65', '0.80', '0.95'] #ig -> gxi
boxplot_g2(g2_L1, g2_cd, method_subset, conv_method='gxi')






