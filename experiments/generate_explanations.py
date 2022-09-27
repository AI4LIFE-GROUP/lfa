import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append("../code")
from models.functions_model_training import LinearRegressionNN, FFNNRegression, FFNNRegressionB, FFNNRegressionC, LogisticRegressionNN, FFNNClassification, FFNNClassificationB, FFNNClassificationC
from experiments.functions_load_model_and_data import load_data_and_model

import functions_create_explainers
from functions_explain_instances import explain_instances_array, explain_instances_grad

from functools import partial



#####load data + model
#using load_data_and_model()


#####create explainers given model_f (using functions from functions_create_explainers.py module)

def create_explainers_all(model_f, task=['regr', 'classif']):
	#approach: existing methods --> use captum library
	captum_lime = partial(functions_create_explainers.captum_lime, model_f=model_f, kernel_width=1) #lime
	captum_ks = partial(functions_create_explainers.captum_ks, model_f=model_f) #kernelshap
	captum_sg = partial(functions_create_explainers.captum_sg, model_f=model_f) #smoothgrad
	captum_ig = partial(functions_create_explainers.captum_ig, model_f=model_f) #integrated gradients
	captum_vg = partial(functions_create_explainers.captum_vg, model_f=model_f) #vanilla gradients
	captum_gxi = partial(functions_create_explainers.captum_gxi, model_f=model_f) #gradientxinput
	captum_oc = partial(functions_create_explainers.captum_oc, model_f=model_f) #occlusion

	#approach: local function approximation --> use framework
	meta_lime = functions_create_explainers.create_explainer_meta_lime(model_f) #lime
	meta_ks = functions_create_explainers.create_explainer_meta_ks(model_f) #kernelshap
	meta_sg_mse = functions_create_explainers.create_explainer_meta_sg_mse(model_f, epsilon=0.1) #smoothgrad (mse loss)
	meta_sg_gm = functions_create_explainers.create_explainer_meta_sg_gm(model_f, epsilon=0.1) #smoothgrad (gm loss)
	meta_sg_gm2 = functions_create_explainers.create_explainer_meta_sg_gm(model_f, epsilon=0.05)#vanilla gradients = smoothgrad limit
	meta_sg_gm3 = functions_create_explainers.create_explainer_meta_sg_gm(model_f, epsilon=0.01)
	meta_sg_gm4 = functions_create_explainers.create_explainer_meta_sg_gm(model_f, epsilon=0.2)
	meta_sg_gm5 = functions_create_explainers.create_explainer_meta_sg_gm(model_f, epsilon=0.3)
	meta_ig = functions_create_explainers.create_explainer_meta_ig(model_f, low=0) #integrated gradients
	meta_ig2 = functions_create_explainers.create_explainer_meta_ig(model_f, low=0.5) #gradientxinput = integrated gradients limit
	meta_ig3 = functions_create_explainers.create_explainer_meta_ig(model_f, low=0.65)
	meta_ig4 = functions_create_explainers.create_explainer_meta_ig(model_f, low=0.8)
	meta_ig5 = functions_create_explainers.create_explainer_meta_ig(model_f, low=0.95)
	meta_oc = functions_create_explainers.create_explainer_meta_oc(model_f) #occlusion

	if task=='classif':
		meta_sg_mse_logistic = functions_create_explainers.create_explainer_meta_sg_mse_logistic(model_f, epsilon=0.1)
		meta_sg_gm_logistic = functions_create_explainers.create_explainer_meta_sg_gm_logistic(model_f, epsilon=0.1)
		meta_ig_logistic = functions_create_explainers.create_explainer_meta_ig_logistic(model_f, low=0)

		meta_sg_gm2_logistic = functions_create_explainers.create_explainer_meta_sg_gm_logistic(model_f, epsilon=0.05)
		meta_sg_gm3_logistic = functions_create_explainers.create_explainer_meta_sg_gm_logistic(model_f, epsilon=0.01)
		meta_sg_gm4_logistic = functions_create_explainers.create_explainer_meta_sg_gm_logistic(model_f, epsilon=0.2)
		meta_sg_gm5_logistic = functions_create_explainers.create_explainer_meta_sg_gm_logistic(model_f, epsilon=0.3)

		meta_ig2_logistic = functions_create_explainers.create_explainer_meta_ig_logistic(model_f, low=0.50)
		meta_ig3_logistic = functions_create_explainers.create_explainer_meta_ig_logistic(model_f, low=0.65)
		meta_ig4_logistic = functions_create_explainers.create_explainer_meta_ig_logistic(model_f, low=0.8)
		meta_ig5_logistic = functions_create_explainers.create_explainer_meta_ig_logistic(model_f, low=0.95)


	#save explainers as dictionaries
	#set of methods that do not depend on n_perturb
	explainer_attr_dict_nonperturb = {
	    'captum_vg': captum_vg,
	    'captum_gxi': captum_gxi,
	    'captum_oc': captum_oc,
	}

	#set of methods that do depend on n_perturb
	explainer_attr_dict_npertrub = {
	    'captum_lime': captum_lime,
	    'captum_ks': captum_ks,
	    'captum_sg': captum_sg,
	    'captum_ig': captum_ig,
	    
	    'meta_lime': meta_lime.attribute,
	    'meta_ks': meta_ks.attribute,
	    'meta_sg_mse': meta_sg_mse.attribute,
	    'meta_sg_gm': meta_sg_gm.attribute,
	    
	    'meta_ig': meta_ig.attribute,
	    
	    'meta_oc': partial(meta_oc.attribute, explain_delta_f=True),
	    
	    'meta_sg_gm2': meta_sg_gm2.attribute,
	    'meta_sg_gm3': meta_sg_gm3.attribute,
	    'meta_sg_gm4': meta_sg_gm4.attribute,
	    'meta_sg_gm5': meta_sg_gm5.attribute,
	    
	    'meta_ig2': meta_ig2.attribute, 
	    'meta_ig3': meta_ig3.attribute, 
	    'meta_ig4': meta_ig4.attribute, 
	    'meta_ig5': meta_ig5.attribute,
	}


	if task=='classif':
		explainer_attr_dict_npertrub_logistic = {
		    'meta_sg_mse_logistic': meta_sg_mse_logistic.attribute,
		    'meta_sg_gm_logistic': meta_sg_gm_logistic.attribute,
		    'meta_ig_logistic': meta_ig_logistic.attribute,

		    'meta_sg_gm2_logistic': meta_sg_gm2_logistic.attribute,
		    'meta_sg_gm3_logistic': meta_sg_gm3_logistic.attribute,
		    'meta_sg_gm4_logistic': meta_sg_gm4_logistic.attribute,
		    'meta_sg_gm5_logistic': meta_sg_gm5_logistic.attribute,
		    
		    'meta_ig2_logistic': meta_ig2_logistic.attribute, 
		    'meta_ig3_logistic': meta_ig3_logistic.attribute, 
		    'meta_ig4_logistic': meta_ig4_logistic.attribute, 
		    'meta_ig5_logistic': meta_ig5_logistic.attribute,
		}
		explainer_attr_dict_npertrub  = {**explainer_attr_dict_npertrub, **explainer_attr_dict_npertrub_logistic} #combine dictionaries

	return explainer_attr_dict_nonperturb, explainer_attr_dict_npertrub





#########run experiments (generate explanations)

model_info_list = [
	['who', 'regr', 'linear'], #regression
	['who', 'regr', 'ffnnA'],
	['who', 'regr', 'ffnnB'],
	['who', 'regr', 'ffnnC'],

	['heloc', 'classif', 'logistic'], #classification
	['heloc', 'classif', 'ffnnA'],
	['heloc', 'classif', 'ffnnB'],
	['heloc', 'classif', 'ffnnC']
	]


if __name__ == "__main__":

	#for each model, generate explanations using all explainers (existing method + lfa framework)
	for model_info in model_info_list:

		data_name = model_info[0]
		task = model_info[1]
		model_name = model_info[2]
		print(f'******************** MODEL: {task}_{data_name}_{model_name} ********************')

		###1. load data and model
		X, y, model_f = load_data_and_model(data_name, model_name, task)

		###2. create explainers
		explainer_attr_dict_nonperturb, explainer_attr_dict_npertrub = create_explainers_all(model_f, task)

		###3. generate explanations
		#shared arguments
		n_subset = X.size(0) #all data points in X 
		instances = X[0:n_subset, :] #X[0:50, :] #[num_points, num_features]
		seed=12345
		debug = False
		results_folder = f'explanations_{task}_{data_name}_{model_name}_n{n_subset}'

		#methods that depend on n_perturb
		n_perturbs_list = [1000]
		explain_instances_array(instances, n_perturbs_list, explainer_attr_dict_npertrub, seed, results_folder, debug)

		#methods that do not depend on n_perturb
		explain_instances_grad(instances, explainer_attr_dict_nonperturb, results_folder)












