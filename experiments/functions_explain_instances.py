import torch
from datetime import datetime
import os

import sys
sys.path.append("../code")
import lfa.config as config


def explain_instances_array(instances, n_perturbs_list, explainer_attr_dict, seed, results_folder, debug=False):
    #explain test instances -- get feature attributions
    #save attributions for each method-perturbation_size tensor

    #function start
    torch.manual_seed(seed)
    n = instances.shape[0] #number of data points in 'instances'

    for method_name, attr_calculator in explainer_attr_dict.items():
        print('-'*40)
        print(f'{method_name}')
        attr_dict = {}

        ###CALCULATE RAW ATTRIBUTIONS
        for n_perturb in n_perturbs_list:
            now = datetime.now()
            print(f'    n_perturb = {n_perturb}')
            print(f'        start: {now}')

            #create directory for results
            results_folder_path = f'experiments/results/{results_folder}/{method_name}/{method_name}_{n_perturb}'
            if not os.path.exists(results_folder_path):
                os.makedirs(results_folder_path)

            #create directory for log files
            log_folder_path = f'{results_folder_path}/log_files'
            if debug:
                if not os.path.exists(log_folder_path):
                    os.makedirs(log_folder_path)  

            attr_list = []

            #function to create log file and calculate attributions for each data point
            def create_log_get_attr(log_folder_path, method_name, n_perturb, i, attr_calculator, debug):
                if debug:
                    #create logger
                    log_file = f'{log_folder_path}/{method_name}_{n_perturb}_{i}.log'
                    config.logger = config.custom_logger(log_file)
                #create attributions (tensor)
                attr_bias_tup = attr_calculator(x=instances[[i], :], target_class=None, n_perturb=n_perturb)
                return attr_bias_tup

            #get attributions for all instances based on 'n_perturb' perturbations --> list of tensors
            attr_bias_tup_list = [create_log_get_attr(log_folder_path, method_name, n_perturb, i, attr_calculator, debug) for i in range(0, n)]
            attr_tup, bias_tup = list(zip(*attr_bias_tup_list))

            #create tensor of attributions ([num_points, num_features]) using list of tensors
            attr_tensor = torch.cat(list(attr_tup))
            torch.save(attr_tensor, f'{results_folder_path}/attr_{method_name}_{n_perturb}.pth')
            bias_tensor = torch.cat(list(bias_tup))
            torch.save(bias_tensor, f'{results_folder_path}/bias_{method_name}_{n_perturb}.pth')

            now = datetime.now()
            print(f'        end: {now}')
    
    print('-'*40)
    print(f'Complete!')
            

            
#explain test instances -- get feature attributions
#save attributions for each method-perturbation_size-data_point tensor
def explain_instances_indiv(instances, n_perturbs_list, explainer_attr_dict, seed, results_folder, start_datapt, end_datapt):
    #function start
    torch.manual_seed(seed)
    n = instances.shape[0] #number of data points in 'instances'

    for method_name, attr_calculator in explainer_attr_dict.items():
        print('-'*40)
        print(f'{method_name}, start')
        attr_dict = {}

        ###CALCULATE RAW ATTRIBUTIONS
        for n_perturb in n_perturbs_list:
            now = datetime.now()
            print(f'n_perturb = {n_perturb}')
            print(f'start: {now}')

            #create directory for results
            results_folder_path = f'experiments/results/{results_folder}/{method_name}/{method_name}_{n_perturb}'
            if not os.path.exists(results_folder_path):
                os.makedirs(results_folder_path)

            #create directory for individual attributions files
            attr_indiv_folder_path = f'{results_folder_path}/attr_indiv'
            if not os.path.exists(attr_indiv_folder_path):
                os.makedirs(attr_indiv_folder_path)

            #create directory for individual attributions files
            bias_indiv_folder_path = f'{results_folder_path}/bias_indiv'
            if not os.path.exists(bias_indiv_folder_path):
                os.makedirs(bias_indiv_folder_path)

            #create directory for log files
            log_folder_path = f'{results_folder_path}/log_files'
            if not os.path.exists(log_folder_path):
                os.makedirs(log_folder_path)  

            attr_list = []
            for i in range(start_datapt, end_datapt):
                #create logger
                log_file = f'{log_folder_path}/{method_name}_{n_perturb}_{i}.log'
                config.logger = config.custom_logger(log_file)

                #create attributions (tensor)
                attr_tensor_i, bias_i = attr_calculator(instances[[i], :], target_class=None, n_perturb=n_perturb)

                now = datetime.now()
                print(f'   {i}: {now}')

                #save attributions for one datapoint (tensor [1, 12])
                torch.save(attr_tensor_i, f'{attr_indiv_folder_path}/attr_{method_name}_{n_perturb}_{i}.pth')
                torch.save(bias_i, f'{bias_indiv_folder_path}/bias_{method_name}_{n_perturb}_{i}.pth')

                attr_list.append(attr_tensor_i)

            # save attributions for all datapoints (tensor [n, 12])
            # attr_tensor = torch.cat(attr_list)
            # torch.save(attr_tensor, f'{results_folder_path}/attr_{method_name}_{n_perturb}.pth')

            now = datetime.now()
            print(f'end: {now}')
            
    print('-'*40)
    print(f'Complete!')


#explain test instances -- get feature attributions
#save attribution tensor for each method (each method is n_perturb-independent)
def explain_instances_grad(instances, explainer_attr_dict, results_folder):
    
    for method_name, attr_calculator in explainer_attr_dict.items():
        print(f'method = {method_name}')
        now = datetime.now()
        print(f'    start: {now}')

        #create tensor of attributions ([num_points, num_features])
        attr_tensor = attr_calculator(x=instances, target_class=None)
        
        if not os.path.exists(f'experiments/results/{results_folder}/{method_name}'):
            os.makedirs(f'experiments/results/{results_folder}/{method_name}')
        
        torch.save(attr_tensor, f'experiments/results/{results_folder}/{method_name}/attr_{method_name}.pth')

        now = datetime.now()
        print(f'    end: {now}')
        
    print('-'*40)
    print(f'Complete!')