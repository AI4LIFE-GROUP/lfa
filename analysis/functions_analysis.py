import torch
import pandas as pd
import numpy as np


###calculate L1 norm
def calc_L1_norm(attrA, attrB, unit_norm=False):
    #attrA: tensor [n_points, n_features]
    #attrB: tensor [n_points, n_features]
    if unit_norm:
        attrA = unit_norm_attr_tensor(attrA)
        attrB = unit_norm_attr_tensor(attrB)
        
    L1_norm = torch.linalg.norm(attrA - attrB, ord=1, dim=1) #[n_points]
    L1_norm = np.round(L1_norm.numpy(), decimals=6) + 0 #round to 6th decimal point, add zero to remove -0
    L1_norm = torch.Tensor(L1_norm)

    return L1_norm



###calculate cosine similarity
cos_sim = torch.nn.CosineSimilarity(dim=1)

def calc_cos_dist(attrA, attrB):
    
    cos_dist = 1 - cos_sim(attrA, attrB)
    cos_dist = np.round(cos_dist.numpy(), decimals=6) + 0 #prevent negative values due to floating point error
    cos_dist = torch.Tensor(cos_dist)
    
    return cos_dist



###format data into dataframe for seaborn boxplot
def create_df_for_seaborn_g2(metric_dict):
    #metric_dict = {name: L1_distribution}
    
    df_all = pd.DataFrame([], columns=['metric', 'method']) #dataframe for seaborn boxplot

    for method_name, metric_distr in metric_dict.items():
        df = pd.DataFrame(metric_distr.numpy(), columns=['metric'])
        df['method'] = method_name
        df_all = pd.concat([df_all, df])
        
    return df_all



###calculate dictionary of metric distributions
def calc_metric_dict_g2(method_pairs_g2, attr_dict, calc_metric_fn):
    
    metric_dict = {}

    for name, (name_captum, name_meta) in method_pairs_g2.items():
        attrA = attr_dict[name_captum] #captum attributions
        attrB = attr_dict[name_meta] #meta-algo attributions
        metric_distr = calc_metric_fn(attrA, attrB)
        metric_dict[name] = metric_distr
    
    return metric_dict

# format
# attr_g2 = {method_name: attr_tensor} #attr_dict
# L1_g2 = {name: L1_distr} #metric_dict