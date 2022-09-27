import torch

from scipy.special import factorial #used for shapley_kernel
import numpy as np #used for generate_one_hot_vector_perturbations

#####generate_perturbations

def generate_binary_perturbations(x, n_perturb):
    '''
    x:
        - tabular: tensor [1, n_features], original input
        - image: tensor [1, C, H, W], original input
    n_perturb: integer, number of perturbations
    
    perturb_X: tensor [n_perturb, n_features], perturbations in x-space, vectors with some original values and other values set to zero
    perturb_Z: tensor [n_perturb, n_features], perturbations in z-space, binary vectors representing presence/absence of features
    '''
    
    #get dimensions of perturbation matrix  
    dim_list = list(x.size()) #x=tabular, dim_list=[1, n_features]; x=image, dim_list=[1, C, H, W]
    dim_list[0] = n_perturb #x=tabular, dim_list=[n_perturb, n_features]; x=image, dim_list=[n_perturb, C, H, W]
    
    #generate perturbations in x-space 
    prob_matrix = torch.ones(dim_list) * 0.5
    binary_mask = torch.bernoulli(prob_matrix)
    perturb_X = x * binary_mask
    
    #generate perturbations in z-space
    perturb_Z = binary_mask
    
    return perturb_X, perturb_Z


def generate_gaussian_perturbations(x, n_perturb, epsilon):
    '''
    x:
        - tabular: tensor [1, n_features], original input
        - image: tensor [1, C, H, W], original input
    n_perturb: integer, number of perturbations
    epsilon: float, standard deviation of Normal distribution [perturbation = x + Normal(0, epsilon^2)]
    
    perturb_X: tensor [n_perturb, n_features], perturbations in x-space, original input + Gaussian noise
    perturb_Z: tensor [n_perturb, n_features], perturbations in z-space, Gaussian noise added to original input
    '''
    
    #get dimensions of perturbation matrix  
    dim_list = list(x.size()) #x=tabular, dim_list=[1, n_features]; x=image, dim_list=[1, C, H, W]
    dim_list[0] = n_perturb #x=tabular, dim_list=[n_perturb, n_features]; x=image, dim_list=[n_perturb, C, H, W]
    
    #generate perturbations in x-space
    noise = torch.randn_like(torch.zeros(dim_list)) * epsilon
    perturb_X = x + noise
    
    #generate perturbations in z-space
    perturb_Z = noise
    
    return perturb_X, perturb_Z



def generate_gaussian_perturbations_image(x, n_perturb, epsilon):
    '''
    x:
        - image: tensor [1, C, H, W], original input
    n_perturb: integer, number of perturbations
    epsilon: float, standard deviation of Normal distribution [perturbation = x + Normal(0, epsilon^2)]
    
    perturb_X: tensor [n_perturb, C, H, W], perturbations in x-space, original input + Gaussian noise
    perturb_Z: tensor [n_perturb, C, H, W], perturbations in z-space, Gaussian noise added to original input
    '''
    
    #generate perturbations in x-space
    C = x.size(-3)
    H = x.size(-2)
    W = x.size(-1) 
    noise = torch.randn_like(torch.zeros([n_perturb, C, H, W])) * epsilon
    perturb_X = x + noise
    
    #generate perturbations in z-space
    perturb_Z = noise
    
    return perturb_X, perturb_Z


def generate_uniform_perturbations(x, n_perturb, low=0, high=1):
    '''
    x: tensor [1, n_features], original input
    n_perturb: integer, number of perturbations
    
    perturb_X: tensor [n_perturb, n_features], perturbations in x-space, original input + Gaussian noise
    perturb_Z: tensor [n_perturb, n_features], perturbations in z-space, Gaussian noise added to original input
    '''
    n_features = x.size(1)
    
    #randomly sample 'n_perturb' values from uniform distribution
    alpha = torch.distributions.uniform.Uniform(low, high).sample([n_perturb, 1]) #[n_perturb, 1]
    
    #horizontally stack 'n_perturb' number of 'x' vectors
    x_stacked = x.repeat(n_perturb, 1) #[n_perturb, n_features]
    
    #generate perturbations in x-space
    perturb_X = alpha * x_stacked #[n_perturb, n_features]
    
    #generate perturbations in z-space
    perturb_Z = alpha.repeat(1, n_features) #[n_perturb, n_features]

    return perturb_X, perturb_Z


def generate_one_hot_vector_perturbations(x, n_perturb):
    '''
    x: tensor [1, n_features], original input
    n_perturb: integer<=n_features, number of perturbations
    
    perturb_X: tensor [n_perturb, n_features], perturbations in x-space, original input + Gaussian noise
    perturb_Z: tensor [n_perturb, n_features], perturbations in z-space, Gaussian noise added to original input
    '''
    n_features = x.size(1)
    
    #generate all possible one-hot vectors (if n_perturb=None)
    if n_perturb == n_features:
        #create all possible one-hot vectors --> identity matrix
        n_perturb = n_features
        one_hot_vectors = torch.eye(n_features) #[n_perturb=n_features, n_features]
    
    #generate a subset of all possible one-hot vectors (if n_perturb=int)
    if n_perturb < n_features:
        #create 'n_perturb' number of random one-hot vectors
        one_hot_vectors = torch.zeros((n_perturb, n_features)) #[n_perturb, n_features]
        one_hot_idx = torch.randint(low=0, high=n_features, size=(n_perturb,)) #[n_perturb]; #randomly sample integers between 0 and n_features --> column indices used to create one-hot vectors
        one_hot_vectors[range(0, n_perturb), one_hot_idx] = 1 #[n_perturb, n_features]; for a given row, change a given column to 1
        
    if n_perturb > n_features:
        #create identity matrix
        identity_matrix = torch.eye(n_features) #[n_perturb=n_features, n_features]
        #draw 'n_perturb' numbers with replacement from 0 to n_features
        row_idxs = list(range(0, n_features))
        random_row_idxs = list(np.random.choice(row_idxs, size=n_perturb, replace=True))
        #extract those rows
        one_hot_vectors = identity_matrix[random_row_idxs, ]
        
    #generate perturbations in z-space
    # perturb_Z = one_hot_vectors #[n_perturb, n_features]
    # perturb_Z = 1-one_hot_vectors #[n_perturb, n_features]
    perturb_Z = one_hot_vectors #[n_perturb, n_features]

    #horizontally stack 'n_perturb' number of 'x' vectors
    x_stacked = x.repeat(n_perturb, 1) #[n_perturb, n_features]

    #generate perturbations in x-space
    # perturb_X = x_stacked * (1-perturb_Z) #[n_perturb, n_features]
    # perturb_X = x_stacked * perturb_Z #[n_perturb, n_features]
    perturb_X = x_stacked * perturb_Z #[n_perturb, n_features]
    
    return perturb_X, perturb_Z



#####calculate_perturbation_weights

#exponential kernel
def exponential_kernel(x, perturb_X, perturb_Z, kernel_width):
    #perturb_Z not used
    #calculate perturbation weights based on exponential kernel
    L2_dist = torch.linalg.norm(x-perturb_X, ord=2, dim=1) #[n_perturb]
    perturb_W = torch.exp(- (L2_dist**2) / (kernel_width**2)) #[n_perturb]
    return perturb_W

def exponential_kernel_image(x, perturb_X, perturb_Z, kernel_width):
    #perturb_Z not used
    #x: [1, C, H, W]
    #perturb_X: [n_perturb, C, H, W]
    diff = (x-perturb_X).flatten(start_dim=1) #[n_perturb, C*H*W]
    L2_dist = torch.linalg.norm(diff, ord=2, dim=1) #[n_perturb]
    perturb_W = torch.exp(- (L2_dist**2) / (kernel_width**2)) #[n_perturb]
    return perturb_W

#shapley kernel
def shapley_kernel(x, perturb_X, perturb_Z):
    #calculate perturbation weights based on shapley kernel
    #x: [1, n_features]
    #perturb_X: [n_perturb, n_features]
    #perturb_Z: [n_perturb, n_interpretable_features]
    
    m = perturb_Z.size(1) #m=int, number of intepretable features
    k = perturb_Z.sum(axis=1) #k:[n_perturb], number of 1's in each binary vector
    mCk = factorial(m) / (factorial(k) * factorial(m-k)) #[n_perturb], term in shapley kernel
    perturb_W = (m-1) / (mCk * k * (m-k)) #[n_perturb]
    
    #if perturb_Z contains a vector (row) of all 0s or all 1s, set the weight to infinite (1000000)
    idx_inf, = torch.where(perturb_Z.sum(axis=1) == 0) #binary vector of all 0s
    perturb_W[idx_inf] = 1000000

    idx_inf, = torch.where(perturb_Z.sum(axis=1) == m) #binary vector of all 1s
    perturb_W[idx_inf] = 1000000

    return perturb_W

#constant kernel
def constant_kernel(x, perturb_X, perturb_Z):
    #perturb_Z not used
    #calculate perturbation weights based on constant kernel
    n_perturb = perturb_X.size(0)
    perturb_W = torch.ones([n_perturb]) #[n_perturb]
    return perturb_W