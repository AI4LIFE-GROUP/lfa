import sys
sys.path.append("../code")
from lfa.v2 import *
import lfa.config as config
from lfa.functions_perturbs_and_kernels import generate_binary_perturbations, generate_gaussian_perturbations, generate_uniform_perturbations, generate_one_hot_vector_perturbations, exponential_kernel, shapley_kernel, constant_kernel 

from captum.attr import Lime, KernelShap, Saliency, NoiseTunnel, IntegratedGradients, InputXGradient, Occlusion
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr._core.lime import get_exp_kernel_similarity_function

from functools import partial



#####approach: existing methods --> use captum library

#lime
def captum_lime(x, target_class, n_perturb, model_f, kernel_width):
    if target_class is None:
        target_class = model_f(x).data.max(1)[1].item() #x:[1, n_features], model_f(x):[1, n_classes], target_class: int
    exponential_kernel = get_exp_kernel_similarity_function('euclidean', kernel_width=kernel_width)
    method = Lime(forward_func=model_f, interpretable_model=SkLearnLinearRegression(), similarity_func=exponential_kernel)
    attr = method.attribute(inputs=x, target=target_class, n_samples=n_perturb) #match kernel_width used in meta-alg
    return attr, torch.Tensor([]) #feature_attribution, bias=empty


#kernelshap
def captum_ks(x, target_class, n_perturb, model_f):
    if target_class is None:
        target_class = model_f(x).data.max(1)[1].item() #x:[1, n_features], model_f(x):[1, n_classes], target_class: int
    method = KernelShap(model_f)
    attr = method.attribute(inputs=x, target=target_class, n_samples=n_perturb)
    return attr, torch.Tensor([]) #feature_attribution, bias=empty


#smoothgrad
def captum_sg(x, target_class, n_perturb, model_f):
    if target_class is None:
        target_class = model_f(x).data.max(1)[1].item() #x:[1, n_features], model_f(x):[1, n_classes], target_class: int
    model_f.zero_grad()
    method = NoiseTunnel(Saliency(model_f))
    attr = method.attribute(inputs=x, nt_type='smoothgrad', target=target_class, nt_samples=n_perturb, abs=False, stdevs=0.1) #match epsilon used for v2
    return attr, torch.Tensor([]) #feature_attribution, bias=empty


#integrated gradients
def captum_ig(x, target_class, n_perturb, model_f):
    if target_class is None:
        target_class = model_f(x).data.max(1)[1].item() #x:[1, n_features], model_f(x):[1, n_classes], target_class: int
    model_f.zero_grad()
    method = IntegratedGradients(model_f)
    attr = method.attribute(inputs=x, target=target_class, n_steps=n_perturb)
    return attr, torch.Tensor([]) #feature_attribution, bias=empty


#vanilla gradients
def captum_vg(x, target_class, model_f):
    if target_class is None:
        target_class = model_f(x).data.max(1)[1] #x:[1, n_features], model_f(x):[1, n_classes], target_class: int
    model_f.zero_grad()
    method = Saliency(model_f)
    x.requires_grad_()
    attr = method.attribute(inputs=x, target=target_class, abs=False)
    return attr #feature_attribution


#gradient x input
def captum_gxi(x, target_class, model_f):
    if target_class is None:
        target_class = model_f(x).data.max(1)[1] #x:[1, n_features], model_f(x):[1, n_classes], target_class: int
    model_f.zero_grad()
    method = InputXGradient(model_f)
    x.requires_grad_()
    attr = method.attribute(inputs=x, target=target_class)
    return attr #feature_attribution


#occlusion
def captum_oc(x, target_class, model_f):
    if target_class is None:
        target_class = model_f(x).data.max(1)[1] #x:[1, n_features], model_f(x):[1, n_classes], target_class: int
    method = Occlusion(model_f)
    attr = method.attribute(inputs=x, target=target_class, sliding_window_shapes=(1,)) #for x:[1, n_features]
    return attr #feature_attribution



#####approach: local function approximation --> implement framework

#lime
def create_explainer_meta_lime(model_f):
	#explainer arguments
	train_fn = partial(train_model_sgd_gridsearch_v2, 
	                   loss_fn = l2_loss,
	                   max_epoch = 50,
	                   initial_lr_list = [1, 0.5], 
	                   gamma_list = [0.5, 0.1], 
	                   milestones_list = [[30, 40], [40, 45]], 
	                   reg = None,
	                   alpha_reg = None,
	                   gradmatch = False, 
	                   alpha_gradmatch = None, 
	                   device = 'cuda' if torch.cuda.is_available() else 'cpu',
	                   debug = False,
	                   noise_type = 'multiplicative')
	model_g = ModelGLinearV2(train_fn)
	generate_perturbations = generate_binary_perturbations
	calculate_perturbation_weights = partial(exponential_kernel, kernel_width=1) 

	#create explainer
	meta_lime = ExplainerV2(model_f, model_g, generate_perturbations, calculate_perturbation_weights)

	return meta_lime


#kernelshap
def create_explainer_meta_ks(model_f):
	#explainer arguments
	train_fn = partial(train_model_sgd_gridsearch_v2, 
	                   loss_fn = l2_loss,
	                   max_epoch = 50,
	                   initial_lr_list = [1, 0.5], 
	                   gamma_list = [0.5, 0.1], 
	                   milestones_list = [[30, 40], [40, 45]], 
	                   reg = None,
	                   alpha_reg = None,
	                   gradmatch = None, 
	                   alpha_gradmatch = None, 
	                   device = 'cuda' if torch.cuda.is_available() else 'cpu',
	                   debug = False,
	                   noise_type = 'multiplicative')
	model_g = ModelGLinearV2(train_fn)
	generate_perturbations = generate_binary_perturbations
	calculate_perturbation_weights = shapley_kernel

	#create explainer
	meta_ks = ExplainerV2(model_f, model_g, generate_perturbations, calculate_perturbation_weights)

	return meta_ks


#smoothgrad with mse loss --> linear_g
def create_explainer_meta_sg_mse(model_f, epsilon=0.1):
	#explainer arguments
	train_fn = partial(train_model_sgd_gridsearch_v2, 
	                   loss_fn = l2_loss,
	                   max_epoch = 50,
	                   initial_lr_list = [1, 0.5], 
	                   gamma_list = [0.5, 0.1], 
	                   milestones_list = [[30, 40], [40, 45]], 
	                   reg = None,
	                   alpha_reg = None,
	                   gradmatch = None, 
	                   alpha_gradmatch = None, 
	                   device = 'cuda' if torch.cuda.is_available() else 'cpu',
	                   debug = False,
	                   noise_type = 'additive')
	model_g = ModelGLinearV2(train_fn)
	generate_perturbations = partial(generate_gaussian_perturbations, epsilon=0.1) 
	calculate_perturbation_weights = constant_kernel 

	#create explainer
	meta_sg_mse = ExplainerV2(model_f, model_g, generate_perturbations, calculate_perturbation_weights)

	return meta_sg_mse



#smoothgrad with mse loss --> logistic_g
def create_explainer_meta_sg_mse_logistic(model_f, epsilon=0.1):
	#explainer arguments
	train_fn = partial(train_model_sgd_gridsearch_v2, 
	                   loss_fn = l2_loss,
	                   max_epoch = 50,
	                   initial_lr_list = [1, 0.5], 
	                   gamma_list = [0.5, 0.1], 
	                   milestones_list = [[30, 40], [40, 45]], 
	                   reg = None,
	                   alpha_reg = None,
	                   gradmatch = None, 
	                   alpha_gradmatch = None, 
	                   device = 'cuda' if torch.cuda.is_available() else 'cpu',
	                   debug = False,
	                   noise_type = 'additive')
	model_g = ModelGLogisticV2(train_fn)
	generate_perturbations = partial(generate_gaussian_perturbations, epsilon=0.1) 
	calculate_perturbation_weights = constant_kernel 

	#create explainer
	meta_sg_mse_logistic = ExplainerV2(model_f, model_g, generate_perturbations, calculate_perturbation_weights)
	
	return meta_sg_mse_logistic



#smoothgrad with gradient-matching loss --> linear_g
def create_explainer_meta_sg_gm(model_f, epsilon=0.1):
	#explainer arguments
	train_fn = partial(train_model_sgd_gridsearch_v2, 
	                   loss_fn = None,
	                   max_epoch = 50,
	                   initial_lr_list = [1, 0.5], 
	                   gamma_list = [0.5, 0.1], 
	                   milestones_list = [[30, 40], [40, 45]], 
	                   reg = None,
	                   alpha_reg = None,
	                   gradmatch = True, 
	                   alpha_gradmatch = 1.0, 
	                   device = 'cuda' if torch.cuda.is_available() else 'cpu',
	                   debug = False,
	                  noise_type = 'additive')
	model_g = ModelGLinearV2(train_fn)
	generate_perturbations = partial(generate_gaussian_perturbations, epsilon=epsilon) 
	calculate_perturbation_weights = constant_kernel 

	#create explainer
	meta_sg_gm = ExplainerV2(model_f, model_g, generate_perturbations, calculate_perturbation_weights)

	return meta_sg_gm



#smoothgrad with gradient-matching loss --> logistic_g
def create_explainer_meta_sg_gm_logistic(model_f, epsilon=0.1):
	#explainer arguments
	train_fn = partial(train_model_sgd_gridsearch_v2, 
	                   loss_fn = None,
	                   max_epoch = 50,
	                   initial_lr_list = [1, 0.5], 
	                   gamma_list = [0.5, 0.1], 
	                   milestones_list = [[30, 40], [40, 45]], 
	                   reg = None,
	                   alpha_reg = None,
	                   gradmatch = True, 
	                   alpha_gradmatch = 1.0, 
	                   device = 'cuda' if torch.cuda.is_available() else 'cpu',
	                   debug = False,
	                   noise_type = 'additive')
	model_g = ModelGLogisticV2(train_fn)
	generate_perturbations = partial(generate_gaussian_perturbations, epsilon=0.1) 
	calculate_perturbation_weights = constant_kernel 

	#create explainer
	meta_sg_gm_logistic = ExplainerV2(model_f, model_g, generate_perturbations, calculate_perturbation_weights)

	return meta_sg_gm_logistic



#vanilla gradients = smoothgrad with gradient-matching loss and sigma --> 0
#use create_explainer_meta_sg_gm() with epsilon --> 0


#integrated gradients --> linear_g
def create_explainer_meta_ig(model_f, low=0):
	#explainer arguments
	train_fn = partial(train_model_sgd_gridsearch_v2, 
	                   loss_fn = None,
	                   max_epoch = 50,
	                   initial_lr_list = [1, 0.5], 
	                   gamma_list = [0.5, 0.1], 
	                   milestones_list = [[30, 40], [40, 45]], 
	                   reg = None,
	                   alpha_reg = None,
	                   gradmatch = True, 
	                   alpha_gradmatch = 1.0, 
	                   device = 'cuda' if torch.cuda.is_available() else 'cpu',
	                   debug = False,
	                   noise_type = 'multiplicative')
	model_g = ModelGLinearV2(train_fn)
	generate_perturbations = partial(generate_uniform_perturbations, low=low) 
	calculate_perturbation_weights = constant_kernel 

	#create explainer
	meta_ig = ExplainerV2(model_f, model_g, generate_perturbations, calculate_perturbation_weights)

	return meta_ig



#integrated gradients --> logistic_g
def create_explainer_meta_ig_logistic(model_f, low=0):
	#explainer arguments
	train_fn = partial(train_model_sgd_gridsearch_v2, 
	                   loss_fn = None,
	                   max_epoch = 50,
	                   initial_lr_list = [1, 0.5], 
	                   gamma_list = [0.5, 0.1], 
	                   milestones_list = [[30, 40], [40, 45]], 
	                   reg = None,
	                   alpha_reg = None,
	                   gradmatch = True, 
	                   alpha_gradmatch = 1.0, 
	                   device = 'cuda' if torch.cuda.is_available() else 'cpu',
	                   debug = False,
	                  noise_type = 'multiplicative')
	model_g = ModelGLogisticV2(train_fn)
	generate_perturbations = partial(generate_uniform_perturbations, low=0) 
	calculate_perturbation_weights = constant_kernel 

	#create explainer
	meta_ig_logistic = ExplainerV2(model_f, model_g, generate_perturbations, calculate_perturbation_weights)
	
	return meta_ig_logistic



#gradientxinput = integrated gradients with lowerbound --> 1
#use create_explainer_meta_ig() with low --> 1


#occlusion
def create_explainer_meta_oc(model_f):
	#explainer arguments
	train_fn = partial(train_model_sgd_gridsearch_v2, 
	                   loss_fn = l2_loss,
	                   max_epoch = 50,
	                   initial_lr_list = [1, 0.5], 
	                   gamma_list = [0.5, 0.1],
	                   milestones_list = [[30, 40], [40, 45]], 
	                   reg = None,
	                   alpha_reg = None,
	                   gradmatch = False, 
	                   alpha_gradmatch = None, 
	                   device = 'cuda' if torch.cuda.is_available() else 'cpu',
	                   debug = False, 
	                   noise_type = None)
	model_g = ModelGLinearNoBiasV2(train_fn)
	generate_perturbations = generate_one_hot_vector_perturbations
	calculate_perturbation_weights = constant_kernel

	#create explainer
	meta_oc = ExplainerV2(model_f, model_g, generate_perturbations, calculate_perturbation_weights)

	return meta_oc












