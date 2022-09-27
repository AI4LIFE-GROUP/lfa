import sys
sys.path.append("../code")
from lfa.functions_model_g import *
from lfa.v1 import train_model_sgd_v1

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import time
import math

from itertools import product



#####model_g

def train_model_sgd_v2(
    model_g,
    model_f, 
    train_dl, #DataLoader
    val_dl, #DataLoader
    
    loss_fn = binary_cross_entropy_loss,
    max_epoch = 100,
    initial_lr = 0.1,
    
    reg = None, #regularization (options: None, 1, 2)
    alpha_reg = 1.0, #strength of regularization term in loss
    gradmatch = True, #whether to include gradient-matching in the loss function
    alpha_gradmatch = 1.0, #strength of gradient-matching term in loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    debug = True,
    noise_type = 'additive',
):
    
    #construct model and initialize parameters
    model_g = construct_model(model_g, train_dl)

    #put models on device
    model_f = model_f.to(device)
    model_g = model_g.to(device)
    
    #set optimization parameters
    optimizer = torch.optim.Adam(model_g.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, max_epoch-5])
    
    #train and validate model, iterate over epochs
    #initialize best validation loss and best model state (used during validation)
    best_val_loss = math.inf
    best_model_state = model_g.state_dict()
    t1 = time.time()
    
    for epoch in range(0, max_epoch):
        
        if debug: 
            config.logger.debug('-'*20 + f'\nEpoch {epoch}')
            config.logger.debug(f'initial_lr, current_lr: {initial_lr, scheduler.get_last_lr()}') #check if scheduler is changing the learning rates
        
        #train model: update model parameters
        train_loss = train_model(model_g, model_f, train_dl, loss_fn, reg, alpha_reg, gradmatch, alpha_gradmatch, optimizer, device, debug, noise_type)

        #validate model: update best_val_loss, best_model_state; save best_model_state
        old_best_val_loss = best_val_loss
        best_val_loss, best_model_state = validate_model(model_g, val_dl, loss_fn, best_val_loss, best_model_state, device, debug, noise_type)
        new_best_val_loss = best_val_loss
        
        if new_best_val_loss < old_best_val_loss:
            best_epoch = epoch
            train_loss_for_best_val_loss = train_loss
        
        #reduce learning rate for next epoch
        scheduler.step() 
            
    #set model to best model (i.e. model with lowest validation loss)
    model_g.load_state_dict(best_model_state)
    
    t2 = time.time()
    
    #move models back to cpu
    model_f = model_f.cpu()
    model_g = model_g.cpu()
    
    if debug:
        config.logger.debug('\n----- training complete! -----')
        config.logger.debug(f'best model weights: {best_model_state}')
        config.logger.debug(f'train_time: {t2 - t1}')
        config.logger.debug(f'best_val_loss: {best_val_loss}')
        config.logger.debug(f'train_loss_for_best_val_loss: {train_loss_for_best_val_loss}')
        config.logger.debug(f'best_epoch: {best_epoch}')
        
    # return {
    #     'train_time': t2 - t1,
    #     'train_loss_for_best_val_loss': train_loss_for_best_val_loss,
    #     'best_val_loss': best_val_loss,
    #     'best_epoch': best_epoch
    # }



def train_model_sgd_gridsearch_v2(
    model_g,
    model_f, 
    train_dl, #DataLoader
    val_dl, #DataLoader
    
    loss_fn = None,
    max_epoch = 100,
    initial_lr_list = [1, 0.5, 0.1],
    gamma_list = [0.5, 0.1, 0.05],
    milestones_list = [[30, 80], [50, 95]],
    
    reg = None, #regularization (options: None, 1, 2)
    alpha_reg = 1.0, #strength of regularization term in loss
    gradmatch = True, #whether to include gradient-matching in the loss function
    alpha_gradmatch = 1.0, #strength of gradient-matching term in loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu', 
    debug = True,
    noise_type = 'additive',
):
    
    #create combinations of hyperparameters for gridsearch
    param_comb = list(product(initial_lr_list, gamma_list, milestones_list))
    
    #initialize best validation loss and best model state (used during validation)
    best_val_loss = math.inf
    best_model_state = model_g.state_dict()
    t1 = time.time()
    
    #iterate through hyperparameter combinations
    for initial_lr, gamma, milestones in param_comb:
        
        #construct model and initialize parameters
        construct_model(model_g, train_dl)

        #put models on device
        model_f = model_f.to(device)
        model_g = model_g.to(device)
        
        #set optimization parameters
        optimizer = torch.optim.Adam(model_g.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
        
        if debug:
            config.logger.debug('*'*20 + f' Hyperparameter combination (initial_lr, gamma, milestones): {initial_lr, gamma, milestones} ' + '*'*20)
    
        #iterate through each epoch
        for epoch in range(0, max_epoch):

            if debug:                
                config.logger.debug('-'*20 + f'\nEpoch {epoch}')
                config.logger.debug(f'initial_lr, current_lr, gamma, milestones: {initial_lr, scheduler.get_last_lr(), gamma, milestones}')
            
            #train model: update model parameters
            train_loss = train_model(model_g, model_f, train_dl, loss_fn, reg, alpha_reg, gradmatch, alpha_gradmatch, optimizer, device, debug, noise_type)

            #validate model: update best_val_loss, best_model_state; save best_model_state
            old_best_val_loss = best_val_loss
            best_val_loss, best_model_state = validate_model(model_g, model_f, val_dl, loss_fn, gradmatch, alpha_gradmatch, best_val_loss, best_model_state, device, debug, noise_type)
            new_best_val_loss = best_val_loss
            
            if new_best_val_loss < old_best_val_loss: #store parameters associated with best_model_state
                best_epoch = epoch
                best_initial_lr = initial_lr
                best_gamma = gamma
                best_milestones = milestones
                train_loss_for_best_val_loss = train_loss
            
            #reduce learning rate for next epoch
            scheduler.step() 

    if debug:
        config.logger.debug('\n----- training complete! -----')
        #config.logger.debug(f'latest model weights: {model_g.state_dict()}')
        
    #set model to best model (i.e. model with lowest validation loss)
    model_g.load_state_dict(best_model_state)
    
    t2 = time.time()
    
    #move models back to cpu
    model_f = model_f.cpu()
    model_g = model_g.cpu()
    
    if debug:        
        config.logger.debug(f'best model weights: {model_g.state_dict()}')
        config.logger.debug(f'train_time: {t2-t1}')
        config.logger.debug(f'best_val_loss: {best_val_loss}')
        config.logger.debug(f'train_loss_for_best_val_loss: {train_loss_for_best_val_loss}')
        config.logger.debug(f'best_initial_lr: {best_initial_lr}')
        config.logger.debug(f'best_gamma: {best_gamma}')
        config.logger.debug(f'best_milestones: {best_milestones}')
        config.logger.debug(f'best_epoch: {best_epoch}')
        
    # return {
    #     'train_time': t2-t1,
    #     'train_loss_for_best_val_loss': train_loss_for_best_val_loss,
    #     'best_val_loss': best_val_loss,
    #     'best_epoch': best_epoch,
    #     'best_initial_lr': best_initial_lr,
    #     'best_gamma': best_gamma,
    #     'best_milestones': best_milestones
    # }


#function to merge two dataloaders
def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


def train_model_sgd_gridsearch_v2_thenretrain(
    model_g,
    model_f, 
    train_dl, #DataLoader
    val_dl, #DataLoader
    
    loss_fn = None,
    max_epoch = 100,
    initial_lr_list = [1, 0.5, 0.1],
    gamma_list = [0.5, 0.1, 0.05],
    milestones_list = [[30, 80], [50, 95]],
    
    reg = None, #regularization (options: None, 1, 2)
    alpha_reg = 1.0, #strength of regularization term in loss
    gradmatch = True, #whether to include gradient-matching in the loss function
    alpha_gradmatch = 1.0, #strength of gradient-matching term in loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu', 
    debug = True,
    noise_type = 'additive',
):
    
    #create combinations of hyperparameters for gridsearch
    param_comb = list(product(initial_lr_list, gamma_list, milestones_list))
    
    #initialize best validation loss and best model state (used during validation)
    best_val_loss = math.inf
    best_model_state = model_g.state_dict()
    t1 = time.time()
    
    #iterate through hyperparameter combinations
    for initial_lr, gamma, milestones in param_comb:
        
        #construct model and initialize parameters
        construct_model(model_g, train_dl)

        #put models on device
        model_f = model_f.to(device)
        model_g = model_g.to(device)
        
        #set optimization parameters
        optimizer = torch.optim.Adam(model_g.parameters(), lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
        
        if debug:
            config.logger.debug('*'*20 + f' Hyperparameter combination (initial_lr, gamma, milestones): {initial_lr, gamma, milestones} ' + '*'*20)
    
        #iterate through each epoch
        for epoch in range(0, max_epoch):

            if debug:                
                config.logger.debug('-'*20 + f'\nEpoch {epoch}')
                config.logger.debug(f'initial_lr, current_lr, gamma, milestones: {initial_lr, scheduler.get_last_lr(), gamma, milestones}')
            
            #train model: update model parameters
            train_loss = train_model(model_g, model_f, train_dl, loss_fn, reg, alpha_reg, gradmatch, alpha_gradmatch, optimizer, device, debug, noise_type)

            #validate model: update best_val_loss, best_model_state; save best_model_state
            old_best_val_loss = best_val_loss
            best_val_loss, best_model_state = validate_model(model_g, model_f, val_dl, loss_fn, gradmatch, alpha_gradmatch, best_val_loss, best_model_state, device, debug, noise_type)
            new_best_val_loss = best_val_loss
            
            if new_best_val_loss < old_best_val_loss: #store parameters associated with best_model_state
                best_epoch = epoch
                best_initial_lr = initial_lr
                best_gamma = gamma
                best_milestones = milestones
                train_loss_for_best_val_loss = train_loss
            
            #reduce learning rate for next epoch
            scheduler.step() 
    
    #use best hyperparameters to retrain g on whole set of perturbations

    #merge train and val dataloaders
    # train_val_dl = itr_merge(train_dl, val_dl)

    #collect data from train_dl
    X_all = torch.Tensor([])
    Z_all = torch.Tensor([])
    Y_all = torch.Tensor([])
    W_all = torch.Tensor([])
    for batch in train_dl:
        X, Z, Y, W = batch
        X_all = torch.cat((X_all, X))
        Z_all = torch.cat((Z_all, Z))
        Y_all = torch.cat((Y_all, Y))
        W_all = torch.cat((W_all, W))

    #collect data from val_dl
    for batch in val_dl:
        X, Z, Y, W = batch
        X_all = torch.cat((X_all, X))
        Z_all = torch.cat((Z_all, Z))
        Y_all = torch.cat((Y_all, Y))
        W_all = torch.cat((W_all, W))

    #make new single dl that contains both train data and val data
    dataset = TensorDataset(X_all, Z_all, Y_all, W_all)
    batch_size = 128
    train_val_size = dataset.__len__()
    train_val_dl = DataLoader(dataset=dataset, batch_size=train_val_size if train_val_size<batch_size else batch_size, shuffle=True)

    # print(X_all.shape)
    # print(Z_all.shape)
    # print(Y_all.shape)
    # print(W_all.shape)

    #retrain g on whole set of perturbations
    train_model_sgd_v1(
        model_g=model_g, 
        train_dl=train_val_dl, #DataLoader, combined train_dl + val_dl
        loss_fn=loss_fn, 
        max_epoch=max_epoch, 
        initial_lr=best_initial_lr, #based on validation
        gamma=best_gamma, #based on validation
        milestones=best_milestones, #based on validation
        reg=reg, 
        alpha_reg=alpha_reg, 
        device=device, 
        debug=debug)
    #all other arguments above based on argument for main train_model_sgd_gridsearch_v2_thenretrain() function
    #note train_model_sgd_gridsearch_v2_thenretrain() cannot do gradient matching bc train_model_sgd_v1() cannot do gradient-matching --> just for proof of concept



    # if debug:
    #     config.logger.debug('\n----- training complete! -----')
    #     #config.logger.debug(f'latest model weights: {model_g.state_dict()}')
        
    # #set model to best model (i.e. model with lowest validation loss)
    # model_g.load_state_dict(best_model_state)
    
    # t2 = time.time()
    
    # #move models back to cpu
    # model_f = model_f.cpu()
    # model_g = model_g.cpu()
    
    # if debug:        
    #     config.logger.debug(f'best model weights: {model_g.state_dict()}')
    #     config.logger.debug(f'train_time: {t2-t1}')
    #     config.logger.debug(f'best_val_loss: {best_val_loss}')
    #     config.logger.debug(f'train_loss_for_best_val_loss: {train_loss_for_best_val_loss}')
    #     config.logger.debug(f'best_initial_lr: {best_initial_lr}')
    #     config.logger.debug(f'best_gamma: {best_gamma}')
    #     config.logger.debug(f'best_milestones: {best_milestones}')
    #     config.logger.debug(f'best_epoch: {best_epoch}')
        
    # return {
    #     'train_time': t2-t1,
    #     'train_loss_for_best_val_loss': train_loss_for_best_val_loss,
    #     'best_val_loss': best_val_loss,
    #     'best_epoch': best_epoch,
    #     'best_initial_lr': best_initial_lr,
    #     'best_gamma': best_gamma,
    #     'best_milestones': best_milestones
    # }


class ModelGLinearV2(nn.Module):
    def __init__(self, train_fn):
        super().__init__()
        self.linear = None
        self.train_fn = train_fn
    
    #construct model parameters; called by .fit()
    def _construct_model_params(self, input_dim):
        self.linear = nn.Linear(input_dim, 1)
    
    #train model using train and validation sets
    def fit(self, model_f, train_dl, val_dl):
        return self.train_fn(self, model_f, train_dl, val_dl)

    def forward(self, x):
        out = self.linear(x)
        return out
    
    #get feature attributions
    def representation(self):
        return self.linear.weight.detach()

    def bias(self):
        return self.linear.bias.detach()

    
    
class ModelGLinearNoBiasV2(nn.Module):
    def __init__(self, train_fn):
        super().__init__()
        self.linear = None
        self.train_fn = train_fn
    
    #construct model parameters; called by .fit()
    def _construct_model_params(self, input_dim):
        self.linear = nn.Linear(in_features=input_dim, out_features=1, bias=False)
    
    #train model using train and validation sets
    def fit(self, model_f, train_dl, val_dl):
        return self.train_fn(self, model_f, train_dl, val_dl)

    def forward(self, x):
        out = self.linear(x)
        return out
    
    #get feature attributions
    def representation(self):
        return self.linear.weight.detach()

    def bias(self):
        return torch.Tensor([0])
    


class ModelGLogisticV2(nn.Module):
    def __init__(self, train_fn):
        super().__init__()
        self.linear = None
        self.sigmoid = None
        self.train_fn = train_fn
    
    #construct model parameters; called by .fit()
    def _construct_model_params(self, input_dim):
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    #train model using train and validation sets
    def fit(self, model_f, train_dl, val_dl):
        return self.train_fn(self, model_f, train_dl, val_dl)

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out
    
    #get feature attributions
    def representation(self):
        return self.linear.weight.detach()

    def bias(self):
        return self.linear.bias.detach()


class ModelGLinearConvV2(nn.Module):
    def __init__(self, train_fn):
        super().__init__()
        self.linear = None
        self.train_fn = train_fn
        
    #construct model parameters; called by .fit()
    def _construct_model_params(self, input_dim):
        '''
        input_dim: int, height (H) of image (assumes image is [C, H, W] with H=W)
        '''
        C=3
        H=input_dim
        self.linear = nn.Conv2d(in_channels=C, out_channels=1, kernel_size=H)
        
    #train model using train and validation sets
    def fit(self, model_f, train_dl, val_dl):
        return self.train_fn(self, model_f, train_dl, val_dl)

    def forward(self, x):
        #x: [B, C, H, W]
        out = self.linear(x) #[B, 1, 1, 1]
        return out
    
    #get feature attributions
    def representation(self):
        return self.linear.weight.detach() #[1, C, H, W]

    def bias(self):
        return self.linear.bias.detach() #[1]
    

class ModelGLogisticConvV2(nn.Module):
    def __init__(self, train_fn):
        super().__init__()
        self.linear = None
        self.sigmoid = None
        self.train_fn = train_fn
        
    #construct model parameters; called by .fit()
    def _construct_model_params(self, input_dim):
        '''
        input_dim: int, height (H) of image (assumes image is [C, H, W] with H=W)
        '''
        C=3
        H=input_dim
        self.linear = nn.Conv2d(in_channels=C, out_channels=1, kernel_size=H)
        self.sigmoid = nn.Sigmoid()
        
    #train model using train and validation sets
    def fit(self, model_f, train_dl, val_dl):
        return self.train_fn(self, model_f, train_dl, val_dl)

    def forward(self, x):
        #x: [B, C, H, W]
        out = self.linear(x) #[B, 1, 1, 1]
        out = self.sigmoid(out) #[B, 1, 1, 1]
        return out
    
    #get feature attributions
    def representation(self):
        return self.linear.weight.detach() #[1, C, H, W]

    def bias(self):
        return self.linear.bias.detach() #[1]


#####Explainer

class ExplainerV2():

    def __init__(
        self,
        model_f,
        model_g,
        generate_perturbations,
        calculate_perturbation_weights):
        
        self.model_f = model_f
        self.model_g = model_g
        self.calculate_perturbation_weights = calculate_perturbation_weights
        self.generate_perturbations = generate_perturbations
        
        
    def attribute(self, x, target_class=None, n_perturb=50, explain_delta_f=False):
        '''
        #x: tensor [1, n_features], instance to be explained
        #target_class: None or integer, predicted class for which to generate an explanation; if target_class=None, explanation will be generated for x's predicted class; model_f outputs [n_data_points, n_classes] and target_class is the index for 'n_classes'
        '''
        
        #generate perturbations of x in x-space and z-space
        perturb_X, perturb_Z = self.generate_perturbations(x, n_perturb) #perturb_X: [n_perturb, n_features], perturb_Z = [n_perturb, n_interpretable_features]

        #get target class for which to generate an explanation
        if target_class is None:
            target_class = self.model_f(x).data.max(1)[1].item() #x:[1, n_features], model_f(x):[1, 1], target_class: int

        #get model_f's predictions for perturbations for target_class
        original_X = x.repeat(n_perturb, 1) #[n_perturb, n_features]
        if explain_delta_f:
            perturb_Y = self.model_f(x)[:, [target_class]] - self.model_f(original_X * (1 - perturb_Z))[:, [target_class]]#[n_perturb, 1]
        else:
            perturb_Y = self.model_f(perturb_X)[:, [target_class]] #[n_perturb, 1]

        #calculate weights for pertubations
        perturb_W = self.calculate_perturbation_weights(x, perturb_X, perturb_Z) #[n_perturb]

        ###train model_g using perturb_Z, perturb_Y, perturb_W

        #create tensordataset containing 1) original data point x to explain, 2) perturbations in z-space, 3) predictions for perturbations, 4) perturbation weights
        dataset = TensorDataset(original_X, perturb_Z, perturb_Y, perturb_W)

        #split dataset into train/validation sets (80/20 split)
        train_size = int(dataset.__len__()*0.8)
        val_size = dataset.__len__() - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        #create dataloaders
        batch_size = 128
        train_dl = DataLoader(dataset=train_ds, batch_size=train_size if train_size<batch_size else batch_size, shuffle=True)
        val_dl = DataLoader(dataset=val_ds, batch_size=val_size if val_size<batch_size else batch_size, shuffle=True)

        #fit interpretable model
        self.model_g.fit(model_f=self.model_f, train_dl=train_dl, val_dl=val_dl)
        
        return self.model_g.representation(), self.model_g.bias()

        # '''
        # perturb_X: tensor [n_perturb, n_features], perturbations in x-space
        # perturb_Z: tensor [n_perturb, n_intepretable_features], perturbations in z-space
        # perturb_W: tensor [], perturbation weights
        # perturb_Y: tensor [n_perturb, 1], model_f's predictions for perturbations
        # '''

