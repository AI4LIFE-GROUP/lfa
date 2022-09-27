import sys
sys.path.append("../code")
from lfa.functions_model_g import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import time


#####model_g

def train_model_sgd_v1(
    model_g,
    train_dl, #DataLoader
    
    loss_fn = binary_cross_entropy_loss,
    max_epoch = 100,
    initial_lr = 0.1,
    gamma = 0.1,
    milestones = None,
    
    reg = None, #regularization (options: None, 1, 2)
    alpha_reg = 1.0, #strength of regularization term in loss
    device = 'cuda' if torch.cuda.is_available() else 'cpu', 
    debug = True
):   
    
    #construct model and initialize parameters
    model_g = construct_model(model_g, train_dl)

    #set optimization parameters
    optimizer = torch.optim.Adam(model_g.parameters(), lr=initial_lr)
    if milestones is None:
        milestones=[50, max_epoch-5]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    #put models on device
    model_g = model_g.to(device)
    
    #train and validate model, iterate over epochs
    t1 = time.time()
    
    for epoch in range(0, max_epoch):
        
        ###TRAIN MODEL
        if debug: 
            # print('-'*20 + f'\nEpoch {epoch}')
            # print(f'initial learning rate: {initial_lr}')
            # print(f'current learning rate: {scheduler.get_last_lr()}') #check if scheduler is changing the learning rates
            
            config.logger.debug('-'*20 + f'\nEpoch {epoch}')
            config.logger.debug(f'initial_lr, current_lr: {initial_lr, scheduler.get_last_lr()}') #check if scheduler is changing the learning rates
            
        train_loss = train_model(model_g, None, train_dl, loss_fn, reg, alpha_reg, None, None, optimizer, device, debug, None)

        #reduce learning rate for next epoch
        scheduler.step()
    
    t2 = time.time()
    
    #move model back to cpu
    model_g = model_g.cpu()
    
    if debug:
        config.logger.debug('\n----- training complete! -----')
        config.logger.debug(f'train_time: {t2 - t1}')
        config.logger.debug(f'train_loss: {train_loss}')
    
    # return {
    #     'train_time': t2 - t1,
    #     'train_loss': train_loss,
    # }


class ModelGLinearV1(nn.Module):
    def __init__(self, train_fn):
        super().__init__()
        self.linear = None
        self.train_fn = train_fn
    
    #construct model parameters; called by .fit()
    def _construct_model_params(self, input_dim):
        self.linear = nn.Linear(input_dim, 1)
    
    #train model using train and validation sets
    def fit(self, train_dl):
        return self.train_fn(self, train_dl)

    def forward(self, x):
        out = self.linear(x)
        return out
    
    #get feature attributions
    def representation(self):
        return self.linear.weight.detach()

    def bias(self):
        return self.linear.bias.detach()
    

class ModelGLogisticV1(nn.Module):
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
    def fit(self, train_dl):
        return self.train_fn(self, train_dl)

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out
    
    #get feature attributions
    def representation(self):
        return self.linear.weight.detach()

    def bias(self):
        return self.linear.bias.detach()


class ModelGLinearConvV1(nn.Module):
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
    def fit(self, train_dl):
        return self.train_fn(self, train_dl)

    def forward(self, x):
        #x: [B, C, H, W]
        out = self.linear(x) #[B, 1, 1, 1]
        return out
    
    #get feature attributions
    def representation(self):
        return self.linear.weight.detach() #[1, C, H, W]

    def bias(self):
        return self.linear.bias.detach() #[1]


class ModelGLogisticConvV1(nn.Module):
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
    def fit(self, train_dl):
        return self.train_fn(self, train_dl)

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

class ExplainerV1():

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
        
    def attribute(self, x, target_class=None, n_perturb=50):
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
        perturb_Y = self.model_f(perturb_X)[:, [target_class]] #[n_perturb, 1]

        #calculate weights for pertubations
        perturb_W = self.calculate_perturbation_weights(x, perturb_X, perturb_Z) #[n_perturb]

        ###train model_g using perturb_Z, perturb_Y, perturb_W

        #get model_f's predictions for perturbations for target_class --> just to create 'dataset' (which needs same format as in ExplainerV2)
        original_X = x.repeat(n_perturb, 1) #[n_perturb, n_features]
        #create tensordataset containing 1) perturbations in z-space, 2) predictions for perturbations, 3) perturbation weights
        dataset = TensorDataset(original_X, perturb_Z, perturb_Y, perturb_W)

        #create dataloaders
        batch_size = 128
        train_size = dataset.__len__()
        train_dl = DataLoader(dataset=dataset, batch_size=train_size if train_size<batch_size else batch_size, shuffle=True)



        #fit interpretable model
        self.model_g.fit(train_dl=train_dl)
        
        return self.model_g.representation(), self.model_g.bias()

        # '''
        # perturb_X: tensor [n_perturb, n_features], perturbations in x-space
        # perturb_Z: tensor [n_perturb, n_intepretable_features], perturbations in z-space
        # perturb_W: tensor [], perturbation weights
        # perturb_Y: tensor [n_perturb, 1], model_f's predictions for perturbations
        # '''



