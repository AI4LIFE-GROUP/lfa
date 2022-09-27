import sys
sys.path.append("../code")
import lfa.config as config

import torch


#####loss functions

#(weighted) square loss
def l2_loss(pred, true, weights):
    weights_normalized = weights / weights.norm(p=1)
    return torch.sum(weights_normalized * ((pred - true) ** 2)) / 2.0

#mean square error loss
def mse_loss(pred, true, weights):
    return torch.mean(((pred - true) ** 2)) #weights not used

#cross entropy loss
def cross_entropy_loss(pred_prob, true_class, weights):
    weights_normalized = weights / weights.norm(p=1)
    return -torch.sum(weights_normalized * true_class * torch.log(pred_prob))

#binary cross entropy loss
def binary_cross_entropy_loss(yhat, y, weights):
    weights_normalized = weights / weights.norm(p=1)
    BCE_point_level = -(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat)) * weights_normalized
    return BCE_point_level.mean()


#####training functions

def construct_model(model_g, train_dl):
    '''
    PARAMETERS
    model_g: model of class ModelG___()
    train_dl: dataloader, only used to calculate number of features
    RETURN
    model_g: model_g with correct number of input features and intialized parameters
    '''
    #construct model parameters
    _, batch_Z, _, _ = next(iter(train_dl)) #get a batch just to calculate number of features
    n_features = batch_Z.size(-1)
    model_g._construct_model_params(input_dim=n_features)
    
    #initialize model parameters
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model_g.linear.weight)
        if 'bias' in model_g.linear.state_dict().keys():
            model_g.linear.bias.zero_()
    
    return model_g


def train_model(model_g, model_f, train_dl, loss_fn, reg, alpha_reg, gradmatch, alpha_gradmatch, optimizer, device, debug, noise_type):
    '''
    trains model for one epoch
    
    PARAMETERS
    model_g: model of class ModelG___()
    train_dl: dataloader for training data
    ...: all parameters of higher function
    
    '''
    
    model_g.train()
    loss_over_epoch_train = 0
    n_train_samples = 0
    
    #for each batch in train_dataloader
    for batch_idx_train, batch_train in enumerate(train_dl):
        #unpack batch
        original_X, Z, Y, W = batch_train #original_X=[batch_size, n_features] stacked matrix of original x, X=[batch_size, n_interpretable_features] perturbations in z-space (perturb_Z), Y=[batch_size, 1] prediction for perturbations, W=[batch_size] weights for perturbations
        original_X = original_X.to(device) 
        Z = Z.to(device)
        Z.requires_grad_()
        Y = Y.to(device)
        W = W.to(device)
        
        # if debug:
            # print(f'batch_idx_train: {batch_idx_train}')
            # print(f'X size: {X.size()}')
            # print(f'Y size: {Y.size()}')
            # print(f'Z size: {W.size()}')
            # print(f'X on device: {X.is_cuda}')
            # print(f'Y on device: {Y.is_cuda}')
            # print(f'Z on device: {W.is_cuda}')
            # print(f'model_f on device: {next(model_f.parameters()).is_cuda}')
            # print(f'model_g on device: {next(model_g.parameters()).is_cuda}')
            
        #calculate model prediction
        pred_g = model_g(Z)
        
        #calculate loss (for this batch)
        loss = 0
        if loss_fn is not None:
            loss += loss_fn(pred_g, Y.detach(), W)
        if reg is not None:
            reg_norm = torch.norm(model_g.linear.weight, p=reg) #torch.norm() may be deprecated in the future
            loss += reg_norm * alpha_reg
        if gradmatch:
            #calculate input gradient of f
            if noise_type=='additive':
                perturb_X = original_X + Z
            if noise_type=='multiplicative':
                perturb_X = original_X * Z
            
            pred_f = model_f(perturb_X)
            input_grad_f = torch.autograd.grad(outputs=pred_f.sum(), inputs=Z, only_inputs=True, retain_graph=True)[0]
            #calculate input gradient of g
            input_grad_g = torch.autograd.grad(outputs=pred_g.sum(), inputs=Z, only_inputs=True, create_graph=True)[0]
            #add gradient term to loss 
            grad_norm = torch.norm(input_grad_f.detach_()-input_grad_g, p=2)**2
            loss += grad_norm * alpha_gradmatch
            
        #update model
        loss.backward() #backpropagate loss
        optimizer.step() #update parameters            
        optimizer.zero_grad()
        
        #calculate runnning loss over batches so far
        loss_over_epoch_train += loss.item()
        n_train_samples += Z.size(0)
        
        if debug:
            config.logger.debug(f'   batch_idx_train, batch_size: {batch_idx_train, Z.size(0)}')
            config.logger.debug(f'train_loss, batch-level: {loss.item()/Z.size(0)}')

    #back to epoch level 
    #calculate train loss
    train_loss = loss_over_epoch_train/n_train_samples #loss per datapoint in train set
    
    if debug:
        # print(f'train_loss: {train_loss}')
        # print(f'n_train_samples: {n_train_samples}')
        config.logger.debug(f'train_loss, epoch-level: {train_loss}')
        config.logger.debug(f'n_train_samples: {n_train_samples}')
        config.logger.debug(f'model_g, weights: \n{model_g.state_dict()}')
        
    return train_loss

#                  model_g, model_f, train_dl, loss_fn, reg, alpha_reg, gradmatch, alpha_gradmatch, optimizer, device, debug
#add model_f, gradmatch, alpha_gradmatch
def validate_model(model_g, model_f, val_dl, loss_fn, gradmatch, alpha_gradmatch, best_val_loss, best_model_state, device, debug, noise_type):
    '''
    PARAMETERS
    ...: all parameters of higher function
    
    RETURNS
    best_val_loss: best validation loss (either the input best_val_loss or a new lower best_val_loss)
    best_model_state: best model state (either the input best_model_state or a new lower best_model_state)
    '''
    
    model_g.eval()
    loss_over_epoch_val = 0
    n_val_samples = 0
    
    for batch_idx_val, batch_val in enumerate(val_dl):
        #unpack batch
        original_X, Z, Y, W = batch_val #X=[batch_size, n_interpretable_features] perturbations in z-space, Y=[batch_size, 1] prediction for perturbations, W=[batch_size] weights for perturbations
        original_X = original_X.to(device) 
        Z = Z.to(device)
        Z.requires_grad_()
        Y = Y.to(device)
        W = W.to(device)
        
        #calculate model prediction -- val
        pred_g = model_g(Z)

        #calculate loss (for this batch)
        loss = 0
        if loss_fn is not None:
            loss = loss_fn(pred_g, Y, W)

        if gradmatch:
            #calculate input gradient of f
            if noise_type=='additive':
                perturb_X = original_X + Z
            if noise_type=='multiplicative':
                perturb_X = original_X * Z
                
            pred_f = model_f(perturb_X)
            input_grad_f = torch.autograd.grad(outputs=pred_f.sum(), inputs=Z, only_inputs=True, retain_graph=True)[0]
            #calculate input gradient of g
            input_grad_g = torch.autograd.grad(outputs=pred_g.sum(), inputs=Z, only_inputs=True, create_graph=True)[0]
            #add gradient term to loss 
            grad_norm = torch.norm(input_grad_f.detach_()-input_grad_g.detach_(), p=2)**2
            loss += grad_norm * alpha_gradmatch 
            
        #calculate runnning loss over epoch
        loss_over_epoch_val += loss.item()
        n_val_samples += Z.size(0)

        if debug:
            config.logger.debug(f'   batch_idx_val, batch_size: {batch_idx_val, Z.size(0)}')
            config.logger.debug(f'val_loss, batch-level: {loss.item()/Z.size(0)}')
            
        #back to epoch level 
        #calculate validation loss
        val_loss = loss_over_epoch_val/n_val_samples #loss per datapoint in validation set
        
        if debug:
            config.logger.debug(f'val_loss, epoch-level: {val_loss}')
            config.logger.debug(f'n_val_samples: {n_val_samples}')
    
    #decide whether current model is the best model yet, using val_loss
    if val_loss < best_val_loss:
        best_model_state = model_g.state_dict()
        best_val_loss = val_loss
        if debug: 
            config.logger.debug(f'---> new lowest validation loss, new best weights: \n{best_model_state}')
            
    if debug: 
        config.logger.debug(f'val_loss: {val_loss}')
        config.logger.debug(f'running best_val_loss: {best_val_loss}')
        config.logger.debug(f'n_val_samples: {n_val_samples}')
        
    return best_val_loss, best_model_state


