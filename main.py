# 2022-09-02 16:23 Seoul

# --- import dataset ---
from utils.dataloader import mel_dataset
from utils.losses import *
from torch.utils.data import DataLoader, random_split

# --- import model ---
from model.supervised_model import *

# --- import framework ---
import flax 
import flax.linen as nn
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import unfreeze, freeze
import jax
import numpy as np
import jax.numpy as jnp
import optax

from tqdm import tqdm
import os
import wandb
import matplotlib.pyplot as plt
from utils.config_hook import yaml_config_hook

from functools import partial

# --- Define config ---
config_dir = os.path.join(os.path.expanduser('~'),'module/config')     
config = yaml_config_hook(os.path.join(config_dir, 'config.yaml'))



# --- collate batch for dataloader ---
def collate_batch(batch):
    x_train = [x for x, _ in batch]
    y_train = [y for _, y in batch]                  
        
    return np.array(x_train), np.array(y_train)



# --- define init state ---
def init_state(model, x_shape, key, lr) -> train_state.TrainState:
    params = model.init({'params': key}, jnp.ones(x_shape))
    optimizer = optax.adam(learning_rate=lr)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=params)

# --- top_k accuarcy ---
@partial(jax.jit, static_argnames=['k'])
def top_k(logits, y,k):
    top_k = jax.lax.top_k(logits, k)[1]
    ts = jnp.argmax(y, axis=1)
    correct = 0
    for i in range(ts.shape[0]):
        b = (jnp.where(top_k[i,:] == ts[i], jnp.ones((top_k[i,:].shape)), 0)).sum()
        correct += b
    correct /= ts.shape[0]
    return correct

# --- define train_step ---
@jax.jit
def train_step(state, x):        
    def loss_fn(params):
        recon_x = model.apply(params, x)
        # kld_loss = kl_divergence(mean, logvar).mean()
        mse_loss = ((recon_x - x)**2).mean()
        # loss = mse_loss + kld_loss
        return mse_loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    return state.apply_gradients(grads=grads), loss

@jax.jit
def linear_freeze_train_step(enc_state, 
                      enc_batch, 
                      linear_state, 
                      x, y):    
    
    latent = Encoder(dilation=config['dilation'],
                    linear=False).apply({'params':enc_state, 'batch_stats':enc_batch}, x)
    
    def loss_fn(params):
        logits = linear_evaluation().apply(params, latent)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(linear_state.params)
    accuracy = top_k(logits, y, 1)
    top_k_accuracy = top_k(logits, y, config['top_k'])
    return linear_state.apply_gradients(grads=grads), loss, accuracy, top_k_accuracy


@jax.jit
def linear_unfreeze_train_step(state, x, y):    
      
    def loss_fn(params):
        logits = Encoder(dilation=config['dilation'],
                    linear=True).apply(params, x)        
        loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = top_k(logits, y, 1)
    top_k_accuracy = top_k(logits, y, config['top_k'])
    return state.apply_gradients(grads=grads), loss, accuracy, top_k_accuracy


# --- define eval step ---
@jax.jit
def eval_step(state, x):
    
    recon_x = model.apply(state.params, x)
    mse_loss = ((recon_x - x)**2).mean()
    
    return recon_x, mse_loss

@jax.jit
def linear_freeze_eval_step(enc_state, 
                      enc_batch, 
                      linear_state, 
                      x, y):
    
    latent = Encoder(dilation=config['dilation'],
                    linear=False).apply({'params':enc_state, 'batch_stats':enc_batch}, x)
    
    logits = linear_evaluation().apply(linear_state.params, latent)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
    accuracy = top_k(logits, y, 1)
    top_k_accuracy = top_k(logits, y, config['top_k'])

    return loss, accuracy, top_k_accuracy

@jax.jit
def linear_unfreeze_eval_step(state, 
                      x, y):
    
    logits = Encoder(dilation=config['dilation'],
                    linear=True).apply(state.params, x)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
    accuracy = top_k(logits, y, 1)
    top_k_accuracy = top_k(logits, y, config['top_k'])

    return loss, accuracy, top_k_accuracy



if __name__ == "__main__":
    batch_size = config['batch_size']
    lr = config['learning_rate']
    dilation = config['dilation']
    
    if config['model_type'] == 'Conv1d':
        
        from model.Conv1d_model import Conv1d_VAE, Encoder        
        model = Conv1d_VAE(dilation=config['dilation'],
                          latent_size=config['latent_size'])
        
    elif config['model_type'] == 'Conv2d':
        
        from model.Conv2d_model import Conv2d_VAE, Encoder
        model = Conv2d_VAE(dilation=config['dilation'],
                          latent_size=config['latent_size'])
        
    else: 
        raise Exception('Input Correct model type. Conv1d, Conv2d.')
    
    rng = jax.random.PRNGKey(303)
    
    
    # ---Load dataset---
    dataset_dir = os.path.join(os.path.expanduser('~'),config['dataset_dir'])            

    print("Loading dataset...")    
    data = mel_dataset(dataset_dir, 0)
    print(f'Loaded data : {len(data)}\n')
    
    dataset_size = len(data)
    train_size = int(dataset_size * 0.8)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=int(batch_size/4), shuffle=True, num_workers=0, collate_fn=collate_batch)
    
    print(f"batch_size = {config['batch_size']}")
    print(f"learning rate = {config['learning_rate']}")
    print(f"train_size = {train_size}")
    print(f"test_size = {test_size}")
    
    
    print('Data load complete!\n')
    print(nn.tabulate(model, rngs={'params': rng})(next(iter(train_dataloader))[0]))
    
    
    
    # ---initializing model---
    print("Initializing model....")
    state = init_state(model, 
                       next(iter(train_dataloader))[0].shape, 
                       rng, 
                       lr)
    
    print("Initialize complete!!\n")
    
    
    
    # ---train model---
    wandb.init(
    project=config['model_type'],
    entity='aiffelthon',
    config=config
    )
    for i in range(config['pretrain_epoch']):
        train_data = iter(train_dataloader)
        test_data = iter(test_dataloader)
        
        # train_loss_mean = 0
        # test_loss_mean = 0
        
        print(f'\nEpoch {i+1}')
        
        for j in tqdm(range(len(train_dataloader))):
            x, y = next(train_data)
            test_x, test_y = next(test_data)
            
            x = (x / 200) + 0.5 
            test_x = (test_x / 200) + 0.5
            
            state, train_loss = train_step(state, x)           
            recon_x, test_loss = eval_step(state, test_x)
            wandb.log({'train_loss' : train_loss, 'test_loss' : test_loss})
#             train_loss_mean += train_loss
#             test_loss_mean += test_loss
            
            
            if j % 100 == 0:
                
                recon_x = recon_x.reshape(recon_x.shape[0], x.shape[1], x.shape[2])       
                
                fig1, ax1 = plt.subplots()
                im1 = ax1.imshow(recon_x[0], aspect='auto', origin='lower', interpolation='none')
                fig1.colorbar(im1)
                fig1.savefig('recon.png')
                plt.close(fig1)

                fig2, ax2 = plt.subplots()
                im2 = ax2.imshow(test_x[0], aspect='auto', origin='lower', interpolation='none')
                fig2.colorbar(im2)
                fig2.savefig('x.png')
                plt.close(fig2)
                
                wandb.log({'reconstruction' : [
                            wandb.Image('recon.png')
                            ], 
                           'original image' : [
                            wandb.Image('x.png')
                            ]})
            

    print('Pre train complete!\n\n\n')
    
    
    
    checkpoints.save_checkpoint(ckpt_dir=config['checkpoints_path'], target=state.params,  prefix='genre_encode0', step=state.step)  
    
    # --- linear evaluation, if true. ---
    if config['linear_evaluation']:
        print('Linear evalutaion step.')
              
        if config['freeze_encoder']:
            enc_state = state.params['params']['encoder']
            enc_batch = state.params['batch_stats']['encoder']
            linear_state = init_state(linear_evaluation(), (config['batch_size'], config['latent_size']), rng, config['learning_rate'])
        
        
        # --- changing parameters ---
        else:
            
            linear_init = linear_evaluation().init(rng, jnp.ones((config['batch_size'], config['latent_size'])))        
            
            enc_unfreeze_variable = unfreeze(state.params)
            
            
            enc_state = enc_unfreeze_variable['params']['encoder']
            enc_batch = enc_unfreeze_variable['batch_stats']['encoder']
            
            
            params = {}
            params['params'] = enc_state
            params['batch_stats'] = enc_batch
            
            params['params']['linear_hidden_layer'] = linear_init['params']['linear_hidden_layer']
            params['params']['linear_classification'] = linear_init['params']['linear_classification']
            
            
            params = freeze(params)
            optimizer = optax.adam(learning_rate=lr)
            
            state = train_state.TrainState.create(
                    apply_fn=Encoder(dilation=config['dilation'], linear=True).apply,
                    tx=optimizer,
                    params=params)
                                           
        
        
        for i in range(config['linear_evaluation_epoch']):
            print(f'\nEpoch {i+1}')
            train_data = iter(train_dataloader)
            test_data = iter(test_dataloader)
            
            
#             linear_train_loss = 0
#             linear_test_loss = 0
            
#             linear_train_accuarcy = 0
#             linear_test_accuarcy = 0
            
#             linear_train_top_5_accuarcy = 0
#             linear_test_top_5_accuarcy = 0
            
            for j in tqdm(range(len(train_dataloader))):
                
                x, y = next(train_data)
                test_x, test_y = next(test_data)

                x = (x / 200)  + 0.5
                test_x = (test_x / 200) + 0.5
                if config['freeze_encoder']:
                    linear_state, train_loss, train_accuarcy, train_top_k_accuarcy = linear_freeze_train_step(enc_state, enc_batch, linear_state, x, y)
                    test_loss, test_accuarcy, test_top_k_accuarcy = linear_freeze_eval_step(enc_state, enc_batch, linear_state, test_x, test_y)
                
                else:
                    state, train_loss, train_accuarcy, train_top_k_accuarcy = linear_unfreeze_train_step(state, x, y)
                    test_loss, test_accuarcy, test_top_k_accuarcy = linear_unfreeze_eval_step(state, test_x, test_y)
                    

                wandb.log({'linear_train_loss' : train_loss, 
                           'linear_test_loss' : test_loss, 
                           'linear_train_accuarcy':train_accuarcy,
                           'linear_test_accuarcy':test_accuarcy,
                           'linear_train_top_k_accuarcy': train_top_k_accuarcy,
                           'linear_test_top_k_accuarcy': test_top_k_accuarcy})
                
#                 linear_train_loss += train_loss
#                 linear_test_loss += test_loss

#                 linear_train_accuarcy += train_accuarcy
#                 linear_test_accuarcy += test_accuarcy
                
#                 linear_train_top_5_accuarcy += train_top_5_accuarcy
#                 linear_test_top_5_accuarcy += test_top_5_accuarcy

        
        checkpoints.save_checkpoint(ckpt_dir=config['checkpoints_path'], target=state.params,  prefix='unfreeze_latent_', step=state.step)  

wandb.finish()
