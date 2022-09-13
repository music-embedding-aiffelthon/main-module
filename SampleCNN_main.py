from dataloader import mel_dataset
from torch.utils.data import DataLoader, random_split
from model.supervised_model import SampleCNN

import flax 
import flax.linen as nn
from flax.training import train_state

import jax
import numpy as np
import jax.numpy as jnp
import optax
from tqdm import tqdm
import os
import wandb
import matplotlib.pyplot as plt

wandb.init(
    project='SampleCNN',
    entity='aiffelthon'
)
    
    
def collate_batch(batch):
    x_train = [x for x, _ in batch]
    y_train = [y for _, y in batch]                  
        
    return np.array(x_train), np.array(y_train)


def init_state(model, shape, key, lr) -> train_state.TrainState:
    params = model.init({'params': key, 'dropout':key}, jnp.ones(shape))
    # Create the optimizer
    optimizer = optax.adam(learning_rate=lr)
    # Create a State
    return train_state.TrainState.create(
        apply_fn=model.apply,
        tx=optimizer,
        params=params)



@jax.jit
def train_step(state,
               inputs,
               labels,
               dropout_rng=None):
    
    inputs = jnp.expand_dims(inputs, axis=-1)
    def loss_fn(params):
        logits = SampleCNN().apply(
            params,
            inputs,
            rngs={"dropout": dropout_rng})
        
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
        return loss, logits
        
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    new_state = state.apply_gradients(grads=grads)
    

    return new_state, loss, accuracy
    
@jax.jit
def eval_step(state,
               inputs,
               labels,
               dropout_rng=None):
    
    inputs = jnp.expand_dims(inputs, axis=-1)
    logits = SampleCNN().apply(state.params,
                               inputs,
                               rngs={"dropout": dropout_rng})

    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=labels))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))

    return loss, accuracy
    

if __name__ == "__main__":
    batch_size = 128
    lr = 0.0001
    rng = jax.random.PRNGKey(303)
    
    # ---Load dataset---
    print("Loading dataset...")
    dataset_dir = os.path.join(os.path.expanduser('~'),'dataset')
    data = mel_dataset(dataset_dir)

    
    dataset_size = len(data)
    train_size = int(dataset_size * 0.7)
    validation_size = int(dataset_size * 0.2)
    test_size = dataset_size - (train_size+validation_size)
    
    train_dataset, validation_dataset, test_dataset = random_split(data, [train_size, validation_size, test_size])

    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    validation_dataloader = DataLoader(validation_dataset, batch_size=int(batch_size/4), shuffle=True, num_workers=0, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=int(batch_size/4), shuffle=True, num_workers=0, collate_fn=collate_batch)
    
    
    print(f'batch_size = {batch_size}')
    print(f'learning rate = {lr}')
    print(f'train_size = {train_size}')
    print(f'validation_size = {validation_size}')
    print(f'test_size = {test_size}')
    print('Data load complete!\n')

    # ---initializing model---
    model = SampleCNN()
    print("Initializing model....")
    state = init_state(model, next(iter(train_dataloader))[0].shape, jax.random.PRNGKey(353), lr)
    print("Initialize complete!!\n")
    
    
    
    rng = jax.random.PRNGKey(353)
    print(nn.tabulate(model, {'params': rng, 'dropout':rng})(next(iter(train_dataloader))[0]))
    # ---train model---
    epoch = 50
    # checkpoint_dir = str(input('checkpoint dir : '))

    for i in range(epoch):
        train_data = iter(train_dataloader)
        validation_data = iter(validation_dataloader)
        
        
        train_loss_mean = 0
        validation_loss_mean = 0
        
        train_accuarcy_mean = 0
        validation_accuarcy_mean = 0
        
        print(f'\nEpoch {i+1}')
        
        for j in range(len(train_dataloader)):
            rng, key = jax.random.split(rng)
            
            x, y = next(train_data)
            validation_x, validation_y = next(validation_data)
            
            state, train_loss, train_accuracy = train_step(state, x, y, dropout_rng=rng)
            validation_loss, validation_accuracy = eval_step(state, validation_x, validation_y, dropout_rng=rng)
            

            train_loss_mean += train_loss
            train_accuarcy_mean += train_accuracy
            
            validation_loss_mean += validation_loss
            validation_accuarcy_mean += validation_accuracy
            
            wandb.log({'train_loss' : train_loss,  'train_accuracy': train_accuracy, 'validation_loss' : validation_loss, 'validation_accuracy' : validation_accuracy})
            
            print(f'step : {j}/{len(train_dataloader)}, t_loss : {train_loss}, t_accuracy : {train_accuracy}, v_loss : {validation_loss}, v_accuracy : {validation_accuracy}',  end='\r')
            # if j % 100 == 0:
            #     checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=state, step=state.step, overwrite=True)
            
        print(f'epoch {i+1} - train average loss : {train_loss_mean/len(train_dataloader)} - train accuracy : {train_accuarcy_mean/len(train_dataloader)} validation average loss : {validation_loss_mean/len(train_dataloader)}, validation average accuracy : {validation_accuarcy_mean/len(train_dataloader)}')
    
    
    # --- test accuarcy ---
    test_data = iter(test_dataloader)
    test_loss_mean = 0
    test_accuarcy_mean = 0
    
    for i in range(len(test_dataloader)):
        
        test_x, test_y = next(test_dataloader)
        test_loss, test_accuracy = eval_step(state, test_x, test_y, dropout_rng=rng)
        
        test_loss_mean += test_loss
        test_accuarcy_mean += test_accuracy
        
    print(f'Test loss : {test_loss_mean/len(test_dataloader)}, Test accuarcy : {test_accuarcy_mean/len(test_dataloader)}\n\n')

        
        

wandb.finish()        
        




