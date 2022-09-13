import flax 
import flax.linen as nn
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import unfreeze, freeze

from utils.dataloader import mel_dataset
from torch.utils.data import DataLoader, random_split

import jax
import numpy as np
import jax.numpy as jnp
import optax

from torch.utils.tensorboard import SummaryWriter
import os
from utils.config_hook import yaml_config_hook
from tqdm import tqdm

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
key = jax.random.PRNGKey(42)

# --- config dir ---
config_dir = os.path.join(os.path.expanduser('~'),'module/config')     
config = yaml_config_hook(os.path.join(config_dir, 'config.yaml'))

# --- dataloader ---
dataset_dir = os.path.join(os.path.expanduser('~'), config['dataset_dir'])
data = mel_dataset(dataset_dir, 'total')
dataset_size = len(data)
train_size = int(dataset_size * 0.8)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(data, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)


# --- initialize parameters ---
if config['projector_target'] == 'freeze':
    if config['model_type'] == 'Conv1d':
        from model.Conv1d_model import Conv1d_VAE, Encoder
        model = Conv1d_VAE(dilation=config['dilation'],
                          latent_size=config['latent_size'])
    elif config['model_type'] == 'Conv2d':
        from model.Conv2d_model import Conv2d_VAE, Encoder
        model = Conv2d_VAE(dilation=config['dilation'],
                           latent_size=config['latent_size'])
elif config['projector_target'] == 'unfreeze':
    from model.Conv2d_model import Encoder
    model = Encoder(dilation=config['dilation'],
                           latent_size=config['latent_size'], linear=True)
        
init_params = model.init(key, jnp.ones((config['batch_size'], 48, 1876)))
    
                       
# --- load checkpoints ---
checkpoint_path = os.path.join(os.path.expanduser('~'), 'module/checkpoints')

if config['projector_target'] == 'freeze':
    try:                               
        checkpoint_path = os.path.join(checkpoint_path, f"freeze_latent_{int(config['pretrain_epoch']*len(train_dataloader))}")
    except:
        raise FileNotFoundError(f'No such file or directory: {checkpoint_path}')
elif config['projector_target'] == 'unfreeze':
    try:
        checkpoint_path = os.path.join(checkpoint_path, f"unfreeze_latent_{int(config['linear_evaluation_epoch']*len(train_dataloader))}")
    except:
        raise FileNotFoundError(f'No such file or directory: {checkpoint_path}')
                                       
checkpoint_params = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=init_params)

# --- adjust parameter dict to encoder ---
if config['projector_target'] == 'freeze':
    unfreeze_params = unfreeze(checkpoint_params)

    params = {}
    params['params'] = unfreeze_params['params']['encoder']
    params['batch_stats'] = unfreeze_params['batch_stats']['encoder']
    params = freeze(params)
    print(params.keys())
    
elif config['projector_target'] == 'unfreeze':
    params = checkpoint_params     
    print(params['params'].keys())
# --- encoder ---
@jax.jit
def encode(x):
    return Encoder(dilation=config['dilation'], linear=False).apply(params, x)

# --- create latent vector
def encoding_data(data):
    embed_list = []
    dataloader = iter(data)
    for i in tqdm(range(len(data)), desc="Encoding latent vector"):
        x, y = next(dataloader)
        x = np.expand_dims(x, axis=0)
        z = encode(x)
        z = jax.device_get(z)
        embed_list.append(z)
    return np.concatenate(embed_list, axis=0)                                       

embed_list = encoding_data(data) # (data_length, latent_size)


# --- execute tensorboard ---
label = [data.genre_keys[np.argmax(y)] for x, y in data]                                       
                                       
writer = SummaryWriter('log_dir')
writer.add_embedding(embed_list, metadata=label)
writer.close()
# os.system('tensorboard --logdir log_dir --host=0.0.0.0 --port 6020')
                                       


                       





