import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std    

class Encoder(nn.Module):
    
    linear:bool=False
    dilation:bool=False
    latent_size:int=512
    linear_hidden_layer:int=512
    n_features:int=30
    
    @nn.compact
    def __call__(self, x):
        if self.dilation:
            x = nn.Conv(512,kernel_size=(3,), strides=1, padding='same', kernel_dilation=1)(x)
        else:
            x = nn.Conv(512,kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:
            x = nn.Conv(256,kernel_size=(3,), strides=2, padding='same', kernel_dilation=2)(x)
        else:
            x = nn.Conv(256,kernel_size=(3,), strides=2, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:
            x = nn.Conv(128,kernel_size=(3,), strides=1, padding='same', kernel_dilation=4)(x)
        else:
            x = nn.Conv(128,kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        x = nn.Conv(128,kernel_size=(3,), strides=2, padding='same', kernel_dilation=8)(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:
            x = nn.Conv(64,kernel_size=(3,), strides=1, padding='same', kernel_dilation=8)(x)
        else:
            x = nn.Conv(64,kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        
        if self.dilation:
            x = nn.Conv(32,kernel_size=(3,), strides=2, padding='same',kernel_dilation=16)(x)
        else:
            x = nn.Conv(32,kernel_size=(3,), strides=2, padding='same')(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:                
            x = nn.Conv(1,kernel_size=(3,), strides=1, padding='same',kernel_dilation=32)(x)
        else:
            x = nn.Conv(1,kernel_size=(3,), strides=1, padding='same')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        x = nn.normalization.BatchNorm(True)(x)

        z = nn.Dense(512, name='latent_vector')
                
        if self.linear:
            z = nn.Dense(self.linear_hidden_layer, name='linear_hidden_layer')(z)    
            z = jax.nn.leaky_relu(z)
            z = nn.Dense(self.n_features, name='linear_classification')(z)
        
        
        return z
    
class Decoder(nn.Module):

    recon_shape : int = 1876
    dilation:bool=False
    @nn.compact
    def __call__(self, x):
        
        if self.dilation:            
            x = nn.ConvTranspose(64, kernel_size=(3,), strides=[1,],kernel_dilation=(16,))(x)
        else:
            x = nn.ConvTranspose(64, kernel_size=(3,), strides=[1,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)

        if self.dilation:            
            x = nn.ConvTranspose(128, kernel_size=(3,), strides=[2,], kernel_dilation=(8,))(x)
        else:
            x = nn.ConvTranspose(128, kernel_size=(3,), strides=[2,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:          
            x = nn.ConvTranspose(256, kernel_size=(3,), strides=[2,], kernel_dilation=(8,))(x)
        else:
            x = nn.ConvTranspose(256, kernel_size=(3,), strides=[2,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:          
            x = nn.ConvTranspose(512, kernel_size=(3,), strides=[3,],  kernel_dilation=(4,))(x)
        else:
            x = nn.ConvTranspose(512, kernel_size=(3,), strides=[3,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)

        if self.dilation:
            x = nn.ConvTranspose(1024, kernel_size=(3,), strides=[2,], kernel_dilation=(2,))(x)
        else:
            x = nn.ConvTranspose(1024, kernel_size=(3,), strides=[2,])(x)
        x = nn.relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        if self.dilation:
            x = nn.ConvTranspose(self.recon_shape, kernel_size=(3,), strides=[2,], kernel_dilation=(1,))(x)
        else:
            x = nn.ConvTranspose(self.recon_shape, kernel_size=(3,), strides=[2,])(x)
        

        
        
        
        return x
        


    
class Conv1d_VAE(nn.Module):
    dilation:bool=False
    latent_size:int=512
    linear_hidden_layer:int=512
    n_features:int=30
    
    def setup(self):
        self.encoder = Encoder(dilation=self.dilation, 
                               linear=False, 
                               latent_size=self.latent_size,
                               n_features=self.n_features)
        
        self.decoder = Decoder(dilation=self.dilation)

    def __call__(self, x, z_rng):
     
        z = self.encoder(x) 
        recon_x = self.decoder(z)
        return recon_x
    
if __name__=='__main__':
    
    test_input = jnp.ones((12, 48, 1876))
    test_label = jnp.ones((12, 30))
    
    test_latent = jnp.ones((12, 20))

    key = jax.random.PRNGKey(32)
    
    params = Conv1d_VAE(dilation=True).init({'params': key}, test_input , key)
    result = Conv1d_VAE(dilation=True).apply(params, test_input, key)

    params = Conv1d_VAE(dilation=False).init({'params': key}, test_input , key)
    result = Conv1d_VAE(dilation=False).apply(params, test_input, key)

    print('test complete!')