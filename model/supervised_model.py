import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional


# SampleCNN implementation

class SampleCNN(nn.Module):
    
    deterministic:bool = False
    
    @nn.compact
    def __call__(self, x):
        
        x = jnp.expand_dims(x, axis=-1)
        
        # 1
        x = nn.Conv(features=128, kernel_size=(3,3), strides=(1,3), padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        
        # 2 
        x = nn.Conv(features=128, kernel_size=(3,3), strides=(1,2), padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        # 3
        x = nn.Conv(features=128, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3,3), strides=(3,3))

        # 4
        x = nn.Conv(features=256, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        # 5
        x = nn.Conv(features=256, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))

        # 6
        x = nn.Conv(features=256, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))

        # 7
        x = nn.Conv(features=512, kernel_size=(3,3), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        x = nn.Dropout(rate=0.5)(x, deterministic=self.deterministic)

        # 8
        x = nn.Conv(features=256, kernel_size=(3,3), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        # 9
        x = nn.Conv(features=128, kernel_size=(3,3), strides=1, padding='same')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        
        # 10
        x = nn.Conv(features=128, kernel_size=(3,3), strides=1, padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        # x = nn.max_pool(x, window_shape=(3,), strides=(3,))
        
        
        # 11
        x = nn.Conv(features=64, kernel_size=(3,3), strides=1, padding='valid')(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=self.deterministic)
        
        # fc
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512)(x)
        x = nn.Dense(512)(x)
        x = nn.Dense(30)(x)
        return x 

    
class linear_evaluation(nn.Module):
    
    n_features:int = 30
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512, name= 'linear_hidden_layer')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.Dense(self.n_features, name='linear_classification')(x)
        
        return x
            
    

if __name__=='__main__':
    
    test_input = jnp.ones((12, 48, 1876))
    test_label = jnp.ones((12, 30))
    
    test_latent = jnp.ones((12, 20))
    
    key = jax.random.PRNGKey(32)
    
    # --- SampleCNN model ---
    params = SampleCNN().init({'params': key, 'dropout':key}, test_input)
    result = SampleCNN().apply(params, test_input, rngs={"dropout": key})
    
    # --- linear_evaluation model --
    params = linear_evaluation().init({'params': key}, test_latent)
    result = linear_evaluation().apply(params, test_latent)
    
    print('test complete!')