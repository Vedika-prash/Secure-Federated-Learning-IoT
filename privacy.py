import numpy as np

def add_dp_noise(weights, noise_level=0.01):
    """
    Applies Differential Privacy by adding small Gaussian noise to model weights.
    
    Parameters:
    - weights: List of NumPy arrays (model weights).
    - noise_level: Standard deviation of the Gaussian noise.
    
    Returns:
    - List of NumPy arrays with added noise.
    """
    if noise_level <= 0.0:
        return weights
        
    privatized_weights = []
    for w in weights:
        # Generate Gaussian noise matching the shape of the weight array
        noise = np.random.normal(0, noise_level, size=w.shape)
        # Add noise to weights
        privatized_weights.append(w + noise)
        
    return privatized_weights
