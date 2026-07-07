import numpy as np

def poison_weights(weights, attack_type="sign_flip"):
    """
    Simulates a malicious client by poisoning/manipulating the model weights.
    
    Parameters:
    - weights: List of NumPy arrays (model weights).
    - attack_type: The type of weight poisoning attack to run.
      Options: 'sign_flip', 'random_noise', 'zero_out', 'scale_distortion'
      
    Returns:
    - List of manipulated NumPy arrays.
    """
    poisoned_weights = []
    
    for w in weights:
        if attack_type == "sign_flip":
            # Flip the sign of all parameters
            poisoned_weights.append(-1.0 * w)
        elif attack_type == "random_noise":
            # Add high-variance noise to destroy trained values
            noise = np.random.normal(0, 1.0, size=w.shape)
            poisoned_weights.append(w + noise)
        elif attack_type == "zero_out":
            # Set all parameters to 0
            poisoned_weights.append(np.zeros_like(w))
        elif attack_type == "scale_distortion":
            # Distort weights by scaling them up by a large factor
            poisoned_weights.append(w * 10.0)
        else:
            # No modification
            poisoned_weights.append(w)
            
    return poisoned_weights

def poison_labels(y_train):
    """
    Simulates data poisoning by flipping the binary labels.
    All normal instances (0) become attacks (1), and vice versa.
    
    Parameters:
    - y_train: NumPy array of labels.
    
    Returns:
    - NumPy array of flipped labels.
    """
    # Flip binary labels: 0 -> 1 and 1 -> 0
    return 1.0 - y_train
