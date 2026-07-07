import tensorflow as tf

def create_model(input_dim):
    """
    Creates a simple neural network for binary intrusion detection.
    Architecture:
      - Input Layer
      - Dense Layer (64 neurons, ReLU activation)
      - Dropout (20% to prevent overfitting)
      - Dense Layer (32 neurons, ReLU activation)
      - Dropout (20% to prevent overfitting)
      - Output Layer (1 neuron, Sigmoid activation)
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model with Adam optimizer, binary crossentropy loss, and accuracy metric
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_model_weights(model):
    """Returns the weights of the model as a list of NumPy arrays."""
    return model.get_weights()

def set_model_weights(model, weights):
    """Sets the weights of the model."""
    model.set_weights(weights)
