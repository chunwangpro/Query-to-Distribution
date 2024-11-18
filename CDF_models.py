import tensorflow as tf
from tensorflow.keras import layers


def ResNetCDFLayer(input_layer, num_residual_blocks=4, hidden_dim=10, output_dim=1):
    """
    A ResNet-style layer with residual connections.

    Parameters
    ----------
    input_layer : tf.Tensor
        Input tensor for the layer.
    num_residual_blocks : int
        Number of residual blocks to stack.
    hidden_dim : int
        Number of hidden units in each Dense layer.
    output_dim : int
        Number of output units (e.g., 1 for a CDF value).

    Returns
    -------
    tf.Tensor
        Output tensor with shape `(batch_size, output_dim)`.
    """

    def residual_block(x, hidden_dim):
        residual = x
        x = layers.Dense(hidden_dim, activation="relu")(x)
        x = layers.Dense(hidden_dim, activation=None)(x)  # No activation in the second layer
        x = layers.Add()([x, residual])  # Add skip connection
        x = layers.Activation("relu")(x)  # Apply activation after addition
        return x

    # Initial dense layer
    x = layers.Dense(hidden_dim, activation="relu")(input_layer)

    # Stack residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, hidden_dim)

    # Final dense layer for output
    output = layers.Dense(output_dim, activation="sigmoid")(x)
    return output
