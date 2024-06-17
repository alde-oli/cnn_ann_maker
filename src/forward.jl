include("layers.jl")

# Function: Linear transformation with weights and biases.
# Utility: Each neuron is connected to all neurons of the previous layer.
# Input: 1D vector
# Output: 1D vector



# Function: Applies a nonlinear function.
# Utility: Allows the network to model nonlinear relationships.
# Input: 1D vector
# Output: 1D vector



# Function: Randomly ignores a fraction of neurons during training.
# Utility: Prevents overfitting.
# Input: 1D vector
# Output: 1D vector



# Function: Normalizes activations of each batch.
# Utility: Speeds up training and stabilizes the network.
# Input: 1D vector
# Output: 1D vector



# Function: Defines the shape of input data.
# Utility: Serves as the starting layer for the network.
# Input: N/A (defines the shape of the model's input)
# Output: Depends on the definition



# Function: Flattens input data into a 1D vector.
# Utility: Prepares data for dense layers.
# Input: 2D matrix or 3D tensor
# Output: 1D vector



# Function: Changes the shape of data without altering its content.
# Utility: Adapts data to specific layers.
# Input: Any shape
# Output: New specified shape



# Function: Adds Gaussian noise to input data.
# Utility: Improves model robustness.
# Input: 1D vector
# Output: 1D vector


# Function: Multiplies inputs by random Gaussian variables.
# Utility: Prevents overfitting.
# Input: 1D vector
# Output: 1D vector



# Function: Preserves SELU activation properties after dropout.
# Utility: Used with SELU activations.
# Input: 1D vector
# Output: 1D vector



# Function: Applies convolutional filters.
# Utility: Extracts local features from images.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 4D tensor (batch_size, new_height, new_width, filters)



# Function: Reduces dimensionality by taking the maximum.
# Utility: Decreases feature size while retaining important information.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 4D tensor (batch_size, pooled_height, pooled_width, channels)



# Function: Reduces dimensionality by taking the average.
# Utility: Decreases feature size while retaining important information.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 4D tensor (batch_size, pooled_height, pooled_width, channels)



# Function: Applies convolutional filters on 1D data.
# Utility: Extracts local features from sequences.
# Input: 3D tensor (batch_size, sequence_length, channels)
# Output: 3D tensor (batch_size, new_length, filters)



# Function: Applies convolutional filters on 3D data.
# Utility: Extracts local features from volumes (e.g., videos).
# Input: 5D tensor (batch_size, depth, height, width, channels)
# Output: 5D tensor (batch_size, new_depth, new_height, new_width, filters)



# Function: Takes the maximum over the entire height and width.
# Utility: Reduces each feature map to a single value.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 2D tensor (batch_size, channels)



# Function: Takes the average over the entire height and width.
# Utility: Reduces each feature map to a single value.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 2D tensor (batch_size, channels)
