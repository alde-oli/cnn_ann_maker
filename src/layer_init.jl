include("layer_struct.jl")


# Function: Linear transformation with weights and biases.
# Utility: Each neuron is connected to all neurons of the previous layer.
# Input: 2D tensor (batch_size, input_features)
# Output: 2D tensor (batch_size, output_features)
function Dense(init_method::String, input_shape::Tuple, layer_size::Int)::Dense
	weights = init_weights(init_method, input_shape, layer_size)
	biases = init_biases(init_method, layer_size)
	return Dense(weights, biases)
end



# Function: Applies a nonlinear function.
# Utility: Allows the network to model nonlinear relationships.
# Input: same as previous layer
# Output: Same as input



# Function: Randomly ignores a fraction of neurons during training.
# Utility: Prevents overfitting.
# Input: 2D tensor (batch_size, input_features) or same as previous layer
# Output: Same as input
function Dropout(probability::Float32)::Dropout
	return Dropout(probability, nothing)
end



# Function: Normalizes activations of each batch.
# Utility: Speeds up training and stabilizes the network.
# Input: 2D tensor (batch_size, input_features) or same as previous layer
# Output: Same as input




# Function: Defines the shape of input data.
# Utility: Serves as the starting layer for the network.
# Input: N/A (defines the shape of the model's input)
# Output: Depends on the definition



# Function: Flattens input data into a 1D vector.
# Utility: Prepares data for dense layers.
# Input: Any shape (commonly 3D or 4D tensor)
# Output: 2D tensor (batch_size, flattened_features)



# Function: Changes the shape of data without altering its content.
# Utility: Adapts data to specific layers.
# Input: Any shape
# Output: New specified shape



# Function: Adds Gaussian noise to input data.
# Utility: Improves model robustness.
# Input: Any shape
# Output: Same as input



# Function: Multiplies inputs by random Gaussian variables.
# Utility: Prevents overfitting.
# Input: Any shape
# Output: Same as input



# Function: Preserves SELU activation properties after dropout.
# Utility: Used with SELU activations.
# Input: Any shape
# Output: Same as input



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