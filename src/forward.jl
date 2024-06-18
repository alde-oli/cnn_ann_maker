include("layers.jl")
include("activation.jl")

# Function: Linear transformation with weights and biases.
# Utility: Each neuron is connected to all neurons of the previous layer.
# Input: 2D tensor (batch_size, input_features)
# Output: 2D tensor (batch_size, output_features)
function forward(layer::Dense, input::Array{Float32,1})::Array{Float32,1}
	return layer.weights * input .+ layer.biases
end


# Function: Applies a nonlinear function.
# Utility: Allows the network to model nonlinear relationships.
# Input: 2D tensor (batch_size, input_features) or same as previous layer
# Output: Same as input
function forward(layer::Activation, input)
	if (layer.activation == "leaky_relu") || (layer.activation == "elu")
		return layer.activation_functions[layer.activation].(input, layer.alpha)
	else if layer.activation == "softmax"
		return layer.activation_functions[layer.activation](input, layer.dimension)
	else if haskey(layer.activation_functions, layer.activation)
		return layer.activation_functions[layer.activation].(input)
	else
		error("Activation function not found.")
	end
end


# Function: Randomly ignores a fraction of neurons during training.
# Utility: Prevents overfitting.
# Input: 2D tensor (batch_size, input_features) or same as previous layer
# Output: Same as input
function forward(layer::Dropout, input::AbstractArray; training::Bool=true)
	if !training
		return input
	end
	layer.mask = rand(size(input)) .< layer.probability
	scaled_input = input .* layer.mask
	output = scaled_input / (1.0 - layer.probability)
	
	return output
end



# Function: Normalizes activations of each batch.
# Utility: Speeds up training and stabilizes the network.
# Input: 2D tensor (batch_size, input_features) or same as previous layer
# Output: Same as input
function forward(layer::BatchNormalization, input)
end


# Function: Defines the shape of input data.
# Utility: Serves as the starting layer for the network.
# Input: N/A (defines the shape of the model's input)
# Output: Depends on the definition
function forward(layer::GaussianNoise, input)
end


# Function: Flattens input data into a 1D vector.
# Utility: Prepares data for dense layers.
# Input: Any shape (commonly 3D or 4D tensor)
# Output: 2D tensor (batch_size, flattened_features)
function forward(layer::Flatten, input)
end


# Function: Changes the shape of data without altering its content.
# Utility: Adapts data to specific layers.
# Input: Any shape
# Output: New specified shape
function forward(layer::Reshape, input)
end


# Function: Adds Gaussian noise to input data.
# Utility: Improves model robustness.
# Input: Any shape
# Output: Same as input
function forward(layer::GaussianNoise, input)
end


# Function: Multiplies inputs by random Gaussian variables.
# Utility: Prevents overfitting.
# Input: Any shape
# Output: Same as input
function forward(layer::GaussianDropout, input)
end


# Function: Preserves SELU activation properties after dropout.
# Utility: Used with SELU activations.
# Input: Any shape
# Output: Same as input
function forward(layer::AlphaDropout, input)
end


# Function: Applies convolutional filters.
# Utility: Extracts local features from images.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 4D tensor (batch_size, new_height, new_width, filters)
function forward(layer::Conv2D, input::Array{Float32,4})
end


# Function: Reduces dimensionality by taking the maximum.
# Utility: Decreases feature size while retaining important information.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 4D tensor (batch_size, pooled_height, pooled_width, channels)
function forward(layer::MaxPooling2D, input::Array{Float32,4})
end


# Function: Reduces dimensionality by taking the average.
# Utility: Decreases feature size while retaining important information.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 4D tensor (batch_size, pooled_height, pooled_width, channels)
function forward(layer::AveragePooling2D, input::Array{Float32,4})
end


# Function: Applies convolutional filters on 1D data.
# Utility: Extracts local features from sequences.
# Input: 3D tensor (batch_size, sequence_length, channels)
# Output: 3D tensor (batch_size, new_length, filters)
function forward(layer::Conv1D, input::Array{Float32,3})
end


# Function: Applies convolutional filters on 3D data.
# Utility: Extracts local features from volumes (e.g., videos).
# Input: 5D tensor (batch_size, depth, height, width, channels)
# Output: 5D tensor (batch_size, new_depth, new_height, new_width, filters)
function forward(layer::Conv3D, input::Array{Float32,5})
end


# Function: Takes the maximum over the entire height and width.
# Utility: Reduces each feature map to a single value.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 2D tensor (batch_size, channels)
function forward(layer::GlobalMaxPooling2D, input::Array{Float32,4})
end


# Function: Takes the average over the entire height and width.
# Utility: Reduces each feature map to a single value.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 2D tensor (batch_size, channels)
function forward(layer::GlobalAveragePooling2D, input::Array{Float32,4})
end