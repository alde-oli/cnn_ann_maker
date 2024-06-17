# abstract base struct for all layers
abstract type Layer end


# Function: Linear transformation with weights and biases.
# Utility: Each neuron is connected to all neurons of the previous layer.
# Input: 1D vector
# Output: 1D vector
struct Dense <: Layer
	weights::Array{Float64,2}
	bias::Array{Float64,1}
end


# Function: Applies a nonlinear function.
# Utility: Allows the network to model nonlinear relationships.
# Input: 1D vector
# Output: 1D vector
struct Activation <: Layer
	activation::Function
	alpha::Float64 # for LeakyReLU
end


# Function: Randomly ignores a fraction of neurons during training.
# Utility: Prevents overfitting.
# Input: 1D vector
# Output: 1D vector
struct Dropout <: Layer
	ratio::Float64
end


# Function: Normalizes activations of each batch.
# Utility: Speeds up training and stabilizes the network.
# Input: 1D vector
# Output: 1D vector
struct BatchNormalization <: Layer
	epsilon::Float64
	gamma::Array{Float64,1}
	beta::Array{Float64,1}
end


# Function: Defines the shape of input data.
# Utility: Serves as the starting layer for the network.
# Input: N/A (defines the shape of the model's input)
# Output: Depends on the definition
struct InputLayer <: Layer
	shape::Tuple
end


# Function: Flattens input data into a 1D vector.
# Utility: Prepares data for dense layers.
# Input: 2D matrix or 3D tensor
# Output: 1D vector
struct Flatten <: Layer
end


# Function: Changes the shape of data without altering its content.
# Utility: Adapts data to specific layers.
# Input: Any shape
# Output: New specified shape
struct Reshape <: Layer
	shape::Tuple
end


# Function: Adds Gaussian noise to input data.
# Utility: Improves model robustness.
# Input: 1D vector
# Output: 1D vector
struct GaussianNoise <: Layer
	stddev::Float64
end


# Function: Multiplies inputs by random Gaussian variables.
# Utility: Prevents overfitting.
# Input: 1D vector
# Output: 1D vector
struct GaussianDropout <: Layer
	ratio::Float64
end


# Function: Preserves SELU activation properties after dropout.
# Utility: Used with SELU activations.
# Input: 1D vector
# Output: 1D vector
struct AlphaDropout <: Layer
	ratio::Float64
end


# Function: Applies convolutional filters.
# Utility: Extracts local features from images.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 4D tensor (batch_size, new_height, new_width, filters)
struct Conv2D <: Layer
	filters::Int
	kernel_size::Tuple{Int,Int}
	strides::Tuple{Int,Int}
	padding::String
	activation::Function
end


# Function: Reduces dimensionality by taking the maximum.
# Utility: Decreases feature size while retaining important information.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 4D tensor (batch_size, pooled_height, pooled_width, channels)
struct MaxPooling2D <: Layer
	pool_size::Tuple{Int,Int}
	strides::Tuple{Int,Int}
	padding::String
end


# Function: Reduces dimensionality by taking the average.
# Utility: Decreases feature size while retaining important information.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 4D tensor (batch_size, pooled_height, pooled_width, channels)
struct AveragePooling2D <: Layer
	pool_size::Tuple{Int,Int}
	strides::Tuple{Int,Int}
	padding::String
end


# Function: Applies convolutional filters on 1D data.
# Utility: Extracts local features from sequences.
# Input: 3D tensor (batch_size, sequence_length, channels)
# Output: 3D tensor (batch_size, new_length, filters)
struct Conv1D <: Layer
	filters::Int
	kernel_size::Int
	strides::Int
	padding::String
	activation::Function
end


# Function: Applies convolutional filters on 3D data.
# Utility: Extracts local features from volumes (e.g., videos).
# Input: 5D tensor (batch_size, depth, height, width, channels)
# Output: 5D tensor (batch_size, new_depth, new_height, new_width, filters)
struct Conv3D <: Layer
	filters::Int
	kernel_size::Tuple{Int,Int,Int}
	strides::Tuple{Int,Int,Int}
	padding::String
	activation::Function
end


# Function: Takes the maximum over the entire height and width.
# Utility: Reduces each feature map to a single value.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 2D tensor (batch_size, channels)
struct GlobalMaxPooling2D <: Layer
end


# Function: Takes the average over the entire height and width.
# Utility: Reduces each feature map to a single value.
# Input: 4D tensor (batch_size, height, width, channels)
# Output: 2D tensor (batch_size, channels)
struct GlobalAveragePooling2D <: Layer
end
