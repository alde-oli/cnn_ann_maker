function sigmoid(x::Float64)::Float64
	return 1 / (1 + exp(-x))
end

function tanh(x::Float64)::Float64
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
end

function relu(x::Float64)::Float64
	return max(0, x)
end

function elu(x::Float64, alpha::Float64)::Float64
	return x < 0 ? alpha * (exp(x) - 1) : x
end

function leaky_relu(x::Float64, alpha::Float64)::Float64
	return x < 0 ? alpha * x : x
end

function softmax(x::AbstractArray, axis::Int)::AbstractArray
	return exp.(x) / sum(exp.(x), dims=axis)
end

activation_functions = Dict(
	"sigmoid" => sigmoid,
	"tanh" => tanh,
	"relu" => relu,
	"elu" => elu,
	"leaky_relu" => leaky_relu,
	"softmax" => softmax
)