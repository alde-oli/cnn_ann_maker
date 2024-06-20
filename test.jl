abstract type Layer end

mutable struct Dropout <: Layer
	probability::Float32
	mask::Union{Nothing, AbstractArray{Bool}}
end

function forward(layer::Dropout, input::AbstractArray; training::Bool=true)
	if !training || layer.probability == 0.0
		return input
	elseif layer.probability == 1.0
		error("Dropout probability is 1.0, division by zero.")
	end
	layer.mask = rand(size(input)...) .< layer.probability
	scaled_input = input .* layer.mask
	output = scaled_input / (1.0 - layer.probability)
	
	return output
end


dropout = Dropout(0.5, nothing)

input = rand(3, 3)

println("Input:")
println(round.(input, digits=2))

output = forward(dropout, input, training=true)

println("Output:")
println(round.(output, digits=2))

println("Mask:")
println(dropout.mask)