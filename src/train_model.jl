using Pkg
using CSV
using DataFrames
using LinearAlgebra
using Statistics
using Plots
using YAML
using FileIO
using Images
using ImageTransformations
using Flux
using Flux: onehotbatch, onecold
using Base.Threads
using CUDA

include("input_management.jl")


function load_config_files()
    input_config_path = joinpath("config", "input_config.yaml")
    neuron_config_path = joinpath("config", "neuron_config.yaml")
    telemetry_config_path = joinpath("config", "telemetry_config.yaml")

    try
        input_config = YAML.load_file(input_config_path)
        neuron_config = YAML.load_file(neuron_config_path)
        telemetry_config = YAML.load_file(telemetry_config_path)
        return input_config, neuron_config, telemetry_config
    catch e
        println("An error occurred while loading the configuration files: $e")
        return nothing
    end
end


# Main function
function main()
    configs = load_config_files()
    if configs === nothing
        println("Failed to load configuration files.")
        return
    end
    
    input_config, neuron_config, telemetry_config = configs
    println("Configuration files loaded successfully.")
    
    data, labels = load_data(input_config)
    if data === nothing
        println("Failed to load data.")
        return
    end
    
    println("Data loaded and processed successfully.")
	println("Data type: ", typeof(data))
    show(data)
    if typeof(data) == Matrix{Any}
        println("Here is a preview of the tabular data:")
        println(first(data, 5))
    elseif typeof(data) == Vector{Any}
        println("Loaded file data. Number of matrices: ", length(data))
        println("Labels: ", labels)
    elseif typeof(data) == String
        println("Loaded text data: ", data[1:100], "...")
    elseif typeof(data) == Tuple
        println("Loaded audio or video data.")
    end
    # Continue with the rest of your machine learning code
end


main()
