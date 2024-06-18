using CSV
using DataFrames
using FileIO
using Images
using ImageTransformations
using Statistics
using YAML
using Flux
using Flux: onehotbatch, onecold
using Base.Threads
using CUDA



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



function identify_data_type(file_path::String)
    if endswith(lowercase(file_path), ".csv")
        df = CSV.File(file_path) |> DataFrame
        if all(occursin(".jpg", string(row[1])) || occursin(".png", string(row[1])) || occursin(".csv", string(row[1])) for row in eachrow(df))
            return "file_paths", df
        else
            return "tabular", df
        end
    elseif endswith(lowercase(file_path), ".txt")
        return "text", file_path
    elseif endswith(lowercase(file_path), ".wav") || endswith(lowercase(file_path), ".mp3")
        return "audio", file_path
    elseif endswith(lowercase(file_path), ".mp4") || endswith(lowercase(file_path), ".avi")
        return "video", file_path
    else
        return "unknown", nothing
    end
end



function normalize_data(data, normalization::String, use_gpu::Bool)
    if isa(data, Matrix)
        if normalization == "standard"
            return (data .- mean(data, dims=1)) ./ std(data, dims=1)
        elseif normalization == "minmax"
            min_vals = minimum(data, dims=1)
            max_vals = maximum(data, dims=1)
            return (data .- min_vals) ./ (max_vals .- min_vals)
        end
    elseif isa(data, Array)
        Threads.@threads eachindex(data) do i
            if normalization == "standard"
                data[i] = use_gpu ? CUDA.fill((data[i] .- mean(data[i])) ./ std(data[i])) : (data[i] .- mean(data[i])) ./ std(data[i])
            elseif normalization == "minmax"
                data[i] = use_gpu ? CUDA.fill((data[i] .- minimum(data[i])) ./ (maximum(data[i]) .- minimum(data[i]))) : (data[i] .- minimum(data[i])) ./ (maximum(data[i]) .- minimum(data[i]))
            end
        end
    else
        println("Data format not supported for normalization")
    end
    return data
end



function rotate_matrix(matrix, angle)
    rounded_angle = round(angle / 90) * 90
    rounded_angle = mod(rounded_angle, 360)

    if rounded_angle == 90
        return reverse(permutedims(matrix), dims=1)
    elseif rounded_angle == 180
        return reverse(reverse(matrix, dims=1), dims=2)
    elseif rounded_angle == 270
        return permutedims(reverse(matrix, dims=2))
    else
        return matrix
    end
end



function augment_data(data, augmentation::Dict, use_gpu::Bool)
    if augmentation["flip"]
        Threads.@threads for i in 1:length(data)
            data = vcat(data, [use_gpu ? CUDA.fill(reverse(d, dims=2)) : reverse(d, dims=2) for d in data])
        end
    end
    if augmentation["rotation"] != 0
        Threads.@threads for i in 1:length(data)
            data = vcat(data, [use_gpu ? CUDA.fill(rotate_matrix(d, augmentation["rotation"])) : rotate_matrix(d, augmentation["rotation"]) for d in data])
        end
    end
    return data
end



function flatten_data(data, use_gpu::Bool)
    flattened_data = Vector{Any}(undef, length(data))
    Threads.@threads eachindex(data) do i
        flattened_data[i] = use_gpu ? CUDA.fill(vec(data[i])) : vec(data[i])
    end
    return flattened_data
end



function load_file_dataset(df::DataFrame, use_gpu::Bool)
    data_matrices = Vector{Any}(undef, nrow(df))
    labels = Vector{Any}(undef, nrow(df))

    Threads.@threads for i in 1:nrow(df)
        row = df[i, :]
        file_path = joinpath(row[1])
        label = row[2]
        try
            if endswith(lowercase(file_path), ".csv")
                sub_df = CSV.File(file_path) |> DataFrame
                matrix = Matrix{Float32}(sub_df)
            else
                img = load(file_path)
                matrix = Float32.(channelview(img))
            end
            data_matrices[i] = use_gpu ? CuArray(matrix) : matrix
            labels[i] = label
        catch e
            println("An error occurred while processing the file $file_path: $e")
        end
    end

    return data_matrices, labels
end




function load_text_data(file_path::String)
    try
        text = read(file_path, String)
        return text
    catch e
        println("An error occurred while loading the text file $file_path: $e")
        return nothing
    end
end



function load_audio_data(file_path::String)
    try
        audio = load(file_path)
        return audio
    catch e
        println("An error occurred while loading the audio file $file_path: $e")
        return nothing
    end
end



function load_video_data(file_path::String)
    try
        video = load(file_path)
        return video
    catch e
        println("An error occurred while loading the video file $file_path: $e")
        return nothing
    end
end



function load_data(input_config)
    while true
        println("Please enter a valid input file path or type !STOP to quit:")
        data_path = readline()
        if data_path == "!STOP"
            return nothing, nothing
        end
        if isfile(data_path)
            data_type, data = identify_data_type(data_path)
            if data_type == "unknown"
                println("Unknown data type or invalid file.")
                continue
            end
            
            use_gpu = input_config["data_pipeline"]["use_gpu"]

            if data_type == "tabular"
                println("Loading tabular data...")
                labels = data[:, end]
                data = data[:, 1:end-1]
                data = Matrix(data)
            elseif data_type == "file_paths"
                println("Loading file data (images or matrices)...")
                data, labels = load_file_dataset(data, use_gpu)
            elseif data_type == "text"
                println("Loading text data...")
                data = load_text_data(data_path)
                return data, nothing
            elseif data_type == "audio"
                println("Loading audio data...")
                data = load_audio_data(data_path)
                return data, nothing
            elseif data_type == "video"
                println("Loading video data...")
                data = load_video_data(data_path)
                return data, nothing
            end

            if data_type == "tabular"
                data = Matrix(data)
            end

            if data_type in ["tabular", "file_paths"]
                data = normalize_data(data, input_config["preprocessing"]["normalization"], use_gpu)
            end

            if data_type == "file_paths" && (input_config["preprocessing"]["augmentation"]["flip"] || input_config["preprocessing"]["augmentation"]["rotation"] != 0)
                data = augment_data(data, input_config["preprocessing"]["augmentation"], use_gpu)
            end

            if data_type == "file_paths" && input_config["preprocessing"]["flatten"]
                data = flatten_data(data, use_gpu)
            end

            return data, labels
        else
            println("File not found: $data_path")
        end
    end
end

