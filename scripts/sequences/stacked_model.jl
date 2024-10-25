using Flux
using LinearAlgebra

include("rnn_model.jl")
include("lstm_model.jl")
include("gru_model.jl")


# Stacked Model that can combine different types of recurrent layers
mutable struct StackedModel
    layers::Vector{Union{RNN, LSTM, GRU}}  # Vector of existing model types
    output::Chain
    state::Vector{Float32}  # Added for pipeline compatibility, jsut a dummy field
end

Flux.@functor StackedModel

"""
Constructor for creating a StackedModel with specified layer types and sizes
"""
function StackedModel(input_size::Int, layer_specs::Vector{Tuple{Symbol, Int}}, output_size::Int)
    layers = []
    current_size = input_size
    
    for (layer_type, hidden_size) in layer_specs
        if layer_type == :RNN
            push!(layers, RNN(current_size, hidden_size, hidden_size))
        elseif layer_type == :LSTM
            push!(layers, LSTM(current_size, hidden_size, hidden_size))
        elseif layer_type == :GRU
            push!(layers, GRU(current_size, hidden_size, hidden_size))
        else
            error("Unknown layer type: $layer_type")
        end
        current_size = hidden_size
    end
    
    output = Chain(Dense(current_size, output_size))

    # Initialize state with size of last layer for compatibility, just a dummy variable
    # This is an ad-hoc quick solution for backward compatibility. May refactor later.
    state = zeros(Float32, current_size)
    
    return StackedModel(layers, output, state)
end

"""
Forward pass for the StackedModel
Takes an input matrix where each column is a token (timestep)
Returns the output after passing through all layers
"""
function (m::StackedModel)(x::AbstractMatrix)
    # Initialize input for first layer
    current_input = copy(x)
    
    # Process through each layer in sequence
    for layer in m.layers
        # Each layer's output becomes input for the next layer
        current_input = layer(current_input)
    end
    
    # Apply final output transformation (in the last layer)
    return m.output(current_input)
end

"""
Reset all layers in the StackedModel.
Each layer handles its own state reset.
"""
function Flux.reset!(m::StackedModel)
    for layer in m.layers
        Flux.reset!(layer)
    end
end