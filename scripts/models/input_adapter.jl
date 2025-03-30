using Flux

include("rnn_model.jl")
include("lstm_model.jl")
include("gru_model.jl")
include("stacked_model.jl")

# First define base sequential model type
const BaseSequentialModel = Union{RNN, LSTM, GRU, StackedModel}

"""
Wrapper structure that adapts variable input sizes to fixed RNN input size
"""
mutable struct InputAdapter
    preprocessor::Dict{Int, Chain}
    target_size::Int
    model::BaseSequentialModel  # Accepts all base model types
end

Flux.@functor InputAdapter # This struct can be optimized end to end

# Add property delegation for state
Base.getproperty(m::InputAdapter, name::Symbol) = 
    name == :state ? getfield(m, :model).state : getfield(m, name)

Base.setproperty!(m::InputAdapter, name::Symbol, value) = 
    name == :state ? setfield!(getfield(m, :model), :state, value) : setfield!(m, name, value)

"""
Constructor for InputAdapter
"""
function InputAdapter(model::BaseSequentialModel, target_size::Int)
    return InputAdapter(
        Dict{Int, Chain}(),
        target_size,
        model
    )
end

"""
Get or create preprocessor for specific input dimension
"""
function get_preprocessor!(model::InputAdapter, input_dim::Int)
    if !haskey(model.preprocessor, input_dim)
        model.preprocessor[input_dim] = Chain(
            Dense(input_dim, model.target_size), # These parameters can be optimized end to end
            relu
        )
    end
    return model.preprocessor[input_dim]
end

"""
Forward pass with automatic input adaptation
"""
function (m::InputAdapter)(x::AbstractVecOrMat{T}) where T <: Real
    input_dim = size(x, 1)
    
    # Get preprocessor for this input size
    preprocessor = get_preprocessor!(m, input_dim)
    
    # Process each column (time step)
    processed =  similar(x, m.target_size, size(x, 2))
    
     # Process each time step while maintaining matrix format
     for t in 1:size(x, 2)
        processed[:, t] = preprocessor(x[:, t])
    end

    # Forward through RNN
    return m.model(processed)
end

# Helper to reset RNN state
function Flux.reset!(m::InputAdapter)
    Flux.reset!(m.model)
end