#=
###############################################
#                                             #
#           RNN Model Implementation          #
#                    BOB                      #
#                                             #
###############################################

This file implements a Recurrent Neural Network (RNN) model
for sequence (next character) prediction based on the dataset generated by ALICE.

Author: Do Viet Anh
=#

using Flux
using LinearAlgebra


# Define the RNN cell structure
struct RNNCell{T}
    w::Matrix{T}  # Input-to-hidden weights
    u::Matrix{T}  # Hidden-to-hidden weights
    b::Vector{T}  # Bias
end

Flux.@functor RNNCell

"""
Initialize an RNN cell with given input and hidden sizes
"""
function RNNCell(input_size::Int, hidden_size::Int; init=Flux.glorot_uniform)
    return RNNCell(
        init(hidden_size, input_size),
        init(hidden_size, hidden_size),
        init(hidden_size)
    )
end

"""
Forward pass for the RNN cell
"""
function (m::RNNCell)(h::AbstractVector, x::AbstractVector)
    return tanh.(m.w * x .+ m.u * h .+ m.b)
end

# Define the full RNN model structure
struct RNN
    cell::RNNCell
    output::Chain
end

Flux.@functor RNN

"""
Initialize the full RNN model
"""
function RNN(input_size::Int, hidden_size::Int, output_size::Int)
    RNN(
        RNNCell(input_size, hidden_size),
        Chain(Dense(hidden_size, output_size), softmax)
    )
end

"""
Forward pass for the full RNN model
"""
function (m::RNN)(x::AbstractMatrix)
    h = zeros(Float32, size(m.cell.u, 1))
    outputs = map(1:size(x,2)) do t
        h = m.cell(h, x[:, t])
        m.output(h)
    end
    return hcat(outputs...)
end