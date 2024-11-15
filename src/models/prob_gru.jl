export GRU, GRUCell

using Distributions

struct GRUCell{T}
    w::Matrix{T}  # Input weights
    u::Matrix{T}  # Hidden weights
    b::Vector{T}  # Bias
end

Flux.@functor GRUCell

# GRU forward pass
function (m::GRUCell{T})(h::AbstractMatrix{T}, x::AbstractMatrix{T}) where {T<:Real}
    # Get parts of weights/biases
    w = _exc(m.w, 1, 3)  # exclude first part
    b = _exc(m.b, 1, 3)
    u = _exc(m.u, 1, 3)

    println("Size w: ", size(w))
    println("Size b: ", size(b))
    println("Size u: ", size(u))
    println("Size h: ", size(h))
    println("Size x: ", size(x))

    # Transform input AND previous state
    g = w*x .+ u*h .+ b  # matrix multiplication

    # Split into two parts
    ĥ, z = _expand(g, Val(2))

    # GRU update using previous state AND candidate hidden state
    h_new = (1 .- sigmoid(z)) .* tanh.(ĥ) .+ sigmoid(z) .* h

    return h_new
end

mutable struct GRU{T}
    cell::GRUCell{T}
    state::Matrix{T}
    prob_layer::Dense
end

Flux.@functor GRU

function GRU(input_size::Int, hidden_size::Int, T=Float32)
    return GRU(
        GRUCell{T}(
            randn(T, hidden_size * 3, input_size),  # w
            randn(T, hidden_size * 3, hidden_size), # u
            zeros(T, hidden_size * 3)               # b
        ),
        zeros(T, hidden_size, 1),
        Dense(hidden_size => 1)
    )
end

# Forward pass
function (m::GRU)(x::AbstractMatrix{T}, bags::Union{AlignedBags{Int64}, Nothing}=nothing) where T <: Real
    println("Size x in GRU: ", size(x))
    println("Size m.state in GRU before if: ", size(m.state))
    # Expand state if needed (entering bag node processing)
    if bags !== nothing
        # Expand state according to bags
        m.state = expand_hidden_state(m.state, bags)
    end
    println("Size m.state in GRU after if: ", size(m.state))
    # Pass both state and input to cell
    m.state = m.cell(m.state, x)
    return m.state
end

# Extended get_probs to handle data type
function get_log_probs(m::GRU, h::Matrix{T}, data) where T
    n_dims = size(data, 1)
    
    # Create layer based on data type and size
    if data isa OneHotMatrix  # Categorical
        m.prob_layer = Dense(size(h,1) => n_dims)
    else  # Gaussian (Float32 Matrix)
        m.prob_layer = Dense(size(h,1) => 2)  # mean, std
    end
    
    logits = m.prob_layer(h)
    
    # Transform to probabilities based on type
    if data isa OneHotMatrix
        return logsoftmax(logits)  # n_dims × batch_size
    else
        μ = logits[1, :]
        σ = exp.(logits[2, :])
        
        # For each observation:
        log_probs = map(1:size(data,2)) do i
            logpdf(Normal(μ[i], σ[i]), data[1,i])
        end
        
        return reshape(log_probs, 1, :)
    end
end

# Add function to expand hidden state according to bags
function expand_hidden_state(h::Matrix{T}, bags::Union{AlignedBags{Int64}, Nothing}) where T
    # h: 8×4893 (hidden state for each bag)
    # bags: vector of ranges like [1:3, 4:6, ...] mapping 4893 -> 10486
    
    n_features, _ = size(h)
    total_obs = sum(length.(bags))  # 10486
    
    expanded_h = zeros(T, n_features, total_obs)
    
    # Copy each column according to bags
    for (bag_idx, bag_range) in enumerate(bags)
        expanded_h[:, bag_range] .= h[:, bag_idx:bag_idx]
    end
    
    return expanded_h
end