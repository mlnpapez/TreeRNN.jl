export GRU, GRUCell

struct GRUCell{T}
    w::Matrix{T}  # Input weights
    u::Matrix{T}  # Hidden weights
    b::Vector{T}  # Bias
end


function GRUCell(input_size::Int, hidden_size::Int; init=Flux.glorot_uniform)
    return GRUCell(
        init(3 * hidden_size, input_size),
        init(3 * hidden_size, hidden_size),
        init(3 * hidden_size)
    )
end

Flux.@functor GRUCell

# GRU forward pass
function (m::GRUCell{T})(h::AbstractMatrix{T}, x::AbstractMatrix{T}) where {T<:Real}
    # Split weights, include input weights
    wr, wh, wz = _expand(m.w, Val(3))  # input weights
    ur, uh, uz = _expand(m.u, Val(3))  # hidden weights
    br, bh, bz = _expand(m.b, Val(3))  # biases

    # Reset gate
    r = sigmoid.(wr*x .+ ur*h .+ br)

    # Create reset state
    h_reset = r .* h
    
    #println("Size h_reset: ", size(h_reset))
    
    # Candidate state
    ĥ = tanh.(wh*x .+ uh*h_reset .+ bh)

    #println("Size ĥ: ", size(ĥ))
    
    # Update gate
    z = sigmoid.(wz*x .+ uz*h .+ bz)
    
    #println("Size z: ", size(z))

    # Final update
    h_new = z .* h + (1 .- z) .* ĥ

    #println("Size h_new: ", size(h_new))
    
    return h_new
end

mutable struct GRU{T}
    cell::GRUCell{T}
    state::Matrix{T}
    prob_layer::Dense
end

Flux.@functor GRU

function GRU(input_size::Int, hidden_size::Int, batch_size::Int, T=Float32)
    return GRU(
        GRUCell(input_size, hidden_size),
        zeros(T, hidden_size, batch_size), # initialize with right batch size
        Dense(hidden_size => 1)
    )
end
    
# Forward pass
function (m::GRU)(x::AbstractMatrix{T}) where T <: Real
    # Pass both state and input to cell
    m.state = m.cell(m.state, x)
    return m.state
end

# Extended get_probs to handle data type
function get_log_probs(m::GRU, data)
    n_dims = size(data, 1)

    # Create layer based on data type and size
    if data isa OneHotMatrix  # Categorical
        m.prob_layer = Dense(size(m.state, 1) => n_dims)
    else  # Gaussian (Float32 Matrix)
        m.prob_layer = Dense(size(m.state, 1) => 2)  # mean, std
    end
    
    println(m.prob_layer)

    logits = m.prob_layer(m.state)
    
    # Transform to probabilities based on type
    if data isa OneHotMatrix
        log_probs = logsoftmax(logits)  # n_dims (categories) × batch_size

        # Multiply with one-hot to select actual categories
        scalar_log_probs = sum(log_probs .* data, dims=1)  # 1×batch_size
        return scalar_log_probs
    else
        μ = logits[1, :]
        σ = exp.(logits[2, :])  # transform from logσ to σ
        x = data[1, :]          # actual values
        log_probs = -0.5 * (log(2π) .+ 2*log.(σ) .+ ((x .- μ)./σ).^2)
        
        return reshape(log_probs, 1, :) # 1×4893 matrix
        # Each column is log probability of actual value
    end
end

# Add function to expand hidden state according to bags
function expand_hidden_state(h::AbstractMatrix{T}, bags::Union{AlignedBags{Int64}, Nothing}) where T <: Real
    # h: 8×4893 (hidden state for each bag)
    # bags: vector of ranges like [1:3, 4:6, ...] mapping 4893 -> 10486
    
    n_features, _ = size(h)
    total_obs = sum(length.(bags))  # 10486
    
    expanded_h = zeros(T, n_features, total_obs)
    
    # Copy each column according to bags
    for (bag_idx, bag_range) in enumerate(bags)
        # Expand model state according to bags
        expanded_h[:, bag_range] .= h[:, bag_idx:bag_idx]
    end

    return expanded_h
end
# Add function to reduce hidden state according to bags
function reduce_hidden_state(h::AbstractMatrix{T}, bags::Union{AlignedBags{Int64}, Nothing}) where T <: Real
    n_features = size(h, 1)
    n_bags = length(bags)

    reduced_h = zeros(T, n_features, n_bags)
    
    for (bag_idx, bag_range) in enumerate(bags)
        reduced_h[:, bag_idx] = sum(view(h, :, bag_range), dims=2)
    end
    return reduced_h
end
# Add function to sum log probs of bag children according to bags
function aggregate_log_probs(log_probs::AbstractMatrix{T}, bags) where T <: Real
    n_bags = length(bags)
    aggregated = zeros(T, 1, n_bags)  # 1×n_bags for log probs
    
    for (i, bag) in enumerate(bags)
        aggregated[1, i] = sum(log_probs[1, bag])
    end
    return aggregated
end