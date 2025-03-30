using DrWatson
@quickactivate
using Mill
using Flux
using JSON3
using Revise
using Flux
using LinearAlgebra
using Random
using Distributions

using TreeRNN

includet("../src/models/models.jl")
includet("../src/models/tree.jl")
includet("utils.jl")
includet("../src/models/tree_gru.jl")
includet("../src/models/prob_gru.jl")


# Add path to data
dirdata = "data"

"""
Generate new Mill SAMPLE guided by existing Mill structure.
Simply creates new content matching original dimensions.
"""
function generate_from_structure(x::AbstractMillNode, m::Union{TreeRecur, Tree}, seq_model=nothing)
    
    # Extract seq_model from TreeRecur if first call and reset its state
    if m isa TreeRecur
        seq_model = m.seq_model
        # println("Size of hidden state: ", size(seq_model.state))

        # Reset state directly
        hidden_size = size(seq_model.state, 1)
        batch_size = numobs(x)
        seq_model.state = zeros(Float32, hidden_size, batch_size)
        # println("Reset state shape: ", size(seq_model.state))
        # println("State sum: ", sum(seq_model.state))          # Should be 0.0
        m = m.tree
    end
    
    # seq_model is available?
    @assert !isnothing(seq_model) "Sequential model must be provided"

    if x isa ArrayNode
        # Get original dimensions
        feature_dim, batch_size = size(x.data)
        # println(typeof(x.data))
        # display(x.data)

        # Direct access to current tree's prob_layer
        logits = m.prob_layer(seq_model.state)
        
        # println("Size of logits: ", size(logits))

        # Create random content matching original dimensions
        if x.data isa OneHotArray

            # For categorical data, random one-hot vectors
            probs = softmax(logits)
            # println("Size of probs: ", size(probs))

            # Sample from categorical distribution for each column
            sampled = [rand(Categorical(probs[:,i])) for i in 1:batch_size]
            # println("Size of sampled: ", size(sampled))
            # display(sampled)

            new_data = Flux.onehotbatch(sampled, 1:feature_dim)
            # display(new_data)

            # Transform to embeddings using Dense layer
            embeddings = m.children(new_data)  # Dense layer transforms predicted data (now imputs) to nh dimensions

            # Update seq_model state with embeddings
            seq_model(embeddings)

            return ArrayNode(new_data) # Converted to one-hot
        else
            # For continuous data, random Float32 matrix
            μ = logits[1,:]
            σ = exp.(logits[2,:])

            # Sample from Gaussian
            sampled = μ .+ σ .* randn(Float32, size(x.data,2))
            # println("Size of sampled: ", size(sampled))
            # display(sampled)

            new_data = reshape(sampled, size(x.data)...)
            # display(new_data)

            # Transform to embeddings using Dense layer
            embeddings = m.children(new_data)  # Dense layer transforms predicted data (now imputs) to nh dimensions

            # Update seq_model state with embeddings
            seq_model(embeddings)

            return ArrayNode(new_data)
        end
        
    elseif x isa ProductNode
        # Recursively generate new children
        new_children = map((x, m) -> generate_from_structure(x, m, seq_model), x.data, m.children) 

        # Create new ProductNode with same keys
        return ProductNode(NamedTuple{keys(x.data)}(new_children))

    elseif x isa BagNode
        # println("Original state shape: ", size(seq_model.state))  # e.g., 80×4893

        # Expand state according to bags
        expanded_state = expand_hidden_state(seq_model.state, x.bags)
        # println("Expanded state shape: ", size(expanded_state))   # e.g., 80×10486
        seq_model.state = expanded_state

        # Generate new content for bag's data
        new_data = generate_from_structure(x.data, m.children, seq_model)

        # Reduce state back
        reduced_state = reduce_hidden_state(seq_model.state, x.bags)
        # println("Reduced state shape: ", size(reduced_state))     # Back to 80×4893
        # Verify reduction works correctly
        # println("Sum before/after: ", sum(seq_model.state), " / ", sum(reduced_state))
        seq_model.state = reduced_state

        # Keep original bag structure
        return BagNode(new_data, x.bags)
    end
end


# Experimenting
function test_sampling()
    # Load your data
    dataset = datasets[1].name
    data = JSON3.read(read("$(dirdata)/$(dataset).json", String))
    x, y = data.x, data.y

    s = schema(x)
    e = suggestextractor(s)
    x = reduce(catobs, e.(x))

    subtree_charge = x[:atoms].data[:charge]
    printtree(subtree_charge)
    # println(typeof(subtree_charge.data))

    # println("Number of obs in node: ", numobs(subtree_charge))

    random_matrix = rand(Float32, 1, 4893)
    # println("Random matrix: ", typeof(random_matrix))
    


    bags = x[:atoms].data[:bonds].bags
    # println("Number of bags/unit ranges: ", size(bags))

    # Build supervised model as before
    ni, nh, hidden_size = 80, 5, 10

    m_charge = TreeGRU(Float32, nh, ni, subtree_charge, hidden_size)

    # println(typeof(m_charge.tree.prob_layer))

    emb_charge, log_prob_embs = m_charge(subtree_charge)
    # println("Embedding size: ", size(emb_charge))


    # println("\n----------Start bonds example-----------")

    subtree_bonds = x[1]
    
    ## println(fieldnames(ProductNode))

    # println(typeof(subtree_bonds))
    printtree(subtree_bonds)

    m_bonds = TreeGRU(Float32, nh, ni, subtree_bonds, hidden_size)

    emb_bonds = m_bonds(subtree_bonds)

    ## println(typeof(m_bonds.tree.children[:element].prob_layer))

    # Generate new structure
    new_structure = generate_from_structure(subtree_bonds, m_bonds)

    # Print both structures to compare
    # println("Original structure:")
    printtree(subtree_bonds)
    ## println(subtree_bonds.bags[1:10])
    ## display(subtree_bonds[:element])
    
    # println("\nGenerated structure:")
    printtree(new_structure)
    ## println(new_structure.bags[1:10])
    ## display(new_structure[:element])
    
    #=p = Flux.params(m_bonds)
    # println(length(p))
    for i in 1:length(p)
        # println("\nSize of model params", i ,": ", length(p[i]))
    end    =#
end