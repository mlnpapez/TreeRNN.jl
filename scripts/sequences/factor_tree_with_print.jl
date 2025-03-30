using Mill
using Flux
using Random
using Zygote
using LinearAlgebra
using Revise

includet("../models/rnn_model.jl")
includet("../models/lstm_model.jl")
includet("../models/gru_model.jl")
includet("../models/stacked_model.jl")
includet("../models/input_adapter.jl")


# Types for factorization
abstract type AbstractFactorizationNode end

# Factorization nodes wrapping Mill nodes
struct ArrayFactorization <: AbstractFactorizationNode
    node::Mill.ArrayNode
end

struct BagFactorization <: AbstractFactorizationNode
    node::Mill.BagNode
    children::Vector{AbstractFactorizationNode}
end

struct ProductFactorization <: AbstractFactorizationNode
    node::Mill.ProductNode
    children::Vector{AbstractFactorizationNode}
end

# Define possible model types
const SequentialModel = Union{BaseSequentialModel, InputAdapter}

# Main tree structure for handling factorization
mutable struct FactorizationTree{T<:SequentialModel}
    model::T  # Sequential model (RNN, LSTM, GRU, or StackedModel)
    root::AbstractFactorizationNode
    Tp::Vector{Mill.ArrayNode} # Array nodes visited in DFS
end

# Main build function
function build_factorization_tree(mill_node::Union{Mill.ArrayNode, Mill.BagNode, Mill.ProductNode}, base_model::SequentialModel)
    # Create factorization tree with empty Tp
    root = build_node(mill_node)
    return FactorizationTree{typeof(base_model)}(base_model, root, Mill.ArrayNode[])
end

# Recursive build function for different node types
function build_node(node::Mill.ArrayNode)
    return ArrayFactorization(node)
end

function build_node(node::Mill.BagNode)
    # Recursively build children
    children = AbstractFactorizationNode[build_node(node.data)]
    return BagFactorization(node, children)
end

function build_node(node::Mill.ProductNode)
    # Recursively build all children
    children = AbstractFactorizationNode[build_node(child) for child in node.data]
    return ProductFactorization(node, children)
end

# Condition tracking structure
struct ConditionSet
    Tp::Vector{Mill.ArrayNode}  # Array nodes from DFS
    T_siblings::Vector{Mill.AbstractMillNode} # Siblings for chain rule (T<w)
    path::Vector{AbstractFactorizationNode}
end

function ConditionSet()
    return ConditionSet(
        Mill.ArrayNode[],
        Mill.AbstractMillNode[], 
        AbstractFactorizationNode[]
    )
end

# Add counter for each level
mutable struct ProcessingCounter
    level_counts::Dict{Int, Int}  # depth -> count of nodes processed at that depth
end

function increment_counter!(counter::ProcessingCounter, depth::Int)
    counter.level_counts[depth] = get(counter.level_counts, depth, 0) + 1
    return counter.level_counts[depth]
end

# Helper function for log transformation of probs using Flux's implementations
function compute_log_probs(logits::AbstractMatrix)
    if size(logits, 1) == 1
        # Binary case: P(x=1) for binary decision
        # Scalar case - use logsigmoid
        return Flux.logsigmoid.(logits)
    else
        # Categorical case: P(x=k) for each category k
        # Vector case - use logsoftmax - for categorical data
        return Flux.logsoftmax(logits; dims=1)
    end
    # For real-valued data:
    # Could use Gaussian likelihood
    # return log_gaussian_prob(logits)
end

# Basic log probability computation 
function compute_probability(tree::FactorizationTree, node::AbstractFactorizationNode; direction::Symbol=:left_to_right)
    # Initialize empty conditions
    conditions = ConditionSet()

    counter = ProcessingCounter(Dict{Int,Int}())

    # Compute log probabilities through DFS
    log_prob, _, data = _compute_probability(tree, node, conditions, direction, nothing, counter)
    prob = exp(log_prob)
    # Update tree's Tp with all array nodes found during computation
    append!(tree.Tp, conditions.Tp)

    # Print summary
    # println("\n=== Computation Summary ===")
    # println("Tree structure: ", typeof(tree.root))
    println("Total array nodes found: ", length(tree.Tp))
    # println("Traversal direction: ", direction)
    # println("\nFinal probability for root level product node is: ", prob)
    # @info "Prob calculation for tree is done."
    # println("------------------------------------------\n")

    return log_prob, data
end

# Array node log probability computation
function _compute_probability(tree::FactorizationTree, node::ArrayFactorization, conditions::ConditionSet, direction::Symbol, slice_indices, counter::ProcessingCounter)
    # Add to DFS path
    push!(conditions.path, node)

    depth = length(conditions.path)
    order = increment_counter!(counter, depth)
    
    println("\n=== Processing Array Node $(depth).$(order) ===")
    # println("DFS Path depth: ", length(conditions.path))
    # println("Current Tp size: ", length(conditions.Tp))
    
    # Use node data and RNN state (which maintains Tp conditioning)
    # Flattening data as vector
    # input_data = reshape(node.node.data, :, 1)

    # Get correct slice if needed
    data = isnothing(slice_indices) ? node.node.data : node.node.data[:, slice_indices]

    # Model state contains conditioning on Tp (previously visited array nodes)
    println("1st phase PREDICTION: Get predicted probability distribution without seeing current data in node")
    prediction_signal =  ones(Float32, size(data)) * -1.0f0  # Simple, distinct signal, e.g. prediction vector or some other signal for masking
    predicted_array_logits = tree.model(prediction_signal)  # Model uses previous state only

    # Ensure prediction matches data input dimension
    if size(predicted_array_logits, 1) != size(data, 1)
        # Add output adapter if needed (fixed transformation)
        output_adapter = Chain(
            Dense(size(predicted_array_logits, 1), size(data, 1)), # Just linear projection
            identity  # or appropriate activation # No optimization needed
        )
        predicted_array_logits = output_adapter(predicted_array_logits)
    end

    # Get logits and convert to log probabilities
    log_array_prob = compute_log_probs(predicted_array_logits)
    # println("\n1st phase - Predicted distribution for current node:")
    display(exp.(log_array_prob))  # Show actual probabilities


    # 2nd phase PROCESS: Update state with actual data for future predictions
    println("\n2nd phase - Processing actual array node data for future conditioning")

    # Show current array node data
    # println("\nArray node data: ")
    # display(data)

    # Let model process this data and udpate its hidden state
    tree.model(data)
    
    # Add this array node to Tp
    push!(conditions.Tp, node.node)

    # Final log probability for array node
    println("\n== Finalizing array node computation ==")
    # println("Output log prob for array node: ", log_array_prob)
    # println("---------------\n")
    
    return log_array_prob[1], log_array_prob, node.node.data
end

# Bag node log probability computation
function _compute_probability(tree::FactorizationTree, node::BagFactorization, conditions::ConditionSet, direction::Symbol, slice_indices, counter::ProcessingCounter)
    push!(conditions.path, node)

    depth = length(conditions.path)
    order = increment_counter!(counter, depth)
    println("\n\n=== Processing Bag Node $(depth).$(order) (Independent Elements) ===")
    # println("DFS Path depth: ", length(conditions.path))
    # println("Current Tp size: ", length(conditions.Tp))
    
    data_node = node.node.data
    bags = node.node.bags # AlignedBags indices
    
    # State management for independence
    initial_Tp_state = copy(tree.model.state)
    
    # Track probabilities maintaining dimensionality
    scalar_bag_log_probs = Float32[]       # For quick probability checks
    all_bag_distributions = []            # Maintain complete probability structures
    
    # Track all descendant array nodes with their corresponding indices
    accumulated_arrays = Dict{Mill.ArrayNode, Vector{Int}}() # Store array_node => [indices]

    if !isnothing(slice_indices)
        println("This bag is a descendant of child: ", slice_indices, " of the closest ancestor bag.")
        println("And this bag has (unit range) ", bags[slice_indices], " children")
    end

    for bag_idx in (!isnothing(slice_indices) ? bags[slice_indices] : bags[1])
        println("\nBag id is: ", bag_idx, " and is type", typeof(bag_idx))
        println("\nProcessing independent children ", bag_idx, "/", length(bags))
        
        # Reset state for conditional independence
        tree.model.state = copy(initial_Tp_state)
        instance_condition = ConditionSet(
            Mill.ArrayNode[],  # Fresh Tp for independence
            [],               # No siblings needed in bags
            copy(conditions.path)
        )

        # Get the correct slice for this bag
        bag_data = node.node.data[bag_idx]  # Mill handles slicing correctly
        printtree(bag_data)

        # Get probabilities maintaining dimensions
        log_prob, prob_distribution, child_data = _compute_probability(
            tree, node.children[1], instance_condition, direction, bag_idx, counter)


        # Store array nodes with their corresponding indices, prevent duplicates
        # println("\nSet Tp containing all descendat array nodes of processed children is: ", instance_condition.Tp)

        for array_node in instance_condition.Tp
            if haskey(accumulated_arrays, array_node)
                push!(accumulated_arrays[array_node], bag_idx)
            else
                accumulated_arrays[array_node] = [bag_idx]
            end
        end

        # Track both scalar and vector of children distributions
        push!(scalar_bag_log_probs, log_prob[1])
        # println("scalar_bag_log_probs", scalar_bag_log_probs)
        push!(all_bag_distributions, prob_distribution)  # Keep full dimensionality and nested structure
        
        # println("Element log probability: ", log_prob)
        # println("Conditonal probability distributions aggregated from child node: ", bag_idx)
        # display(prob_distribution)

        # Update array node tracking (set Tp)
        println("\nBag Node processing: Updating processed DFS array nodes (set Tp) tracking: ", length(conditions.Tp), " -> ", length(conditions.Tp) + length(instance_condition.Tp))
        append!(conditions.Tp, instance_condition.Tp)
    end
    
    # Final probability aggregation across groups
    println("\n\n== Finalizing Bag Node Computation ==")

    # After collecting all indices, fix any duplicate indices, create sequence based on length
    for (array_node, indices) in accumulated_arrays
        # Only transform if there are duplicates
        if length(unique(indices)) != length(indices)
            # Use length of original indices array (counting duplicates)
            n = length(indices)  # e.g., length([1,2,2,3]) = 4
            # Create new sequence 1:n as new values for a key
            accumulated_arrays[array_node] = collect(1:n)
        end
        # Has duplicates (e.g., [1,2,2,3]), so transform
        # If no duplicates (e.g., [2,3,4]), keep original indices

        #println("\n Check each array node indeces")
        #println(accumulated_arrays)
    end

    # Update state with all collected properly sliced array data before returning
    # println("\nUpdating model state with sliced array nodes from bag children...")
    tree.model.state = copy(initial_Tp_state)  # Start fresh

    # Get unique indices in order
    ordered_indices = sort(unique(vcat([indices for indices in values(accumulated_arrays)]...)))

    # Process each array node and its indices (given by parent) in order
    println(accumulated_arrays)

    for idx in ordered_indices
        # println("\nProcessing arrays for index: ", idx) # Process each unique index once
        
        # Process each array node's data for this index
        for array_node in keys(accumulated_arrays)
            if idx in accumulated_arrays[array_node]
                sliced_data = array_node.data[:, idx]
                    #println("\nProcessing slice of array node data with dimensions: ", size(sliced_data))
                    #display(sliced_data)
                tree.model(sliced_data) # Process correct slice of array node data
            end
        end
    end

    # Maintain dimensionality in results
    total_scalar_log_prob = reshape([sum(scalar_bag_log_probs)], :, 1)
    
    # println("\nTotal log probability across groups: ", total_scalar_log_prob)
    # println("\nProbability Structure Information:")
    # println("Number of independent groups: ", length(all_bag_distributions))

    #= for (i, bag) in enumerate(all_bag_distributions)
        # println("Group $i contains $(length(bag)) probability distributions")
        for (j, dist) in enumerate(bag)
            # println("  Element $j dimensions: ", size(dist))
        end
    end =#

    println("---------------\n")
    
    # Return both scalar and full dimensional results
    return total_scalar_log_prob, all_bag_distributions, data_node.data
end

# Product node log probability computation
function _compute_probability(tree::FactorizationTree, node::ProductFactorization, conditions::ConditionSet, direction::Symbol, slice_indices, counter::ProcessingCounter)
    push!(conditions.path, node)

    depth = length(conditions.path)
    order = increment_counter!(counter, depth)

    println("\n\n=== Processing Product Node $(depth).$(order) (Chain Rule Factorization) ===")
    # println("DFS Path depth in tree: ", length(conditions.path))
    # println("Current Tp (processed array nodes) size: ", length(conditions.Tp))
    
    # Order children for factorization direction
    factorization_order = direction == :right_to_left ? reverse(collect(node.children)) : collect(node.children)
    
    # Tracking structures for chain rule factorization
    processed_siblings = []  # T<w: previous siblings for chain rule
    full_probability_distributions = []  # Store complete probability vectors/tensors
    scalar_log_probs = Float32[]  # Store first elements for quick prob. computation
    processed_data = []
    array_node_tracker = Mill.ArrayNode[]  # Track array nodes for Tp set
    
    # Initialize state with Tp conditioning
    conditioned_state = copy(tree.model.state)
    
    # Process each child following chain rule
    for (child_idx, current_child) in enumerate(factorization_order)
        # println("\n--- Processing Factor ", child_idx, "/", length(factorization_order), " ---")
        # println("Factor type: ", typeof(current_child))
        
        # Use state conditioned on Tp and previous siblings
        tree.model.state = copy(conditioned_state)
        
        # Setup conditioning for current factor
        factor_conditions = ConditionSet(
            Mill.ArrayNode[],      # Tp conditioning
            processed_siblings,        # T<w conditioning
            copy(conditions.path)
        )
        
        # Compute factor's conditional probability
        conditional_log_prob, prob_distribution, child_data = _compute_probability(
            tree, current_child, factor_conditions, direction, slice_indices, counter
        )
        
        # Track probabilities maintaining dimensions
        push!(scalar_log_probs, conditional_log_prob[1])  # For scalar computation
        push!(full_probability_distributions, prob_distribution)  # Maintain full dimensions
        
        # Debug probability information
        # println("Factor conditional log probability: ", conditional_log_prob)
        # println("Current chain rule log probabilities: ", scalar_log_probs, "\n")
        
        # println("Full conditional probability distribution:")
        # display(exp.(conditional_log_prob))  # Show actual probabilities
        # println("Accumulated probability distributions:")
        # display(full_probability_distributions)
        
        # Update tracking based on factor type
        if isa(current_child, ArrayFactorization)
            # @info "Array factor: updating chain rule conditions (T<w)"
            push!(processed_siblings, current_child.node)
            push!(array_node_tracker, current_child.node)
        else
            # @info "Structural factor: accumulating descendant array nodes"
            append!(array_node_tracker, factor_conditions.Tp)
        end
        
        # Update state for next factor in chain
        conditioned_state = copy(tree.model.state)
        push!(processed_data, child_data)
    end
    
    # Update Tp tracking
    println("\n\n== Finalizing Product Chain Rule Computation ==")
     println("\nProduct Node processing: Updating processed DFS array nodes (set Tp) tracking: ", length(conditions.Tp), " -> ", length(conditions.Tp) + length(array_node_tracker))
    append!(conditions.Tp, array_node_tracker)
    
    # Report joint probability information
    # println("\n=== Joint Distribution Information ===")
    # println("Number of factors: ", length(full_probability_distributions))
    #=for (i, dist) in enumerate(full_probability_distributions)
         println("Factor $i dimensions: ", size(dist))
    end =#
    
    # Maintain proper dimensionality in result
    joint_log_prob = reshape([sum(scalar_log_probs)], :, 1)
    # println("Joint log probability for the first entry of joint log prob tensor of random variables: ", joint_log_prob)
    # println("---------------\n")
    
    return joint_log_prob, full_probability_distributions, processed_data
end
