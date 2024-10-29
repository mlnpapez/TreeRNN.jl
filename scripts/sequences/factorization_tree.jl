using Mill
using Flux
using Random
using Zygote
using LinearAlgebra
using Revise

includet("rnn_model.jl")
includet("lstm_model.jl")
includet("gru_model.jl")
includet("stacked_model.jl")


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
const SequentialModel = Union{RNN, LSTM, GRU, StackedModel}

# Main tree structure for handling factorization
mutable struct FactorizationTree{T<:SequentialModel}
    model::T  # Sequential model (RNN, LSTM, GRU, or StackedModel)
    root::AbstractFactorizationNode
    Tp::Vector{Mill.ArrayNode} # Array nodes visited in DFS
end

# Main build function
function build_factorization_tree(mill_node::Union{Mill.ArrayNode, Mill.BagNode, Mill.ProductNode}, model)
    # Create factorization tree with empty Tp
    root = build_node(mill_node)
    return FactorizationTree{typeof(model)}(model, root, Mill.ArrayNode[])
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

# Basic probability computation 
function compute_probability(tree::FactorizationTree, node::AbstractFactorizationNode; direction::Symbol=:left_to_right)
    # Initialize empty conditions
    conditions = ConditionSet()

    # Compute probabilities through DFS
    prob, data = _compute_probability(tree, node, conditions, direction)
    
    # Update tree's Tp with all array nodes found during computation
    append!(tree.Tp, conditions.Tp)

    # Print summary
    println("\n=== Computation Summary ===")
    println("Tree structure: ", typeof(tree.root))
    println("Total array nodes found: ", length(tree.Tp))
    println("Traversal direction: ", direction)

    return prob, data
end

# Array node probability computation
function _compute_probability(tree::FactorizationTree, node::ArrayFactorization, conditions::ConditionSet, direction::Symbol)
    # Add to DFS path
    push!(conditions.path, node)
    
    # Use node data and RNN state (which maintains Tp conditioning)
    # flattening data as vector
    # input_data = reshape(node.node.data, :, 1)

    # Model state contains conditioning on Tp (previously visited array nodes)
    prob = tree.model(node.node.data)
    
    # Add this array node to Tp
    push!(conditions.Tp, node.node)
    
    return prob, node.node.data
end

# Bag node probability computation
function _compute_probability(tree::FactorizationTree, node::BagFactorization, conditions::ConditionSet, direction::Symbol)
    push!(conditions.path, node)
    println("\n=== Processing Bag Node ===")
    println("DFS state - Current Tp size: ", length(conditions.Tp))
    
    # Get bag node structure
    data_node = node.node.data
    bags = node.node.bags
    println("Bag structure: ", length(bags), " bags")
    
    # Save state with current Tp for conditional independence
    initial_Tp_state = copy(tree.model.state)
    
    # Process each bag independently
    all_probs = Float32[]
    for (bag_idx, bag) in enumerate(bags)
        println("\nProcessing bag ", bag_idx, "/", length(bags))
        bag_probs = Float32[]
        
        # Reset state for conditional independence
        tree.model.state = copy(initial_Tp_state)

        # Fresh conditions for independence
        instance_condition = ConditionSet(
            Mill.ArrayNode[],  # Empty Tp for this instance
            [],               # No T<w needed in bag nodes
            copy(conditions.path)
        )
        
        println("Processing bag elements: ", length(bag), " elements")
        for i in bag
            # Process each child with same conditions (independence)
            prob, child_data = _compute_probability(tree, node.children[1], instance_condition, direction)
            push!(bag_probs, prob[1])
        end

        # Update main Tp with array nodes from this bag
        println("Updating DFS Tp: ", length(conditions.Tp), " -> ", length(conditions.Tp) + length(instance_condition.Tp))
        append!(conditions.Tp, instance_condition.Tp)

        # Multiply independent probabilities within bag
        push!(all_probs, prod(bag_probs))
    end
    
    total_prob = reshape([prod(all_probs)], :, 1)
    return total_prob, data_node.data
end

# Product node probability computation
function _compute_probability(tree::FactorizationTree, node::ProductFactorization, conditions::ConditionSet, direction::Symbol)
    push!(conditions.path, node)
    println("\n=== Processing Product Node ===")
    println("DFS Path depth: ", length(conditions.path))
    println("Current Tp size: ", length(conditions.Tp))
    
    # Get children in appropriate traversal order
    children = direction == :right_to_left ? reverse(collect(node.children)) : collect(node.children)
    
    # Setup for chain rule factorization
    current_siblings = []  # T<w for chain rule
    child_probs = Float32[]
    all_data = []
    accumulated_array_nodes = Mill.ArrayNode[]  # Collect for Tp
    
    # Save state with current Tp
    initial_Tp_state = copy(tree.model.state)
    
    # Process each child
    for (child_idx, child) in enumerate(children)
        println("\n--- Processing Child ", child_idx, "/", length(children), " ---")
        println("Child type: ", typeof(child))
        is_direct = node == conditions.path[end]
        println("Direct child of current product node: ", is_direct)

        # Reset to Tp state for each child
        tree.model.state = copy(initial_Tp_state)
        
        # Setup child conditions
        child_condition = ConditionSet(
            copy(conditions.Tp),    # Pass current Tp
            current_siblings,       # Pass T<w for chain rule
            copy(conditions.path)
        )
        
        # Process child
        prob, data = _compute_probability(tree, child, child_condition, direction)
        push!(child_probs, prob[1])
        
        # Handle array nodes based on child type and position
        if is_direct
            if isa(child, ArrayFactorization)
                println("Direct array child: adding to T<w and accumulating")
                push!(current_siblings, child.node)          # For chain rule
                push!(accumulated_array_nodes, child.node)   # For future Tp
            else
                println("Direct non-array child: accumulating its descendants")
                append!(accumulated_array_nodes, child_condition.Tp)
            end
        else
            # Non-direct children: all array nodes go to accumulation
            if isa(child, ArrayFactorization)
                push!(accumulated_array_nodes, child.node)
            end
            append!(accumulated_array_nodes, child_condition.Tp)
        end
        
        push!(all_data, data)
    end
    
    # Update main Tp with all accumulated array nodes
    println("\nFinalizing product node computation")
    println("Updating DFS Tp: ", length(conditions.Tp), " -> ", 
            length(conditions.Tp) + length(accumulated_array_nodes))
    append!(conditions.Tp, accumulated_array_nodes)
    
    total_prob = reshape([prod(child_probs)], :, 1)
    return total_prob, all_data
end
