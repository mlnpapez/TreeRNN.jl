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

abstract type AbstractFactorizationNode end

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

const SequentialModel = Union{BaseSequentialModel, InputAdapter}

mutable struct FactorizationTree{T<:SequentialModel}
    model::T  
    root::AbstractFactorizationNode
    Tp::Vector{Mill.ArrayNode} 
end

function build_factorization_tree(mill_node::Union{Mill.ArrayNode, Mill.BagNode, Mill.ProductNode}, base_model::SequentialModel)
    root = build_node(mill_node)
    return FactorizationTree{typeof(base_model)}(base_model, root, Mill.ArrayNode[])
end

function build_node(node::Mill.ArrayNode)
    return ArrayFactorization(node)
end

function build_node(node::Mill.BagNode)
    children = AbstractFactorizationNode[build_node(node.data)]
    return BagFactorization(node, children)
end

function build_node(node::Mill.ProductNode)
    children = AbstractFactorizationNode[build_node(child) for child in node.data]
    return ProductFactorization(node, children)
end

struct ConditionSet
    Tp::Vector{Mill.ArrayNode}  
    T_siblings::Vector{Mill.AbstractMillNode} 
    path::Vector{AbstractFactorizationNode}
end

function ConditionSet()
    return ConditionSet(
        Mill.ArrayNode[],
        Mill.AbstractMillNode[], 
        AbstractFactorizationNode[]
    )
end

mutable struct ProcessingCounter
    level_counts::Dict{Int, Int}  
end

function increment_counter!(counter::ProcessingCounter, depth::Int)
    counter.level_counts[depth] = get(counter.level_counts, depth, 0) + 1
    return counter.level_counts[depth]
end 

function compute_log_probs(logits::AbstractMatrix)
    if size(logits, 1) == 1
        return Flux.logsigmoid.(logits)
    else
        return Flux.logsoftmax(logits; dims=1)
    end
end

function compute_probability(tree::FactorizationTree, node::AbstractFactorizationNode; direction::Symbol=:left_to_right)
    conditions = ConditionSet()

    counter = ProcessingCounter(Dict{Int,Int}())

    log_prob, _, data = _compute_probability(tree, node, conditions, direction, nothing, counter)
    prob = exp(log_prob[1])
    append!(tree.Tp, conditions.Tp)

    return log_prob, data
end

function _compute_probability(tree::FactorizationTree, node::ArrayFactorization, conditions::ConditionSet, direction::Symbol, slice_indices, counter::ProcessingCounter)
    push!(conditions.path, node)

    data = isnothing(slice_indices) ? node.node.data : node.node.data[:, slice_indices]

    prediction_signal =  ones(Float32, size(data)) * -1.0f0  
    predicted_array_logits = tree.model(prediction_signal)  

    if size(predicted_array_logits, 1) != size(data, 1)
        output_adapter = Chain(
            Dense(size(predicted_array_logits, 1), size(data, 1)), 
            identity  
        )
        predicted_array_logits = output_adapter(predicted_array_logits)
    end

    log_array_prob = compute_log_probs(predicted_array_logits)

    tree.model(data)

    push!(conditions.Tp, node.node)

    return log_array_prob[1], log_array_prob, node.node.data
end

function _compute_probability(tree::FactorizationTree, node::BagFactorization, conditions::ConditionSet, direction::Symbol, slice_indices, counter::ProcessingCounter)
    push!(conditions.path, node)

    data_node = node.node.data
    bags = node.node.bags 

    initial_Tp_state = copy(tree.model.state)

    scalar_bag_log_probs = Float32[]       
    all_bag_distributions = []            

    accumulated_arrays = Dict{Mill.ArrayNode, Vector{Int}}() 

    for bag_idx in (!isnothing(slice_indices) ? bags[slice_indices] : bags[1])

        tree.model.state = copy(initial_Tp_state)
        instance_condition = ConditionSet(
            Mill.ArrayNode[],  
            [],               
            copy(conditions.path)
        )

        bag_data = node.node.data[bag_idx]  
        printtree(bag_data)

        log_prob, prob_distribution, child_data = _compute_probability(
            tree, node.children[1], instance_condition, direction, bag_idx, counter)

        for array_node in instance_condition.Tp
            if haskey(accumulated_arrays, array_node)
                push!(accumulated_arrays[array_node], bag_idx)
            else
                accumulated_arrays[array_node] = [bag_idx]
            end
        end

        push!(scalar_bag_log_probs, log_prob[1])
        push!(all_bag_distributions, prob_distribution)  

        append!(conditions.Tp, instance_condition.Tp)
    end

    for (array_node, indices) in accumulated_arrays
        if length(unique(indices)) != length(indices)
            n = length(indices)  
            accumulated_arrays[array_node] = collect(1:n)
        end

    end

    tree.model.state = copy(initial_Tp_state)  
    ordered_indices = sort(unique(vcat([indices for indices in values(accumulated_arrays)]...)))

    for idx in ordered_indices
        for array_node in keys(accumulated_arrays)
            if idx in accumulated_arrays[array_node]
                sliced_data = array_node.data[:, idx]
                tree.model(sliced_data) 
            end
        end
    end

    total_scalar_log_prob = reshape([sum(scalar_bag_log_probs)], :, 1)

    return total_scalar_log_prob, all_bag_distributions, data_node.data
end

function _compute_probability(tree::FactorizationTree, node::ProductFactorization, conditions::ConditionSet, direction::Symbol, slice_indices, counter::ProcessingCounter)
    push!(conditions.path, node)

    factorization_order = direction == :right_to_left ? reverse(collect(node.children)) : collect(node.children)

    processed_siblings = []  
    full_probability_distributions = []  
    scalar_log_probs = Float32[]  
    processed_data = []
    array_node_tracker = Mill.ArrayNode[]  

    conditioned_state = copy(tree.model.state)

    for (child_idx, current_child) in enumerate(factorization_order)

        tree.model.state = copy(conditioned_state)

        factor_conditions = ConditionSet(
            Mill.ArrayNode[],      
            processed_siblings,        
            copy(conditions.path)
        )

        conditional_log_prob, prob_distribution, child_data = _compute_probability(
            tree, current_child, factor_conditions, direction, slice_indices, counter
        )

        push!(scalar_log_probs, conditional_log_prob[1])  
        push!(full_probability_distributions, prob_distribution)  



        if isa(current_child, ArrayFactorization)
            push!(processed_siblings, current_child.node)
            push!(array_node_tracker, current_child.node)
        else
            append!(array_node_tracker, factor_conditions.Tp)
        end

        conditioned_state = copy(tree.model.state)
        push!(processed_data, child_data)
    end

    append!(conditions.Tp, array_node_tracker)

    joint_log_prob = reshape([sum(scalar_log_probs)], :, 1)

    return joint_log_prob, full_probability_distributions, processed_data
end
