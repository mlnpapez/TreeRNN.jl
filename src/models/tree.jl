using Revise
using Profile

abstract type AbstractTree{C} end

_make_imputing(x, t) = t
_make_imputing(x::Array{Mill.Maybe{T}},  t::Dense) where T <: Number = preimputing_dense(t)
_make_imputing(x::Mill.MaybeHotArray, t::Dense) = postimputing_dense(t)
_make_imputing(x::NGramMatrix{Mill.Maybe{T}}, t::Dense) where T <: Sequence = postimputing_dense(t)


mutable struct Tree{C, T} <: AbstractTree{C}
    cell::Union{C, Chain{Tuple{C}}}
    children::T
end
Flux.@functor Tree
function Tree(m, x::AbstractProductNode, nh::Int=5)
    return Tree(m, map(x->Tree(m, x, nh), x.data))
end
function Tree(m, x::AbstractBagNode, nh::Int=5)
    return Tree(m, Tree(m, x.data, nh))
end
function Tree(m, x::ArrayNode, nh::Int=5)
    return Tree(m, _make_imputing(x.data, Dense(size(x.data, 1) => nh)))
end

function (m::Tree)(x::AbstractProductNode, seq_model)
    if Mill.numobs(values(x.data)) == 0
        return latent_empty(m)
    else
        results = map((m, x)->m(x, seq_model), m.children, x.data)
        # Split embeddings and log_probs maintaining structure
        @time embs = NamedTuple{keys(x.data)}(first.(values(results)))
        # Now embs is like:
        # (element = 5×batch, type_bond = 5×batch, ...)

        @time log_probs = last.(values(results))
        println(typeof(log_probs))

        # Sum log probabilities for joint prob
        @time joint_log_probs = reduce(.+, log_probs)  # sum along array nodes
        println("\nJoint log probs matrix for product node: ", size(joint_log_probs))
        display(joint_log_probs)

        emb = embs |> values |> state |> m.cell
        println("Product embedding: ", size(emb))
        return emb, joint_log_probs
    end
end
function (m::Tree)(x::AbstractBagNode, seq_model)
    println("\nBag node number of unit ranges/obs: ", length(x.bags))

    # 1. Expand state before processing child
    println("Expanded state: ")
    @time expanded_state = expand_hidden_state(seq_model.state, x.bags)
    seq_model.state = expanded_state
    #println("Expanded state: ", size(expanded_state))
    #display(expanded_state)

    # 2. Process child
    child_emb, child_log_probs = m.children(x.data, seq_model)
    
    # 3. Reduce state after child processing
    println("Reduced state: ")
    #@time reduced_state = mapreduce(b->sum(seq_model.state[:, b], dims=2), hcat, x.bags)

    @time reduced_state = reduce_hidden_state(seq_model.state, x.bags)
    seq_model.state = reduced_state
    #println("Reduced state: ")
    #display(reduced_state)
    
    # Aggregate log probs by bags
    #@time joint_log_probs = mapreduce(b->sum(child_log_probs[:, b], dims=2), hcat, x.bags)
    @time joint_log_probs = aggregate_log_probs(child_log_probs, x.bags)
    println("\nJoint log probs matrix for bag node: ", size(joint_log_probs))
    display(joint_log_probs)

    emb = (child_emb, x.bags) |> m.cell
    println("Bag embedding: ", size(emb))
    return emb, joint_log_probs
end
function (m::Tree)(x::ArrayNode, seq_model)
    @time log_probs = get_log_probs(seq_model, x.data)
    println("\nSize of log_probs matrix for array node: ", size(log_probs))
    display(log_probs)

    emb = m.children(x.data) |> m.cell

    println("Array embedding: ", size(emb), "\n")

    # Use passed seq_model here
    @time h = seq_model(emb)
    println("\nSize of hidden state of seq model after processing array embedding: ", size(h))

    return emb, log_probs
end

a2(x)    = reshape(hcat(x...), size(x[1])..., :)
a3(x, i) = reshape(hcat(getindex.(x, i)...), size(x[1][1])..., :)

state(x::NTuple{N, A2{T}})               where {N,T<:Real} = a2(x)
state(x::NTuple{N, Tuple{A2{T}, A2{T}}}) where {N,T<:Real} = a3(x, 1), a3(x, 2)


mutable struct TreeRecur{C, S}
    tree::AbstractTree{C}
    seq_model::S
end
Flux.@functor TreeRecur
(m::TreeRecur{<:OneStateCell})(x::AbstractMillNode) = m.tree(x, m.seq_model) # Pass seq_model to tree forward pass
(m::TreeRecur{<:TwoStateCell})(x::AbstractMillNode) = m.tree(x)[2]
(m::TreeRecur)(x::AbstractVector{<:AbstractMillNode}) = ChainRulesCore.ignore_derivatives() do
    return reduce(catobs, x)
end |> m


latent_empty(m::Tree)      = latent_empty(m.cell)
latent_empty(m::TreeRecur) = latent_empty(m.tree)
