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

function (m::Tree)(x::AbstractProductNode)
    if Mill.numobs(values(x.data)) == 0
        return latent_empty(m)
    else
        return map((m, x)->m(x), m.children, x.data) |> values |> state |> m.cell
    end
end
(m::Tree)(x::AbstractBagNode) = (m.children(x.data), x.bags) |> m.cell
(m::Tree)(x::ArrayNode)       =  m.children(x.data)          |> m.cell

a2(x)    = reshape(hcat(x...), size(x[1])..., :)
a3(x, i) = reshape(hcat(getindex.(x, i)...), size(x[1][1])..., :)

state(x::NTuple{N, A2{T}})               where {N,T<:Real} = a2(x)
state(x::NTuple{N, Tuple{A2{T}, A2{T}}}) where {N,T<:Real} = a3(x, 1), a3(x, 2)


mutable struct TreeRecur{C}
    tree::AbstractTree{C}
end
Flux.@functor TreeRecur
(m::TreeRecur{<:OneStateCell})(x::AbstractMillNode) = m.tree(x)
(m::TreeRecur{<:TwoStateCell})(x::AbstractMillNode) = m.tree(x)[2]
(m::TreeRecur)(x::AbstractVector{<:AbstractMillNode}) = ChainRulesCore.ignore_derivatives() do
    return reduce(catobs, x)
end |> m


latent_empty(m::Tree)      = latent_empty(m.cell)
latent_empty(m::TreeRecur) = latent_empty(m.tree)
