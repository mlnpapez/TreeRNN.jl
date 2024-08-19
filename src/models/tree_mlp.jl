

mutable struct TreeMLPCell{T} <: OneStateCell
    w::A2{T}
    u::A2{T}
    b::A1{T}
    initial_state::A1{T}
end
Flux.@functor TreeMLPCell
function TreeMLPCell{T}(no::Int, ni::Int; init=Flux.glorot_uniform) where {T<:Real}
    return TreeMLPCell(init(no, ni), init(no, no), init(no), zeros(T, no))
end

function (m::TreeMLPCell{T})(h::A3{T}) where {T<:Real}
    h = _sum(h; dims=3)
    h = sigmoid(m.u*h .+ m.b)

    return h
end
function (m::TreeMLPCell{T})((h, b)::Tuple{A2{T}, AlignedBags}) where {T<:Real}
    h = mapreduce(b->sum(h[:, b]; dims=2), hcat, b)
    h = sigmoid(m.u*h .+ m.b)

    return h
end
function (m::TreeMLPCell{T})(x::A2{T}) where {T<:Real}
    h = sigmoid(m.w*x .+ m.b)

    return h
end

latent_empty(m::TreeMLPCell{T}) where {T<:Real} = zeros(T, size(m.w, 1), 1)

TreeMLP(type, no, ni, x) = TreeRecur(Tree(TreeMLPCell{type}(no, ni), x, ni))
