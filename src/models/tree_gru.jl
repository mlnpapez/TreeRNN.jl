

mutable struct TreeGRUCell{T} <: OneStateCell
    w::A2{T}
    u::A2{T}
    b::A1{T}
    initial_state::A1{T}
end
Flux.@functor TreeGRUCell
function TreeGRUCell{T}(no::Int, ni::Int; init=Flux.glorot_uniform) where {T<:Real}
    return TreeGRUCell(init(no * 3, ni), init(no * 3, no), init(no * 3), zeros(T, no))
end

function (m::TreeGRUCell{T})(h::A3{T}) where {T<:Real}
    ur, uh, uz = _expand(m.u, Val(3))
    br, bh, bz = _expand(m.b, Val(3))

    r = reshape(mapreduce(h->ur*h .+ br, hcat, eachslice(h; dims=3)), size(h)...)
    h̃ = _sum(h; dims=3)
    ĥ = uh*_sum(sigmoid(r) .* h; dims=3) .+ bh
    z = sigmoid(uz*h̃ .+ bz)

    h = z .* h̃ + (1 .- z) .* tanh.(ĥ)

    return h
end
function (m::TreeGRUCell{T})((h, b)::Tuple{A2{T}, AlignedBags}) where {T<:Real}
    ur, uh, uz = _expand(m.u, Val(3))
    br, bh, bz = _expand(m.b, Val(3))

    r = sigmoid(ur*h .+ br) .* h
    h = mapreduce(b->sum(h[:, b]; dims=2), hcat, b)
    r = mapreduce(b->sum(r[:, b]; dims=2), hcat, b)
    ĥ =   tanh.(uh*r .+ bh)
    z = sigmoid(uz*h .+ bz)

    h = z .* h + (1 .- z) .* ĥ

    return h
end
function (m::TreeGRUCell{T})(x::A2{T}) where {T<:Real}
    w = _exc(m.w, 1, 3)
    b = _exc(m.b, 1, 3)

    g = w*x .+ b

    ĥ, z = _expand(g, Val(2))

    h = (1 .- sigmoid(z)) .* tanh.(ĥ)

    return h
end

latent_empty(m::TreeGRUCell{T}) where {T<:Real} = zeros(T, size(m.w, 1) ÷ 3, 1)

TreeGRU(type, no, ni, x) = TreeRecur(Tree(TreeGRUCell{type}(no, ni), x, ni))
