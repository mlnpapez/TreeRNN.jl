

mutable struct TreeLSTMCell{T} <: TwoStateCell
    w::A2{T}
    u::A2{T}
    b::A1{T}
    initial_state::A1{T}
end
Flux.@functor TreeLSTMCell
function TreeLSTMCell{T}(no::Int, ni::Int; init=Flux.glorot_uniform) where {T<:Real}
    return TreeLSTMCell(init(no * 4, ni), init(no * 4, no), init(no * 4), zeros(T, no))
end

function (m::TreeLSTMCell{T})((c, h)::Tuple{A3{T}, A3{T}}) where {T<:Real}
    uf, uu = _decomp(m.u, 2, Val(4))
    bf, bb = _decomp(m.b, 2, Val(4))

    f = reshape(mapreduce(h->uf*h .+ bf, hcat, eachslice(h; dims=3)), size(h)...)
    g = uu*_sum(h; dims=3) .+ bb

    i, o, u = _expand(g, Val(3))

    c = sigmoid(i) .* tanh.(u) + _sum(sigmoid(f) .* c; dims=3)
    h = o .* tanh.(c)

    return c, h
end
function (m::TreeLSTMCell{T})(((c, h), b)::Tuple{Tuple{A2{T}, A2{T}}, AlignedBags}) where {T<:Real}
    uf, uu = _decomp(m.u, 2, Val(4))
    bf, bb = _decomp(m.b, 2, Val(4))

    r = sigmoid(uf*h .+ bf) .* c
    h = mapreduce(b->sum(h[:, b]; dims=2), hcat, b)
    r = mapreduce(b->sum(r[:, b]; dims=2), hcat, b)

    g = uu*h .+ bb

    i, o, u = _expand(g, Val(3))

    c = sigmoid(i) .* tanh.(u) + r
    h = o .* tanh.(c)

    return c, h
end
function (m::TreeLSTMCell{T})(x::A2{T}) where {T<:Real}
    g = m.w*x .+ m.b

    i, f, o, u = _expand(g, Val(4))

    c = sigmoid(i) .* tanh.(u) + sigmoid(f) .* m.initial_state
    h = o .* tanh.(c)

    return c, h
end

latent_empty(m::TreeLSTMCell{T}) where {T<:Real} = zeros(T, size(m.w, 1) ÷ 4, 1), zeros(T, size(m.w, 1) ÷ 4, 1)

TreeLSTM(type, no, ni, x) = TreeRecur(Tree(TreeLSTMCell{type}(no, ni), x, ni))
