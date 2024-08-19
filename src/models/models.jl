const A0{T} = AbstractArray{T, 0}
const A1{T} = AbstractArray{T, 1}
const A2{T} = AbstractArray{T, 2}
const A3{T} = AbstractArray{T, 3}

const Code = Union{AbstractVector{<:Integer}, Base.CodeUnits}
const Sequence = Union{AbstractString, Code}


abstract type AbstractCell end
abstract type OneStateCell <: AbstractCell end
abstract type TwoStateCell <: AbstractCell end

_decomp(x::AbstractArray, n::Int, ::Val{N}) where {N} = _ind(x, n, N), _exc(x, n, N)
_expand(x::AbstractArray,         ::Val{N}) where {N} = ntuple(i->_ind(x, i, N), N)

_ind(n::Int, i::Int) = (1:n) .+ n*(i-1)
_ind(x::AbstractArray{T, 1}, i::Int, d::Int) where {T<:Real} = x[_ind(Int(size(x, 1) / d), i)]
_ind(x::AbstractArray{T, 2}, i::Int, d::Int) where {T<:Real} = x[_ind(Int(size(x, 1) / d), i), :]

_exc(x::AbstractArray, n::Int, m::Int) = mapreduce(i->_ind(x, i, m), vcat, filter(j->j!=n, 1:m))

_sum(x::AbstractArray; dims=:) = dropdims(sum(x; dims=dims); dims=dims)

Base.size(m::AlignedBags) = length(m)
# Base.length(x::Mill.ProductNode) = Mill.nobs(x)


include("tree.jl")

include("tree_mlp.jl")
include("tree_gru.jl")
include("tree_lstm.jl")
