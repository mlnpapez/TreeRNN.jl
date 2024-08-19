using Mill
using Flux
using Random
using Printf
using Statistics
using JsonGrinder
using OneHotArrays
using Mill: OneHotArray, MaybeHotArray, OneHotMatrix


dirdata = "/data"
datasets = [
    (name="mutagenesis",     ndata=188,   nclass=2 ) # 1
    (name="genes",           ndata=862,   nclass=15) # 2
    (name="cora",            ndata=2708,  nclass=7 ) # 3
    (name="citeseer",        ndata=3312,  nclass=6 ) # 4
    (name="webkp",           ndata=877,   nclass=5 ) # 5
    (name="world",           ndata=239,   nclass=7 ) # 6
    (name="chess",           ndata=295,   nclass=3 ) # 7
    (name="uw_cse",          ndata=278,   nclass=4 ) # 8
    (name="hepatitis",       ndata=500,   nclass=2 ) # 9
    (name="pubmed_diabetes", ndata=19717, nclass=3 ) # 10
    (name="ftp",             ndata=30000, nclass=3 ) # 11
    (name="ptc",             ndata=343,   nclass=2 ) # 12
    (name="dallas",          ndata=219,   nclass=7 ) # 13
    (name="premier_league",  ndata=380,   nclass=3 ) # 14
]
attributes = (dataset=map(d->d.name, datasets), data_per_class=map(d->round(Int, d.ndata/d.nclass), datasets))


Base.length(x::Mill.ProductNode) = Mill.numobs(x)


function split_data(x::Mill.AbstractMillNode, y::AbstractArray{Ti,1}, seed::Ti=Ti(1), ratio::Array{Tr,1}=[64f-2, 16f-2, 2f-1]) where {Tr<:Real,Ti<:Int}
    Random.seed!(seed)
    i = randperm(length(y))
    n = cumsum(map(n->ceil(Ti, n), ratio*length(x)))

    x_trn = x[i[1:n[1]]]
    x_val = x[i[n[1]+1:n[2]]]
    x_tst = x[i[n[2]+1:end]]

    y_trn = y[i[1:n[1]]]
    y_val = y[i[n[1]+1:n[2]]]
    y_tst = y[i[n[2]+1:end]]

    x_trn, x_val, x_tst, y_trn, y_val, y_tst
end

function default_scalar_extractor()
	[
	(e -> length(keys(e)) <= 100 && JsonGrinder.is_numeric_or_numeric_string(e),
		(e, uniontypes) -> ExtractCategorical(keys(e), uniontypes)),
	(e -> JsonGrinder.is_intable(e),
		(e, uniontypes) -> JsonGrinder.extractscalar(Int32, e, uniontypes)),
	(e -> JsonGrinder.is_floatable(e),
	 	(e, uniontypes) -> JsonGrinder.extractscalar(FloatType, e, uniontypes)),
	(e -> (keys_len = length(keys(e)); keys_len < 1000 && !JsonGrinder.is_numeric_or_numeric_string(e)),
		(e, uniontypes) -> ExtractCategorical(keys(e), uniontypes)),
	(e -> true,
		(e, uniontypes) -> JsonGrinder.extractscalar(JsonGrinder.unify_types(e), e, uniontypes)),]
end




function reliability(y::Vector{<:Int}, p::Matrix{T}, ::Val{M}) where {T<:Real,M}
    o = mapslices(findmax, p, dims=1)
    p̂ = getindex.(o, 1)
    ŷ = getindex.(o, 2)
    n = length(y)

    bin = map(m->((m-1)/M, m/M), 1:M)
    idx = map(bin) do b
        mapreduce(vcat, enumerate(p̂)) do (j, p)
            ((b[1] <= p) && (p <= b[2])) ? j : []
        end
    end

    acc = map(i->length(i) > 0 ? mean(ŷ[i] .== y[i]) : T(0e0), idx)
    con = map(i->length(i) > 0 ? mean(p̂[i])          : T(0e0), idx)
    his = map(length, idx)

    ece = mapreduce((i, a, c)->(length(i)/n)*abs(a-c), +, idx, acc, con)

    acc, con, ece, his
end

entropy(p::Matrix{T}; dims=1) where {T<:Real} = -sum(p.*log.(p); dims=dims)[:]

function outlier_entropy(p::Matrix{T}, ::Val{V}, ::Val{M}) where {T<:Real,V,M}
    e = entropy(p)
    n = size(p, 2)

    thr = V*collect(1:M) / M
    idx = map(thr) do t
        mapreduce(vcat, enumerate(e)) do (j, e)
            e >= t ? j : []
        end
    end

    his = map(length, idx) / n

    thr, his
end


function evaluate(m, x_trn::Ar, x_val::Ar, x_tst::Ar, y_trn::Ai, y_val::Ai, y_tst::Ai) where {Ar<:Mill.AbstractMillNode,Ai<:Vector{<:Int}}
    acc_trn = mean(Flux.onecold(softmax(m(x_trn))) .== y_trn)
    acc_val = mean(Flux.onecold(softmax(m(x_val))) .== y_val)
    acc_tst = mean(Flux.onecold(softmax(m(x_tst))) .== y_tst)

    acc_bin_trn, con_bin_trn, ece_trn, his_trn = reliability(y_trn, softmax(m(x_trn)), Val(10))
    acc_bin_val, con_bin_val, ece_val, his_val = reliability(y_val, softmax(m(x_val)), Val(10))
    acc_bin_tst, con_bin_tst, ece_tst, his_tst = reliability(y_tst, softmax(m(x_tst)), Val(10))

    (; acc_trn,     acc_val,     acc_tst,
       ece_trn,     ece_val,     ece_tst,
       his_trn,     his_val,     his_tst,
       acc_bin_trn, acc_bin_val, acc_bin_tst,
       con_bin_trn, con_bin_val, con_bin_tst)
end

obj(m, x, y, n) = Flux.Losses.logitcrossentropy(m(x), OneHotArrays.onehotbatch(y, 1:n))

function gd!(m, x_trn::Ar, x_val::Ar, x_tst::Ar,
                y_trn::Ai, y_val::Ai, y_tst::Ai,
                o, nepoc::Int, bsize::Int, ne::Int, config_exp=nothing, config_wat=nothing, folder=""; p::Flux.Params=Flux.params(m), ftype::Type=Float32) where {Ar<:Mill.AbstractMillNode,Ai<:AbstractArray{<:Int,1}}
    t_trn, a_trn, a_val, a_tst = ftype[], ftype[], ftype[], ftype[]
    d_trn = Flux.DataLoader((x_trn, y_trn); batchsize=bsize)
    final = :maximum_iterations
    o_trn = -ftype(Inf)
    o_val = +ftype(0e0)

    for e in 1:nepoc
        t̄_trn = @elapsed begin
            for (x_trn, y_trn) in d_trn
                g = gradient(()->obj(m, x_trn, y_trn, ne), p)
                Flux.Optimise.update!(o, p, g)
            end
        end

        eval = evaluate(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst)

        a_dif = eval.acc_trn - o_trn
        o_trn = eval.acc_trn

        abs(a_dif) <= -ftype(1e-8) && (final = :absolute_tolerance; break)
        isnan(eval.acc_trn)        && (final = :nan;                break)

        if mod(e, 2) == 1
            push!(t_trn, t̄_trn)
            push!(a_trn, eval.acc_trn)
            push!(a_val, eval.acc_val)
            push!(a_tst, eval.acc_tst)
        end

        @printf("gd: epoch: %i | a_trn %2.2f | a_val %2.2f | a_tst %2.2f |\n",
            e, eval.acc_trn, eval.acc_val, eval.acc_tst)

        if (eval.acc_val > o_val) && (config_exp !== nothing) && (config_wat !== nothing)
            npars = length(Flux.destructure(m)[1])
            produce_or_load(datadir("tree_structures/$(folder)"), config_exp; config_wat...) do config
                ntuple2dict(merge(config, eval, (; t_trn, a_trn, a_val, a_tst, final), (; m, npars)))
            end
            o_val = eval.acc_val
        end
    end
end

function cell_builder(ctype::Symbol)
    ctype == :tree_mlp  && return TreeMLP
    ctype == :tree_gru  && return TreeGRU
    ctype == :tree_lstm && return TreeLSTM
end




const Maybe{T} = Union{T, Missing}
const RAND_STR_LEN = 25

make_missing(x::Mill.ArrayNode,   r::Real) = Mill.ArrayNode(make_missing(x.data, r), x.metadata)
make_missing(x::Mill.BagNode,     r::Real) = Mill.BagNode(make_missing(x.data, r), x.bags, x.metadata)
make_missing(x::Mill.ProductNode, r::Real) = Mill.ProductNode(map(x->make_missing(x, r), x.data), x.metadata)

make_missing(o::Array{T, N},          n::Int, r) where {T, N} = (selectdim(o, N, rand(1:n, clamp(round(Int, n*r), 0, n))) .= missing; o)
make_missing(x::Array{T, N}, t::Type, n::Int, r) where {T, N} = make_missing(Array{Maybe{t}, N}(x), n, r)

make_missing(x::OneHotArray{T},   r) where {T<:Integer}                              = MaybeHotMatrix(make_missing(x.indices, T, size(x, 2), r), size(x, 1))
make_missing(x::MaybeHotArray{T}, r) where {T<:Maybe{Integer}}                       = MaybeHotMatrix(make_missing(x.I,       T, size(x, 2), r), size(x, 1))
make_missing(x::Array{T},         r) where {T<:Union{U, Maybe{U}}} where {U<:Real}   =                make_missing(x,         T, size(x, 2), r)
make_missing(x::NGramMatrix{T},   r) where {T<:Union{U, Maybe{U}}} where {U<:String} = NGramMatrix(   make_missing(x.S,       T, size(x, 2), r))


make_uniform(x::Mill.ArrayNode)   = Mill.ArrayNode(make_uniform(x.data), x.metadata)
make_uniform(x::Mill.BagNode)     = Mill.BagNode(make_uniform(x.data), x.bags, x.metadata)
make_uniform(x::Mill.ProductNode) = Mill.ProductNode(map(x->make_uniform(x), x.data), x.metadata)

make_uniform(x::OneHotArray{T})   where {T<:Integer}                              = OneHotMatrix(  rand(1:size(x, 1), size(x, 2)), size(x, 1))
make_uniform(x::MaybeHotArray{T}) where {T<:Maybe{Integer}}                       = MaybeHotMatrix(rand(1:size(x, 1), size(x, 2)), size(x, 1))
make_uniform(x::Array{T})         where {T<:Union{U, Maybe{U}}} where {U<:Real}   = rand(U, size(x)...)
make_uniform(x::NGramMatrix{T})   where {T<:Union{U, Maybe{U}}} where {U<:String} = NGramMatrix(map(_->randstring(rand(1:RAND_STR_LEN)), 1:size(x, 2)))
