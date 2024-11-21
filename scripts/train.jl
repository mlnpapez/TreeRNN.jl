using DrWatson
@quickactivate
using JSON3
using JsonGrinder
using TreeRNN
using Revise
using Profile

include("utils.jl")

dirdata = "data/"

function train_hmil()
    dataset = datasets[1].name
    data = read("$(dirdata)/$(dataset).json", String)
	data = JSON3.read(data)
    x, y = data.x, data.y

    s = schema(x)
    e = suggestextractor(s)
    x = reduce(catobs, e.(x))
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = split_data(x, y, 1)

    no = 10
    ne = length(unique(y))
    nepoc = 50
    bsize = 10

    m = Dense(no, ne) ∘ reflectinmodel(s, e, d->Dense(d=>no, sigmoid))

    gd!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, Adam(), nepoc, bsize, ne)
end

function train_rnns()
    dataset = datasets[1].name
    data = read("$(dirdata)/$(dataset).json", String)
	data = JSON3.read(data)
    x, y = data.x, data.y

    s = schema(x)
    e = suggestextractor(s)
    x = reduce(catobs, e.(x))
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = split_data(x, y, 1) # rand(1:1000)

    #printtree(x_trn)
    #printtree(x_val)
    #printtree(x_tst)

    ni = 80
    nh = 5
    #no = length(unique(y))
    hidden_size = 10
    nepoc = 4
    bsize = 121

    m = TreeGRU(Float32, nh, ni, x_trn, hidden_size)

    #m = Dense(nh, no) ∘ TreeGRU(Float32, nh, ni, x_trn)

    #= Print model parameters for debugging
    p = Flux.params(m)
    println(length(p))
    for i in 1:length(p)
        println("\nSize of model params", i ,": ", length(p[i]))
        println(size(p[i]))
    end  =#
    
    #gd!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, Adam(0.01), nepoc, bsize, no)
    @time gd_unsupervised!(m, x_trn, x_val, x_tst, Adam(0.01), nepoc, bsize)
end

# train_hmil()
train_rnns()

nothing
