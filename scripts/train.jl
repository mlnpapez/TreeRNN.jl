using DrWatson
@quickactivate
using JSON3
using JsonGrinder
using TreeRNN
using Revise
using Profile

include("utils.jl")
include("prob_tree.jl")

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

    #printtree(x[1])
    #printtree(x_trn)
    #printtree(x_val)
    #printtree(x_tst)

    ni = 80
    nh = 5
    #no = length(unique(y))
    hidden_size = 10
    nepoc = 100
    bsize = 121
    patience = 15

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
    @time best_model, _ = gd_unsupervised!(m, x_trn, x_val, x_tst, Adam(0.01), nepoc, bsize; patience)

    println(typeof(best_model))
    original = x_trn

    printtree(original)

    # Generate new structure
    new_structure = generate_from_structure(original, best_model)

    # Print both structures to compare
    println("Original structure:")
    printtree(original)
    display(original[:lumo])
    display(original[:inda])
    display(original[:ind1])
    display(original[:atoms].data[:element])
    display(original[:atoms].data[:bonds].data[:element])
    display(original[:atoms].data[:bonds].data[:type_bond])

    println("\nGenerated structure:")
    printtree(new_structure)
    display(new_structure[:lumo])
    display(new_structure[:inda])
    display(new_structure[:ind1])
    display(new_structure[:atoms].data[:element])
    display(new_structure[:atoms].data[:bonds].data[:element])
    display(original[:atoms].data[:bonds].data[:type_bond])
end

# train_hmil()
train_rnns()

nothing
