using DrWatson
@quickactivate
using Mill
using Flux
using JSON3
using Revise
using Flux
using LinearAlgebra

using TreeRNN

includet("../src/models/models.jl")
includet("../src/models/tree.jl")
includet("utils.jl")
includet("../src/models/tree_gru.jl")
includet("../src/models/prob_gru.jl")


# Add path to data
dirdata = "data"

# Experimenting
function test_prob_model()
    # Load your data
    dataset = datasets[1].name
    data = JSON3.read(read("$(dirdata)/$(dataset).json", String))
    x, y = data.x, data.y

    s = schema(x)
    e = suggestextractor(s)
    x = reduce(catobs, e.(x))

    subtree_charge = x[:atoms].data[:charge]
    printtree(subtree_charge)

    println("Number of obs in node: ", numobs(subtree_charge))

    bags = x[:atoms].data[:bonds].bags
    println("Number of bags/unit ranges: ", size(bags))

    # Build supervised model as before
    ni, nh, hidden_size = 80, 5, 10

    m_charge = TreeGRU(Float32, nh, ni, subtree_charge, hidden_size)

    emb_charge, log_prob_embs = m_charge(subtree_charge)
    println("Embedding size: ", size(emb_charge))


    println("\n----------Start bonds example-----------")

    subtree_bonds = x

    printtree(subtree_bonds)

    m_bonds = TreeGRU(Float32, nh, ni, subtree_bonds, hidden_size)

    emb_bonds = m_bonds(subtree_bonds)
    
    p = Flux.params(m_bonds)
    println(length(p))
    for i in 1:length(p)
        println("\nSize of model params", i ,": ", length(p[i]))
    end    
end