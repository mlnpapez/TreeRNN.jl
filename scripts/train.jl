using DrWatson
@quickactivate
using JSON3
using JsonGrinder
using TreeRNN
using Revise
using Profile

include("utils.jl")
include("prob_tree.jl")
include("hash_tree.jl")

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
    nepoc = 5
    bsize = 121
    patience = 15
    initial_lr = 0.01
    decay_steps = 10

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
    @time best_model, stats = gd_unsupervised!(m, x_trn, x_val, x_tst, initial_lr, nepoc, bsize; patience, decay_steps)

    println(typeof(best_model))
    printtree(x_trn)

    # Generate new structure
    @time new_set = generate_from_structure(x_trn, best_model)

    #= Print both structures to compare
    println("Original set:")
    printtree(x_trn)
    display(x_trn[:lumo])
    display(x_trn[:inda])
    display(x_trn[:ind1])
    display(x_trn[:atoms].data[:element])
    display(x_trn[:atoms].data[:bonds].data[:element])
    display(x_trn[:atoms].data[:bonds].data[:type_bond])

    println("\nGenerated set:")
    printtree(new_set)
    display(new_set[:lumo])
    display(new_set[:inda])
    display(new_set[:ind1])
    display(new_set[:atoms].data[:element])
    display(new_set[:atoms].data[:bonds].data[:element])
    display(new_set[:atoms].data[:bonds].data[:type_bond])=#

    println("---------Train set----------")
    hash_trn = compute_hashes(x_trn)
    uniqueness_trn = compute_uniqueness(hash_trn)
    println("uniqueness_trn: ", round(uniqueness_trn * 100, digits = 3), "%")

    println("---------Validation set----------")
    hash_val = compute_hashes(x_val)
    uniqueness_val = compute_uniqueness(hash_val)
    println("uniqueness_val: ", round(uniqueness_val * 100, digits = 3), "%")

    println("---------Test set----------")
    hash_tst = compute_hashes(x_tst)
    uniqueness_tst = compute_uniqueness(hash_tst)
    println("uniqueness_tst: ", round(uniqueness_tst * 100, digits = 3), "%")

    println("---------Novelty----------")
    novelty_val_trn = compute_novelty(hash_val, hash_trn)
    println("novelty_val_trn: ", round(novelty_val_trn * 100, digits = 3), "%")

    novelty_tst_trn = compute_novelty(hash_tst, hash_trn)
    println("novelty_tst_trn: ", round(novelty_tst_trn * 100, digits = 3), "%")

    novelty_tst_val = compute_novelty(hash_tst, hash_val)
    println("novelty_tst_val: ", round(novelty_tst_val * 100, digits = 3), "%")#

    println("---------Compare distributions----------")
    # Get frequencies
    freq_trn = analyze_hash_frequencies(vec(hash_trn.hash))
    freq_val = analyze_hash_frequencies(vec(hash_val.hash))
    freq_tst = analyze_hash_frequencies(vec(hash_tst.hash))
   
    # Compare distributions
    jsd1 = compute_js_divergence(freq_trn, freq_val)

    println("\nDistribution Matching (train vs. validation):")
    println("Jensen-Shannon Divergence: ", round(jsd1, digits=3))
    println("(0 = identical, 1 = completely different)")

    jsd2 = compute_js_divergence(freq_trn, freq_tst)

    println("\nDistribution Matching (train vs. test):")
    println("Jensen-Shannon Divergence: ", round(jsd2, digits=3))
    println("(0 = identical, 1 = completely different)")

    jsd3 = compute_js_divergence(freq_val, freq_tst)

    println("\nDistribution Matching (validation vs. test):")
    println("Jensen-Shannon Divergence: ", round(jsd3, digits=3))
    println("(0 = identical, 1 = completely different)")

    #=
    println("---------Generated set----------")
    hash_new_set = compute_hashes(new_set)
    uniqueness_new_set = compute_uniqueness(hash_new_set)
    println("uniqueness_new_set: ", round(uniqueness_new_set * 100, digits = 3), "%")

    novelty_new_set = compute_novelty(hash_new_set, hash_trn)
    println("novelty_new_set: ", round(novelty_new_set * 100, digits = 3), "%")

    #display(hash_new_set[:atoms].children[:charge].hash)
    #println(typeof(hash_new_set[:atoms].children[:charge].hash))

    # Get and print frequencies
    freq_element1 = analyze_hash_frequencies(vec(hash_new_set[:atoms].children[:element].hash))
    freq_element2 = analyze_hash_frequencies(vec(hash_trn[:atoms].children[:element].hash))

    freq_charge1 = analyze_hash_frequencies(vec(hash_new_set[:atoms].children[:charge].hash))
    freq_charge2 = analyze_hash_frequencies(vec(hash_trn[:atoms].children[:charge].hash))

    # Compare distributions
    jsd1 = compute_js_divergence(freq_element1, freq_element2)

    println("\nDistribution Matching (element):")
    println("Jensen-Shannon Divergence: ", round(jsd1, digits=3))
    println("(0 = identical, 1 = completely different)")#

    jsd2 = compute_js_divergence(freq_charge1, freq_charge2)

    println("\nDistribution Matching (charge):")
    println("Jensen-Shannon Divergence: ", round(jsd2, digits=3))
    println("(0 = identical, 1 = completely different)")=#
end

# train_hmil()
train_rnns()

nothing
