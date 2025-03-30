using JsonGrinder, Mill, Flux, JSON3, MLUtils, Statistics, Revise

using Random; Random.seed!(42);

includet("../../scripts/utils.jl")
includet("factorization_tree.jl")
includet("../models/rnn_model.jl")
includet("../models/lstm_model.jl")
includet("../models/gru_model.jl")
includet("../models/stacked_model.jl")
includet("../models/input_adapter.jl")

dirdata = "data/"

dataset = datasets[1].name
println(dataset)
data = read("$(dirdata)/$(dataset).json")

data = JSON3.read(data)
x, y = data.x, data.y

# data = JSON.parsefile("mutagenesis.json") 

sch = schema(x)

delete!(sch, :mutagenic)

e = suggestextractor(sch)

observation = e(x[1])
observations = e.(x)

#encoder = reflectinmodel(sch, e)

x = reduce(catobs, observations)
x_trn, x_val, x_tst, y_trn, y_val, y_tst = split_data(x, y, 1)


printtree(sch)
printtree(e)
printtree(x)
#printtree(encoder)

# model = vec ∘ Dense(10, 1) ∘ encoder

# Before building tree, convert all OneHot data to Float32
function prepare_mill_data(observation)
    function convert_node(node::Mill.ArrayNode)
        if node.data isa OneHotMatrix
            return Mill.ArrayNode(Float32.(node.data))
        end
        return node
    end
    
    function process_tree(node)
        if node isa Mill.ArrayNode
            return convert_node(node)
        elseif node isa Mill.ProductNode
            new_data = NamedTuple(
                name => process_tree(child) 
                for (name, child) in zip(keys(node.data), node.data)
            )
            return Mill.ProductNode(new_data)
        elseif node isa Mill.BagNode
            processed_data = process_tree(node.data)
            return Mill.BagNode(processed_data, node.bags)
        end
    end
    
    return process_tree(observation)
end

observation_float = prepare_mill_data(observations[1])

printtree(observation_float[:atoms].data[1])

# Create and test tree
base_model = GRU(5, 10, 10)

adapted_model = InputAdapter(base_model, 5)

tree = build_factorization_tree(observation_float, adapted_model)
# tree = build_factorization_tree(observation_float[:atoms], model)

# Test left-to-right traversal
#log_prob_l2r, data_l2r = compute_probability(tree, tree.root; direction=:left_to_right) #
#log_prob_l2r

# Test right-to-left traversal
log_prob_r2l, data_r2l = compute_probability(tree, tree.root; direction=:right_to_left) #
log_prob_r2l
