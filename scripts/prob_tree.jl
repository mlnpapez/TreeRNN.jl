
using DrWatson
@quickactivate
using Mill
using Flux
using JSON3
using Revise
using Flux
using LinearAlgebra

using TreeRNN

includet("utils.jl")

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

    bags = x[:atoms].data[:bonds].bags
    println("Number of bags/unit ranges: ", size(bags))

    # Build supervised model as before
    ni, nh = 80, 5

    m_charge = TreeGRU(Float32, nh, ni, subtree_charge)

    emb_charge = m_charge(subtree_charge)
    println(size(emb_charge))


    # For embedding matrix 5×10486
    # Initialize GRU matching dimensions:
    input_size = 5    # embedding dimension
    hidden_size = 8   # desired hidden state size

    # Initialize GRU
    gru = GRU(input_size, hidden_size)

    h_charge = emb_charge |> gru
    println("\nHidden state: ", size(h_charge))
    display(h_charge[:, 1:6])

    log_probs_charge = get_log_probs(gru, h_charge, subtree_charge.data)
    println("\nSize of log probs matrix of charge: ", size(log_probs_charge))
    display(log_probs_charge)


    # Expand hidden state for element conditioning
    h_expanded = expand_hidden_state(h_charge, bags)  # 8×10486
    println("\nExpanded hidden: ", size(h_expanded))
    display(h_expanded[:, bags[1]])

    subtree_element = x[:atoms].data[:bonds].data[:element]
    printtree(subtree_element)

    # Get probabilities matching data type/size
    log_probs_expanded = get_log_probs(gru, h_expanded, subtree_element.data)
    # Check first few probabilities match bag structure
    println("\nFirst bag probabilities:")
    display(log_probs_expanded[:, bags[1]]) # should be conditioned on first charge


    m_element = TreeGRU(Float32, nh, ni, subtree_element)
    emb_element = m_element(subtree_element)
    println("\nSize of element embeddings: ", size(emb_element))
    println("Size of gru hidden state: ", size(gru.state))
    h_element = gru(emb_element, bags)
    println("\nHidden state: ", size(h_element))
    display(h_element[:, 1:6])

    # Get probabilities matching data type/size
    log_probs_element = get_log_probs(gru, h_element, subtree_element.data)
    println("\nLog probs of the first 6 element obs: ")
    display(log_probs_element[:, 1:6])
end