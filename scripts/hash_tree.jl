using SHA
using Mill
using Revise
using JSON3
using Flux
using Profile

includet("utils.jl")

# Add path to data
dirdata = "data"

# HashTree structure for storing tree hashes
struct HashTree
    hash::AbstractMatrix{String}  # Store hash for current node
    children::Union{
        Nothing,  # For array nodes (leaves)
        HashTree,  # For bag nodes
        NamedTuple{K, NTuple{N, HashTree}} where {K,N}  # For product nodes
    }
end

"""
Compute hashes for node data in batch mode.
Returns HashTree containing vector/matrix of hashes.
"""
function compute_hashes(node::ArrayNode)
    if node.data isa OneHotArray

       # Vectorized category to hash conversion
        hashes = bytes2hex.(sha256.(string.(node.data.indices)))

        # Convert each category number directly to hash
        # hashes = map(cat -> bytes2hex(sha256(string(cat))), categories) 
    else
        
        # Vectorized rounding and hashing
        rounded_values = round.(node.data, digits=1)
        hashes = bytes2hex.(sha256.(string.(rounded_values)))

        println(size(node.data))
        #analyze_hash_frequencies(vec(hashes))
    end
    
    # Create new ArrayNode with hashes
    # Each column's hash stored as String in a Vector
    return HashTree(reshape(hashes, 1, length(hashes)), nothing)
end

function compute_hashes(node::ProductNode)
    # Recursively compute hashes for all children using map
    # Recursively compute hashes for all children using map
    child_hashes = map(child -> compute_hashes(child), node.data)
    
    # Get number of observations from first child
    n_obs = size(first(values(child_hashes)).hash, 2)
    
    # Combine hashes per observation using broadcasting
    combined_hashes = map(1:n_obs) do i
        # Get hashes for this observation from all children
        obs_hashes = [child_hashes[key].hash[1,i] for key in sort(collect(keys(node.data)))]
        # Combine and create final hash
        bytes2hex(sha256("product_node|" * join(obs_hashes, "|")))
    end
    #analyze_hash_frequencies(combined_hashes)
    
    return HashTree(reshape(combined_hashes, 1, length(combined_hashes)), child_hashes)
end

function compute_hashes(node::BagNode)
    # Get hashes for all children first
    child_hashes = compute_hashes(node.data)  # Returns 1×N matrix of hashes
    
    # Process each bag (unit range) separately
    n_bags = length(node.bags)
    bag_hashes = Vector{String}(undef, n_bags)
    
    for (i, bag) in enumerate(node.bags)
        # Get hashes for this bag's children
        bag_child_hashes = child_hashes.hash[1, bag]
        
        # Sort hashes to ensure permutation invariance
        # e.g. [c,a,r] and [r,a,c] will produce same hash
        sorted_hashes = sort(bag_child_hashes)
        
        # Combine into single bag hash
        bag_hashes[i] = bytes2hex(sha256("bag_node|" * join(sorted_hashes, "|")))
    end

    #analyze_hash_frequencies(bag_hashes)
    
    return HashTree(reshape(bag_hashes, 1, n_bags), child_hashes)
end
# Example:
# bag1 = [1:3] with hashes ["c","a","r"]  -> hash("bag_node|a|c|r") (one hash to represent all permutations of bag1)
# bag2 = [4:7] with hashes ["x","w","y","z"] -> hash("bag_node|w|x|y|z")

function analyze_hash_frequencies(hashes::AbstractVector{String})
    # Convert 1×n matrix to vector for easier processing
    hashes = vec(hashes)
    
    freq_dict = Dict{String, Int}()
    for hash in hashes
        freq_dict[hash] = get(freq_dict, hash, 0) + 1
    end
    
    # Print statistics
    num_unique = length(freq_dict)
    total_samples = length(hashes)
    
    println("Statistics:")
    println("Total samples: ", total_samples)
    println("Unique values: ", num_unique)
    println("Uniqueness ratio: ", round(num_unique/total_samples, digits=3))

    println("\nFrequency distribution:")
    for (hash, count) in sort(collect(freq_dict), by=x->x[2], rev=true)
        println("Hash: $(hash[1:10])... appears $count times ($(round(count/total_samples*100, digits=1))%)")
    end
    
    return freq_dict
end

function compute_uniqueness(hash_tree::HashTree)
    hashes = vec(hash_tree.hash)
    unique_count = length(unique(hashes))
    return unique_count / length(hashes)  # 1.0 means all unique, 0.0 means all same
end

function compute_novelty(generated_hash_tree::HashTree, original_hash_tree::HashTree)

    generated_hash_tree = Set(vec(generated_hash_tree.hash))
    original_hash_tree = Set(vec(original_hash_tree.hash))

    # Count hashes that appear in generated but not in training
    novel_count = length(setdiff(generated_hash_tree, original_hash_tree))

    return novel_count / length(generated_hash_tree)  # 1.0 means all novel, 0.0 means all seen before
end

"""
Compute Jensen-Shannon Divergence between two categorical distributions.
Works even when distributions have different categories.
Returns value between 0 (identical) and 1 (completely different).
"""
function compute_js_divergence(dist1::Dict{String,Int}, dist2::Dict{String,Int})
    # Convert counts to probabilities
    total1 = sum(values(dist1))
    total2 = sum(values(dist2))
    
    probs1 = Dict(k => v/total1 for (k,v) in dist1)
    probs2 = Dict(k => v/total2 for (k,v) in dist2)
    
    # Get all unique categories
    all_categories = union(keys(probs1), keys(probs2))
    
    # Create average distribution M
    m = Dict{String,Float64}()
    for cat in all_categories
        p1 = get(probs1, cat, 0.0)
        p2 = get(probs2, cat, 0.0)
        m[cat] = (p1 + p2) / 2
    end
    
    # Compute KL divergence for each distribution to M
    function kl_div(p::Dict, q::Dict)
        sum = 0.0
        for (k,p_k) in p
            q_k = get(q, k, 1e-10)  # Small constant for unseen categories
            sum += p_k * log2(p_k / q_k)
        end
        return sum
    end
    
    # JSD = (KL(P||M) + KL(Q||M))/2
    jsd = (kl_div(probs1, m) + kl_div(probs2, m)) / 2
    
    return jsd
end

# Helper function to access nested HashTrees (similar to Mill.jl syntax)
function Base.getindex(tree::HashTree, key::Symbol)
    if tree.children isa NamedTuple  # ProductNode case
        return getproperty(tree.children, key)
    elseif tree.children isa HashTree  # BagNode case
        if key == :children  # Special case to access bag's child
            return tree.children
        end
        throw(ArgumentError("BagNode only supports :data indexing"))
    elseif tree.children === nothing  # ArrayNode case
        throw(ArgumentError("Cannot index into ArrayNode"))
    else
        throw(ArgumentError("Unknown node type"))
    end
end

# Experimenting
function test_hashing()
    # Load your data
    dataset = datasets[1].name
    data = JSON3.read(read("$(dirdata)/$(dataset).json", String))
    x, y = data.x, data.y

    s = schema(x)
    e = suggestextractor(s)
    printtree(e)
    x = reduce(catobs, e.(x))

    subtree_array = x[:atoms]
    printtree(subtree_array)
    println(typeof(subtree_array.data))

    println("Number of obs in node: ", numobs(subtree_array))
   
    # Example:
    @time hash_tree = compute_hashes(subtree_array)
    # Returns ArrayNode with 1×26 Vector{String} containing hashes

    println(typeof(hash_tree.hash))
    display(hash_tree.hash)

    compute_uniqueness(hash_tree)
end
