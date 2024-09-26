#=
###############################################
#                                             #
#              Data(set) Generator            #
#                   ALICE                     #
#                                             #
###############################################

This file is responsible for 
generating our dataset based on Markov chains.

Author: Do Viet Anh
=#

using Random
using DataStructures
using StatsBase

# Define our vocabulary
const VOCAB = ['h', 'i', 'd', 'e', 'n']

"""
Generate initial and transition probabilities for our vocabulary
"""
function generate_probabilities()
    initial_probs = Dict{Char, Float64}()
    transition_probs = Dict{Char, Dict{Char, Float64}}()
    
    # Initial probabilities (marginal probabilities)
    for char in VOCAB
        initial_probs[char] = rand()
    end

    # Normalize initial probabilities (so prob values sum to one)
    total = sum(values(initial_probs))
    for (key, value) in initial_probs
        initial_probs[key] = value / total
    end
    
    # Transition probabilities
    for char1 in VOCAB
        transition_probs[char1] = Dict{Char, Float64}()
        for char2 in VOCAB
            transition_probs[char1][char2] = rand()
        end
        # Normalize transition probabilities for each starting character
        total = sum(values(transition_probs[char1]))
        for (key, value) in transition_probs[char1]
            transition_probs[char1][key] = value / total
        end
    end
    
    return initial_probs, transition_probs
end

"""
Generate a single sequence based on defined probabilities
"""
function generate_sequence(initial_probs, transition_probs)
    length = rand(2:6)
    sequence = Char[]
    
    # Generate first character
    first_char = sample(VOCAB, Weights(collect(values(initial_probs))))
    push!(sequence, first_char)
    
    # Generate subsequent characters
    for i in 2:length
        prev_char = sequence[end]
        next_char = sample(VOCAB, Weights(collect(values(transition_probs[prev_char]))))
        push!(sequence, next_char)
    end
    
    return sequence
end

"""
Generate a dataset of sequences
"""
function generate_dataset(initial_probs, transition_probs, num_samples::Int)
    return [generate_sequence(initial_probs, transition_probs) for _ in 1:num_samples]
end

"""
Calculate the joint probability of a sequence
"""
function joint_probability(sequence, initial_probs, transition_probs)
    p = initial_probs[sequence[1]]
    for i in 2:length(sequence)
        p *= transition_probs[sequence[i-1]][sequence[i]]
    end
    return p
end

"""
Calculate conditional probabilities for a sequence
"""
function conditional_probabilities(sequence, initial_probs, transition_probs)
    cond_probs = OrderedDict()
    cond_probs["P($(sequence[1]))"] = initial_probs[sequence[1]]
    for i in 2:length(sequence)
        prev = join(sequence[1:i-1])
        cond_probs["P($(sequence[i])|$prev)"] = transition_probs[sequence[i-1]][sequence[i]]
    end
    return cond_probs
end

function main()
    Random.seed!(42)  # For reproducibility

    # Generate probabilities
    initial_probs, transition_probs = generate_probabilities()

    println("Generated Probabilities:")
    println("\nInitial Probabilities:")
    for (char, prob) in sort(collect(initial_probs))
        println("  P($char) = $(round(prob, digits=4))")
    end

    println("\nTransition Probabilities:")
    for char1 in sort(collect(keys(transition_probs)))
        println("  From $char1:")
        for (char2, prob) in sort(collect(transition_probs[char1]))
            println("    P($char2|$char1) = $(round(prob, digits=4))")
        end
    end

    # Generate dataset based on generated probabilities
    num_samples = 5
    dataset = generate_dataset(initial_probs, transition_probs, num_samples)

    println("\nGenerated Dataset with Probabilities:")
    for (i, sample) in enumerate(dataset)
        println("\nSample $i: ", join(sample))
        
        # Calculate and print joint probability
        jp = joint_probability(sample, initial_probs, transition_probs)
        println("Joint Probability P(X₁:ₙ) = $(round(jp, digits=8))")
        
        # Calculate and print conditional probabilities
        cp = conditional_probabilities(sample, initial_probs, transition_probs)
        println("Conditional Probabilities:")
        for (condition, prob) in cp
            println("  $condition = $(round(prob, digits=4))")
        end
    end
end

# Run the main function
# main()