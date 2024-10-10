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
using LinearAlgebra
using StatsBase

"""
Generate initial probabilities and transition matrix
"""
function generate_probabilities(vocab_size::Int)
    # Initial probabilities
    initial_probs = rand(Float64, vocab_size)
    initial_probs ./= sum(initial_probs)  # Normalize

    # Transition matrix
    transition_matrix = rand(Float64, (vocab_size, vocab_size))
    transition_matrix ./= sum(transition_matrix, dims=2)  # Normalize rows

    return initial_probs, transition_matrix
end

"""
Generate a single sequence based on defined probabilities
"""
function generate_sequence(initial_probs::Vector{Float64}, transition_matrix::Matrix{Float64}, max_length::Int)
    length = rand(3:max_length)
    sequence = Vector{Int}(undef, length)
    
    # Generate first character
    sequence[1] = sample(1:size(initial_probs, 1), Weights(initial_probs))
    
    # Generate subsequent characters
    for i in 2:length
        prev_char = sequence[i-1]
        sequence[i] = sample(1:size(transition_matrix, 2), Weights(transition_matrix[prev_char, :]))
    end
    
    return sequence
end

"""
Generate a dataset of sequences
"""
function generate_dataset(initial_probs::Vector{Float64}, transition_matrix::Matrix{Float64}, num_samples::Int, max_length::Int)
    return [generate_sequence(initial_probs, transition_matrix, max_length) for _ in 1:num_samples]
end

"""
Calculate the joint probability of a sequence
"""
function joint_probability(sequence::Vector{Int}, initial_probs::Vector{Float64}, transition_matrix::Matrix{Float64})
    p = initial_probs[sequence[1]]
    for i in 2:length(sequence)
        p *= transition_matrix[sequence[i-1], sequence[i]]
    end
    return p
end

"""
Calculate conditional probabilities for a sequence
"""
function conditional_probabilities(sequence::Vector{Int}, initial_probs::Vector{Float64}, transition_matrix::Matrix{Float64})
    # Pre-allocation of vector
    cond_probs = Vector{Tuple{Int, Union{Nothing, Int}, Float64}}(undef, length(sequence))
    cond_probs[1] = (1, nothing, initial_probs[sequence[1]])
    for i in 2:length(sequence)
        cond_probs[i] = (i, sequence[i-1], transition_matrix[sequence[i-1], sequence[i]])
    end
    return cond_probs
end

function main()
    Random.seed!(42)  # For reproducibility
    
    vocab_size = 10

    # Generate probabilities
    initial_probs, transition_matrix = generate_probabilities(vocab_size)

    println("Generated Probabilities:")
    println("\nInitial Probabilities:")
    for (i, prob) in enumerate(initial_probs)
        println("  P($i) = $(round(prob, digits=4))")
    end
    println("\nSum of probabilities: ", sum(initial_probs))

    println("\nTransition Matrix:")
    for i in 1:vocab_size
        println("  From $i:")
        for j in 1:vocab_size
            println("    P($j|$i) = $(round(transition_matrix[i,j], digits=4))")
        end
    println("Sum of transition probabilities P(X|$i): ", sum(transition_matrix[i,:]), "\n")
    end

    # Generate dataset based on generated probabilities
    max_length = 14
    num_samples = 10000
    dataset = generate_dataset(initial_probs, transition_matrix, num_samples, max_length)


    println("\nGenerated Dataset:")
    for (i, sample) in enumerate(dataset[1:min(10, length(dataset))])
        println("\nSample $i: ", join(['a' + x - 1 for x in sample]))
        
        # Calculate and print joint probability
        jp = joint_probability(sample, initial_probs, transition_matrix)
        println("Joint Probability P(X₁:ₙ) = $(round(jp, digits=8))")
        
        # Calculate and print conditional probabilities
        cp = conditional_probabilities(sample, initial_probs, transition_matrix)
        println("Conditional Probabilities:")
        for (i, prev, prob) in cp
            if isnothing(prev)
                println("  P(X$i=$('a' + sample[i] - 1)) = $(round(prob, digits=4))")
            else
                println("  P(X$i=$('a' + sample[i] - 1)|X$(i-1)=$('a' + prev - 1)) = $(round(prob, digits=4))")
            end
        end 
    end
end

# Run the main function
# main()