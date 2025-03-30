using Test
include("../scripts/sequences/data_generator.jl")

@testset "Data Generator Tests" begin
    @testset "generate_probabilities" begin
        vocab_size = 3
        initial_probs, transition_matrix = generate_probabilities(vocab_size)

        @test length(initial_probs) == vocab_size
        @test size(transition_matrix) == (vocab_size, vocab_size)
        @test isapprox(sum(initial_probs), 1.0, atol=1e-8)
        @test all(isapprox.(sum(transition_matrix, dims=2), 1.0, atol=1e-8))
    end

    @testset "generate_sequence" begin
        vocab_size = 3
        max_length = 10
        initial_probs, transition_matrix = generate_probabilities(vocab_size)
        sequence = generate_sequence(initial_probs, transition_matrix, max_length)

        @test 3 <= length(sequence) <= max_length
        @test all(1 .<= sequence .<= vocab_size)
    end

    @testset "generate_dataset" begin
        vocab_size = 3
        num_samples = 100
        max_length = 10
        initial_probs, transition_matrix = generate_probabilities(vocab_size)
        dataset = generate_dataset(initial_probs, transition_matrix, num_samples, max_length)

        @test length(dataset) == num_samples
        @test all(3 .<= length.(dataset) .<= max_length)
        @test all(all(1 .<= seq .<= vocab_size) for seq in dataset)
    end

    @testset "joint_probability" begin
        vocab_size = 3
        initial_probs, transition_matrix = generate_probabilities(vocab_size)
        sequence = [1, 2, 3, 1]
        prob = joint_probability(sequence, initial_probs, transition_matrix)

        @test 0 <= prob <= 1
        @test isapprox(prob, 
            initial_probs[1] * 
            transition_matrix[1, 2] * 
            transition_matrix[2, 3] * 
            transition_matrix[3, 1], 
            atol=1e-8)
    end
    
    @testset "conditional_probabilities" begin
        vocab_size = 3
        initial_probs, transition_matrix = generate_probabilities(vocab_size)
        sequence = [1, 2, 3, 1]
        cond_probs = conditional_probabilities(sequence, initial_probs, transition_matrix)

        @test length(cond_probs) == length(sequence)
        @test cond_probs[1] == (1, nothing, initial_probs[1])
        @test cond_probs[2] == (2, 1, transition_matrix[1, 2])
        @test cond_probs[3] == (3, 2, transition_matrix[2, 3])
        @test cond_probs[4] == (4, 3, transition_matrix[3, 1])
    end
end