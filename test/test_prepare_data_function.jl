using Test
include("../scripts/sequences/pipeline_index.jl")

@testset "prepare_data function tests" begin
    # Test case 1: Basic functionality
    @testset "Basic functionality" begin
        sequences = [[1, 2, 3], [2, 1, 3, 2]]
        vocab_size = 3
        X, Y, sequence_indices = prepare_data(sequences, vocab_size)
        
        @test size(X) == (3, 5)  # 3 x (2 + 3) pairs
        @test size(Y) == (3, 5)
        @test length(sequence_indices) == 5
        
        # Check if X is correctly one-hot encoded
        @test X[:, 1] == [1, 0, 0]
        @test X[:, 2] == [0, 1, 0]
        @test X[:, 3] == [0, 1, 0]
        @test X[:, 4] == [1, 0, 0]
        @test X[:, 5] == [0, 0, 1]
        
        # Check if Y is correctly one-hot encoded
        @test Y[:, 1] == [0, 1, 0]
        @test Y[:, 2] == [0, 0, 1]
        @test Y[:, 3] == [1, 0, 0]
        @test Y[:, 4] == [0, 0, 1]
        @test Y[:, 5] == [0, 1, 0]
        
        # Check sequence indices
        @test sequence_indices == [1, 1, 2, 2, 2]
    end

     # Test case 2: Empty input
     @testset "Empty input" begin
        sequences = []
        vocab_size = 3
        X, Y, sequence_indices = prepare_data(sequences, vocab_size)
        
        @test isempty(X)
        @test isempty(Y)
        @test isempty(sequence_indices)
    end

     # Test case 3: Single character sequences
     @testset "Single character sequences" begin
        sequences = [[1], [2], [3]]
        vocab_size = 3
        X, Y, sequence_indices = prepare_data(sequences, vocab_size)
        
        @test isempty(X)
        @test isempty(Y)
        @test isempty(sequence_indices)
    end

     # Test case 4: Larger vocabulary
     @testset "Larger vocabulary" begin
        sequences = [[1, 4, 2], [5, 3, 1]]
        vocab_size = 5
        X, Y, sequence_indices = prepare_data(sequences, vocab_size)
        
        @test size(X) == (5, 4)
        @test size(Y) == (5, 4)
        @test length(sequence_indices) == 4
        
        # Check if X uses the correct vocabulary size
        @test sum(X[:, 3]) == 1
        @test findmax(X[:, 2])[2] == 4
    end
end