using Test
using Flux
include("../scripts/sequences/rnn_model.jl")
include("../scripts/sequences/gru_model.jl")
include("../scripts/sequences/lstm_model.jl")
include("../scripts/sequences/stacked_model.jl")

@testset "Model Tests" begin
    input_size, hidden_size, output_size = 3, 5, 3
    x = rand(Float32, input_size, 10)  # 10 time steps
    x1 = rand(Float32, input_size, 5)
    x2 = rand(Float32, input_size, 5)

    @testset "RNN Model" begin
        model = RNN(input_size, hidden_size, output_size)

        output = softmax(model(x))
        @test size(output) == (output_size, 10)
        @test all(0 .<= output .<= 1)

        # Test state reset
        Flux.reset!(model)
        @test all(model.state .== 0)

        # Test gradients
        loss(m, x) = sum(m(x))
        grads = gradient(loss, model, x)
        @test !isnothing(grads[1])

        # Test state persistence
        output1 = model(x1)
        state1 = copy(model.state)
        output2 = model(x2)
        state2 = copy(model.state)

        @test !all(state1 .≈ state2)
        @test !all(output1 .≈ output2)

        # Test random initialization and deterministic reset
        Flux.reset!(model)
        output3 = model(x1)
        @test !all(output1 .≈ output3)
    end

    @testset "GRU Model" begin
        model = GRU(input_size, hidden_size, output_size)

        output = softmax(model(x))
        @test size(output) == (output_size, 10)
        @test all(0 .<= output .<= 1)

        # Test state reset
        Flux.reset!(model)
        @test all(model.state .== 0)

        # Test gradients
        loss(m, x) = sum(m(x))
        grads = gradient(loss, model, x)
        @test !isnothing(grads[1])

        # Test state persistence
        output1 = model(x1)
        state1 = copy(model.state)
        output2 = model(x2)
        state2 = copy(model.state)

        @test !all(state1 .≈ state2)
        @test !all(output1 .≈ output2)

        # Test random initialization and deterministic reset
        Flux.reset!(model)
        output3 = model(x1)
        @test !all(output1 .≈ output3)
    end

    @testset "LSTM Model" begin
        model = LSTM(input_size, hidden_size, output_size)

        output = softmax(model(x))
        @test size(output) == (output_size, 10)
        @test all(0 .<= output .<= 1)

        # Test state reset
        Flux.reset!(model)
        @test all(model.state .== 0)

        # Test gradients
        loss(m, x) = sum(m(x))
        grads = gradient(loss, model, x)
        @test !isnothing(grads[1])

        # Test state persistence
        output1 = model(x1)
        state1 = copy(model.state)
        output2 = model(x2)
        state2 = copy(model.state)

        @test !all(state1 .≈ state2)
        @test !all(output1 .≈ output2)

        # Test random initialization and deterministic reset
        Flux.reset!(model)
        output3 = model(x1)
        @test !all(output1 .≈ output3)
    end

    @testset "Stacked Model" begin
        input_size, hidden_sizes, output_size = 3, [5, 4], 3
        batch_size = 10
        x = rand(Float32, input_size, 10)  # 10 time steps
        x1 = rand(Float32, input_size, 5)
        x2 = rand(Float32, input_size, 5)
    
        # Test different model configurations
        @testset "Different configurations" begin
            # Two RNN layers
            model_rnn = StackedModel(input_size, [(:RNN, 5), (:RNN, 4)], output_size)
            output_rnn = model_rnn(x)
            @test size(output_rnn) == (output_size, 10)
            
            # Mixed RNN-LSTM layers
            model_mixed = StackedModel(input_size, [(:RNN, 5), (:LSTM, 4)], output_size)
            output_mixed = model_mixed(x)
            @test size(output_mixed) == (output_size, 10)
            
            # Test that different configurations give different results
            @test !all(output_rnn .≈ output_mixed)
        end
    
        # Test a specific configuration in detail
        model = StackedModel(input_size, [(:RNN, 5), (:LSTM, 4)], output_size)
    
        # Test dimensions
        output = model(x)
        @test size(output) == (output_size, 10)
        @test length(model.layers) == 2  # Number of layers
        
        # Test input type handling
        @testset "Input type handling" begin   
            x_float = zeros(Float32, input_size, batch_size)
            output_float = model(x_float)
            @test eltype(output_float) == Float32
        end
    
        # Test different layer combinations
        @testset "Layer type combinations" begin
            combinations = [
                ([(:RNN, 5), (:RNN, 4)], "RNN-RNN"),
                ([(:RNN, 5), (:LSTM, 4)], "RNN-LSTM"),
                ([(:LSTM, 5), (:RNN, 4)], "LSTM-RNN"),
                ([(:LSTM, 5), (:LSTM, 4)], "LSTM-LSTM"),
                ([(:GRU, 5), (:GRU, 4)], "GRU-GRU"),
                ([(:GRU, 5), (:LSTM, 4)], "GRU-LSTM")
            ]
    
            for (layer_specs, name) in combinations
                @testset "$name combination" begin
                    model = StackedModel(input_size, layer_specs, output_size)
                    
                    # Test forward pass
                    x = rand(Float32, input_size, batch_size)
                    output = model(x)
                    @test size(output) == (output_size, batch_size)
                    @test eltype(output) == Float32
                end
            end
        end
    
        # Test state persistence and reset
        @testset "State reset and persistence" begin
            model = StackedModel(input_size, [(:RNN, 5), (:LSTM, 4)], output_size)
            
            # Test reset
            Flux.reset!(model)
            for layer in model.layers
                @test all(layer.state .== 0)
            end
    
            # Test state changes after forward pass
            output = model(x)
            @test !all(collect(all(layer.state .== 0) for layer in model.layers))
        end
    
        # Test gradients
        @testset "Gradients" begin
            loss(m, x) = sum(m(x))
            grads = gradient(loss, model, x)
            @test !isnothing(grads[1])
        end
    
        # Test compatibility with single models
        @testset "Compatibility with single models" begin
            single_rnn = RNN(input_size, 5, output_size)
            stacked_rnn = StackedModel(input_size, [(:RNN, 5)], output_size)
            
            Flux.reset!(single_rnn)
            Flux.reset!(stacked_rnn)
            
            # Test output dimensions match
            out1 = single_rnn(x)
            out2 = stacked_rnn(x)
            @test size(out1) == size(out2)
        end
    end
end