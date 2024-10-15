using Test
using Flux
include("../scripts/sequences/rnn_model.jl")
include("../scripts/sequences/gru_model.jl")
include("../scripts/sequences/lstm_model.jl")

@testset "Model Tests" begin
    input_size, hidden_size, output_size = 3, 5, 3
    x = rand(Float32, input_size, 10)  # 10 time steps
    x1 = rand(Float32, input_size, 5)
    x2 = rand(Float32, input_size, 5)

    @testset "RNN Model" begin
        model = RNN(input_size, hidden_size, output_size)

        output = model(x)
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

        output = model(x)
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

        output = model(x)
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
end