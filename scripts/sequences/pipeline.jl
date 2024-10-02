using Flux
using Statistics
using Flux: DataLoader
using Statistics: mean
using LinearAlgebra

include("data_generator.jl")
include("gru_model.jl")
include("lstm_model.jl")
include("rnn_model.jl")


"""
Prepare sequence data for the sequential model
"""
function prepare_data(sequences, vocab_size)
    # Calculate total number of pairs
    total_pairs = sum(length(seq) - 1 for seq in sequences)
    
    # Pre-allocation of arrays
    X = zeros(Int32, vocab_size, total_pairs)
    Y = zeros(Int32, vocab_size, total_pairs)
    
    idx = 1
    for seq in sequences
        for i in 1:(length(seq)-1)
            X[seq[i], idx] = 1
            Y[seq[i+1], idx] = 1
            idx += 1
        end
    end
    
    return X, Y
end

"""
Train the sequential model using mini-batch processing
"""
function train_model!(model, X, Y, epochs, batch_size)
    data = DataLoader((X, Y), batchsize=batch_size, shuffle=true)
    opt = ADAM()
    ps = Flux.params(model)

    losses = zeros(Float32, length(data))
    
    for epoch in 1:epochs
        for (i, (x, y)) in enumerate(data)
            loss, grads = Flux.withgradient(() -> Flux.crossentropy(model(x), y), ps)
            Flux.update!(opt, ps, grads)
            losses[i] = loss
        end
        
        avg_loss = mean(losses)
        
        if epoch % 10 == 0
            println("Epoch $epoch: average loss = $avg_loss")
        end
    end
end

"""
Evaluate the sequential model's accuracy
"""
function evaluate_model(model, X, Y)
    predictions = model(X)
    accuracy = mean(Flux.onecold(predictions) .== Flux.onecold(Y))
    return accuracy
end


"""
Main function to run the entire pipeline
"""
function main()
    vocab_size = 5
    samples = 100
    max_length = 200

    # Generate dataset using ALICE
    initial_probs, transition_probs = generate_probabilities(vocab_size)
    dataset = generate_dataset(initial_probs, transition_probs, samples, max_length) 

    # Prepare data for the model
    X, Y = prepare_data(dataset, vocab_size)
    
    println("X type: ", typeof(X))
    println("Y type: ", typeof(Y))
    println("X shape: ", size(X))
    println("Y shape: ", size(Y))

    # Split data into train and test sets
    split_idx = Int(floor(0.8 * size(X, 2)))
    X_train, Y_train = X[:, 1:split_idx], Y[:, 1:split_idx]
    X_test, Y_test = X[:, split_idx+1:end], Y[:, split_idx+1:end]

    # Create the model
    input_size = vocab_size
    hidden_size = 64
    output_size = vocab_size
    model = GRU(input_size, hidden_size, output_size)

    # model = LSTM(input_size, hidden_size, output_size)
    # model = RNN(input_size, hidden_size, output_size)
    
    # Print model output for a small batch
    small_batch = X[:, 1:5]
    println("Model output shape: ", size(model(small_batch)))

    # Print initial loss
    initial_loss = Flux.crossentropy(model(X), Y)
    println("Initial loss: ", initial_loss)

    # Train the model
    epochs = 100
    batch_size = 32
    train_model!(model, X_train, Y_train, epochs, batch_size)

    # Evaluate the model
    train_accuracy = evaluate_model(model, X_train, Y_train)
    test_accuracy = evaluate_model(model, X_test, Y_test)

    println("Train accuracy: ", train_accuracy)
    println("Test accuracy: ", test_accuracy)
end

# Run the main function
main()