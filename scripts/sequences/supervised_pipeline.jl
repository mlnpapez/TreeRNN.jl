using Flux
using Statistics
using Flux: DataLoader
using Statistics: mean
using LinearAlgebra
using Revise

includet("data_generator.jl")
includet("gru_model.jl")
includet("lstm_model.jl")
includet("rnn_model.jl")

@info "Revise is ready"


"""
Prepare sequence data for the sequential model
"""
 function prepare_data(sequences, vocab_size)
    # Calculate total number of pairs
    total_pairs = sum(length(seq) - 1 for seq in sequences)
    
    # Pre-allocation of arrays
    X = zeros(Int32, vocab_size, total_pairs)
    Y = zeros(Int32, vocab_size, total_pairs)
    sequence_indices = zeros(Int32, total_pairs)
    
    idx = 1
    for (seq_num, seq) in enumerate(sequences)
        for i in 1:(length(seq)-1)
            X[seq[i], idx] = 1
            Y[seq[i+1], idx] = 1
            sequence_indices[idx] = seq_num
            idx += 1
        end
    end
    
    return X, Y, sequence_indices
end 


"""
Train the sequential model using mini-batch processing
"""
function train_model!(model, X, Y, sequence_indices, epochs, batch_size)
    data = DataLoader((X, Y, sequence_indices), batchsize=batch_size, shuffle=false)
    opt = ADAM()
    ps = Flux.params(model)

    sequence_states = Dict{Int, Vector{Float32}}() # Add dict to save hidden states 

    for epoch in 1:epochs
        epoch_loss = 0f0
        total_tokens = 0

        for (x, y, indices) in data
            loss, grads = Flux.withgradient(ps) do
                total_loss = 0f0

                for seq_id in unique(indices)
                    seq_mask = indices .== seq_id
                    seq_x = x[:, seq_mask]
                    seq_y = y[:, seq_mask]

                    if haskey(sequence_states, seq_id)
                        model.state = sequence_states[seq_id]
                    else
                        Flux.reset!(model)
                    end
                    
                    # Sum loss over all tokens in sequence
                    seq_loss = sum(Flux.crossentropy(model(seq_x[:, i:i]), seq_y[:, i:i]) for i in 1:size(seq_x, 2))
                    total_loss += seq_loss # Sum loss over all sequences in batch
                    total_tokens += size(seq_x, 2) # Sum number of tokens over all sequences in batch

                    sequence_states[seq_id] = copy(model.state)
                end

                total_loss / total_tokens  # Average loss per token
            end
            Flux.update!(opt, ps, grads)
            epoch_loss += loss
        end

        empty!(sequence_states)

        avg_loss = epoch_loss / length(data)
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
Implementing random baseline model that makes random predictions
"""
function random_baseline(X, vocab_size)
    return rand(Float32, vocab_size, size(X, 2))
end

"""
Main function to run the entire pipeline
"""
function main()
    vocab_size = 3
    samples = 400
    max_length = 10

    # Generate dataset using ALICE
    initial_probs, transition_probs = generate_probabilities(vocab_size)

    # Print the true distributions
    println("Initial Probabilities:")
    println(initial_probs)
    println("Sum of initial probs: ", sum(initial_probs))
    
    println("\nTransition Matrix:")
    display(transition_probs)  # Using display for better formatting of matrices
    println()

    # Check row sums of transition matrix
    row_sums = sum(transition_probs, dims=2)
    println("Row sums of transition matrix:")
    display(row_sums)
    println()

    dataset = generate_dataset(initial_probs, transition_probs, samples, max_length) 

    X, Y, sequence_indices = prepare_data(dataset, vocab_size)
    
    println("X type: ", typeof(X))
    println("Y type: ", typeof(Y))
    println("sequence_indices type: ", typeof(sequence_indices))
    println("X shape: ", size(X))
    println("Y shape: ", size(Y))
    println("sequence_indices shape: ", size(sequence_indices))

    # Split data into train and test sets
    split_idx = Int(floor(0.8 * size(X, 2)))
    X_train, Y_train, sequence_indices_train = X[:, 1:split_idx], Y[:, 1:split_idx], sequence_indices[1:split_idx]
    X_test, Y_test = X[:, split_idx+1:end], Y[:, split_idx+1:end]

    # Create the model
    input_size = vocab_size
    hidden_size = 64
    output_size = vocab_size

    # model = GRU(input_size, hidden_size, output_size)
    model = LSTM(input_size, hidden_size, output_size)
    # model = RNN(input_size, hidden_size, output_size)
    
    # Define batch size and number of epochs
    epochs = 350
    batch_size = 30

    # Check individual batches
    # data = DataLoader((X, Y, sequence_indices), batchsize=batch_size, shuffle=false)
    #= for (i, (X_batch, Y_batch, indices_batch)) in enumerate(data)
        if i > 2
            break
        end
        println("Batch $i:")
        println("X:")
        display(X_batch)
        println("\nY:")
        display(Y_batch)
        println("\nSequence indices:")
        println(indices_batch)
        println("\n----------------------\n")
    end =#

    # Print model output for a small batch
    small_batch = X[:, 1:5]
    println("Model output shape: ", size(model(small_batch)))

    # Print initial loss
    initial_loss = Flux.crossentropy(model(X), Y)
    println("Initial loss: ", initial_loss)

    # Train the model
    train_model!(model, X_train, Y_train, sequence_indices_train, epochs, batch_size)
    # train_model!(model, X, Y, sequence_indices, epochs, batch_size)

    # Evaluate the model
    train_accuracy = evaluate_model(model, X_train, Y_train)
    test_accuracy = evaluate_model(model, X_test, Y_test)

    println("Train accuracy: ", train_accuracy)
    println("Test accuracy: ", test_accuracy)

    # Evaluate random baseline
    random_predictions = random_baseline(X_test, vocab_size)
    random_accuracy = mean(Flux.onecold(random_predictions) .== Flux.onecold(Y_test))
    println("Random Baseline Accuracy: ", random_accuracy)
end

# Run the main function
# main()