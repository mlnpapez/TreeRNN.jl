using Flux
using Statistics
using Flux: DataLoader
using LinearAlgebra
using Revise

includet("data_generator.jl")
includet("rnn_model.jl")
includet("gru_model.jl") 
includet("lstm_model.jl") 

@info "MLE is up and running"

"""
Prepare sequence data for the sequential model
"""
function prepare_data(sequences::Vector{Vector{T}}, vocab_size::Int) where T <: Integer
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
# Helper function to compute log-likelihood of a sequence
"""
function sequence_log_likelihood(model::Union{GRU, LSTM, RNN}, seq_x::AbstractMatrix, initial_state=nothing)
    if isnothing(initial_state)
        Flux.reset!(model)
    else
        model.state = initial_state
    end

    log_likelihood = 0f0
    for i in 1:size(seq_x, 2)
        probs = model(seq_x[:, i:i])
        true_label = argmax(seq_x[:, i])
        log_likelihood += log(probs[true_label])
    end
    return log_likelihood, copy(model.state)
end

"""
Helper function to compute perplexity
"""
function calculate_perplexity(log_likelihood::Int, sequence_length::Int)
    return exp(-log_likelihood / sequence_length)
end

# Process a batch of sequences
function process_batch(model::Union{GRU, LSTM, RNN}, X::Matrix{Int64}, indices::Vector{Int64}, sequence_states::Dict{Int, Vector{Float32}}, sequence_log_likelihoods::Dict{Int, Float32}, sequence_lengths::Dict{Int, Int})
    batch_log_likelihood = 0f0
    
    for seq_id in unique(indices)
        seq_mask = indices .== seq_id
        seq_x = X[:, seq_mask]
        
        initial_state = if haskey(sequence_states, seq_id)
            sequence_states[seq_id]
        else
            zeros(Float32, size(model.state))
        end
        
        seq_log_likelihood, final_state = sequence_log_likelihood(model, seq_x, initial_state)
        
        
        # Update dictionaries outside of the gradient computation
        sequence_states[seq_id] = final_state
        sequence_log_likelihoods[seq_id] = get(sequence_log_likelihoods, seq_id, 0f0) + seq_log_likelihood
        sequence_lengths[seq_id] = get(sequence_lengths, seq_id, 0) + size(seq_x, 2)

        batch_log_likelihood += seq_log_likelihood
    end
    
    return -batch_log_likelihood  # Negative log-likelihood for minimization
end

"""
Compute average log-likelihood and perplexity for an epoch
"""
function compute_epoch_metrics(sequence_log_likelihoods::Dict{Int, Float32}, sequence_lengths::Dict{Int, Int})
    total_log_likelihood = sum(values(sequence_log_likelihoods))
    total_tokens = sum(values(sequence_lengths))
    avg_log_likelihood_per_token = total_log_likelihood / total_tokens
    perplexity = exp(-avg_log_likelihood_per_token)
    return avg_log_likelihood_per_token, perplexity
end


"""
Train the sequential model using MLE and mini-batch processing
"""
function train_model_mle!(model::Union{RNN, GRU, LSTM}, X::Matrix{Int32}, sequence_indices::Vector{Int64}, epochs::Int, batch_size::Int)
    data = DataLoader((X, sequence_indices), batchsize=batch_size, shuffle=false)
    opt = ADAM()
    ps = Flux.params(model)

    sequence_states = Dict{Int, Vector{Float32}}()
    sequence_log_likelihoods = Dict{Int, Float32}()
    sequence_lengths = Dict{Int, Int}()

    logs = String[]

    for epoch in 1:epochs
        epoch_log_likelihood = 0f0
        total_sequences = 0
        
        for (x, indices) in data
            loss, grads = Flux.withgradient(ps) do
                process_batch(model, x, indices, sequence_states, sequence_log_likelihoods, sequence_lengths)
            end
            
            Flux.update!(opt, ps, grads)
            epoch_log_likelihood += loss
            total_sequences += length(unique(indices))
        end
        
        avg_log_likelihood, perplexity = compute_epoch_metrics(sequence_log_likelihoods, sequence_lengths)
        
        if epoch % 10 == 0 || epoch == 1
            log_message = "Epoch $epoch: Avg Log-Likelihood = $avg_log_likelihood, Perplexity = $perplexity"
            push!(logs, log_message)
        end

        # Clear sequence tracking at the end of each epoch
        empty!(sequence_states)
        empty!(sequence_log_likelihoods)
        empty!(sequence_lengths)
    end

    return logs
end

"""
Evaluate the MLE-trained sequential model using log-likelihood and perplexity
"""
function evaluate_model_mle(model::Union{RNN, GRU, LSTM}, X::Matrix{Int32}, sequence_indices::Vector{Int64})
    total_log_likelihood = 0f0
    total_tokens = 0

    # Group the data by sequences
    unique_sequences = unique(sequence_indices)
    
    for seq_id in unique_sequences
        seq_mask = sequence_indices .== seq_id
        seq_x = X[:, seq_mask]
        
        Flux.reset!(model)
        
        seq_log_likelihood = 0f0
        for t in 1:(size(seq_x, 2) - 1)  # We predict up to the second-to-last token
            probs = model(seq_x[:, t:t])
            true_next_token = argmax(seq_x[:, t+1])  # The next token is the "target"
            seq_log_likelihood += log(probs[true_next_token])
        end
        
        total_log_likelihood += seq_log_likelihood
        total_tokens += size(seq_x, 2) - 1  # We have one less prediction than tokens
    end

    # Compute metrics based on the comparison
    avg_log_likelihood = total_log_likelihood / total_tokens
    perplexity = exp(-avg_log_likelihood)

    return Dict(
        "avg_log_likelihood" => avg_log_likelihood,
        "perplexity" => perplexity
    )
end

function main()
    # Set random seed for reproducibility
    Random.seed!(36)

    # Parameters
    vocab_size = 3
    num_samples = 100
    max_length = 20
    hidden_size = 32
    epochs = 100
    batch_size = 30

    # Generate dataset
    initial_probs, transition_matrix = generate_probabilities(vocab_size)
    dataset = generate_dataset(initial_probs, transition_matrix, num_samples, max_length)

    # Prepare data
    X, Y, sequence_indices = prepare_data(dataset, vocab_size)

    # Split data into train and test sets
    split_ratio = 0.8
    split_idx = Int(floor(split_ratio * size(X, 2)))
    
    X_train, Y_train = X[:, 1:split_idx], Y[:, 1:split_idx]
    X_test, Y_test = X[:, split_idx+1:end], Y[:, split_idx+1:end]
    sequence_indices_train = sequence_indices[1:split_idx]
    sequence_indices_test = sequence_indices[split_idx+1:end] .- split_idx  # Adjust indices for test set

    # Create models
    models = [
        RNN(vocab_size, hidden_size, vocab_size),
        GRU(vocab_size, hidden_size, vocab_size),
        # LSTM(vocab_size, hidden_size, vocab_size)
    ]

    for (i, model) in enumerate(models)
        println("\nTraining $(typeof(model))...")
        
        # Train model
        @time logs = train_model_mle!(model, X_train, sequence_indices_train, epochs, batch_size)
        
        # Print logs
        println("Training logs:")
        foreach(println, logs)

        # Evaluate model
        train_results = evaluate_model_mle(model, X_train, sequence_indices_train)
        test_results = evaluate_model_mle(model, X_test, sequence_indices_test)

        println("\nTraining Results:")
        for (metric, value) in train_results
            println("$metric: $value")
        end

        println("\nTest Results:")
        for (metric, value) in test_results
            println("$metric: $value")
        end
    end

    # Compare with true distribution
    total_tokens = sum(length(seq) for seq in dataset)
    true_log_likelihood = sum(log(joint_probability(seq, initial_probs, transition_matrix)) for seq in dataset)
    true_avg_log_likelihood = true_log_likelihood / total_tokens
    true_perplexity = exp(-true_avg_log_likelihood)
    
    println("\nTrue Distribution Metrics:")
    println("Average Log-Likelihood: ", true_avg_log_likelihood)
    println("Perplexity: ", true_perplexity)
end

# Run the main function
main()