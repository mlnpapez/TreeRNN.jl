using Distributions
using Downloads
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

function load_tiny_shakespeare(sequence_length=50, max_sequences=1000)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    local_file = "tiny_shakespeare.txt"
    Downloads.download(url, local_file)

    # Read the file
    text = read(local_file, String)

    # Create vocabulary (character-level)
    vocab = sort(unique(text))
    vocab_size = length(vocab)
    char2id = Dict(char => i for (i, char) in enumerate(vocab))

    # Create short sequences
    sequences = []
    for i in 1:sequence_length:length(text)-sequence_length
        if length(sequences) >= max_sequences
            break
        end
        seq = text[i:i+sequence_length-1]
        push!(sequences, [char2id[char] for char in seq])
    end

    return sequences, vocab_size, char2id
end

"""
Prepare sequence data for the sequential model
"""
function prepare_data(sequences, vocab_size::Int)
    # Calculate total number of pairs
    total_pairs = sum(length(seq) - 1 for seq in sequences)
    
    # Pre-allocation of arrays
    X = zeros(Int32, vocab_size, total_pairs)
    sequence_indices = zeros(Int32, total_pairs)
    
    idx = 1
    for (seq_num, seq) in enumerate(sequences)
        for i in 1:(length(seq)-1)
            X[seq[i], idx] = 1
            sequence_indices[idx] = seq_num
            idx += 1
        end
    end
    
    return X, sequence_indices
end

"""
Helper function to compute log-likelihood of a sequence
"""
function sequence_log_likelihood(model::Union{GRU, LSTM, RNN}, seq_x::AbstractMatrix, initial_state=nothing)::Tuple{Float32, Vector{Float32}}    
    if isnothing(initial_state)
        Flux.reset!(model)
    else
        model.state = initial_state
    end

    log_likelihood = 0f0
    for i in 1:(size(seq_x, 2) - 1)
        probs = model(seq_x[:, i:i])
        true_label = argmax(seq_x[:, i +1])
        log_likelihood += log(probs[true_label])
    end
    return log_likelihood, copy(model.state)
end

"""
Process a batch of sequences
"""
function process_batch(model::Union{GRU, LSTM, RNN}, X::Matrix{Int32}, indices::AbstractVector{<:Integer}, 
    sequence_states::Dict{Int, Vector{Float32}}, sequence_log_likelihoods::Dict{Int, Float32}, sequence_lengths::Dict{Int, Int},  λ::Float32, reg_type::String = "none")::Float32

    batch_log_likelihood = 0f0

    # Add regularization
    reg_term = if reg_type == "none"
        0f0
    elseif reg_type == "L1"
        λ * sum(x -> sum(abs), Flux.params(model))
    elseif reg_type == "L2"
        λ * sum(x -> sum(abs2), Flux.params(model))
    else
        error("Unknown regularization type")
    end
    
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
    
    return -batch_log_likelihood + reg_term  # Negative log-likelihood for minimization
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
function train_model_mle!(model::Union{RNN, GRU, LSTM}, X::Matrix{Int32}, sequence_indices::AbstractVector{<:Integer}, epochs::Int, batch_size::Int, λ, reg_type)
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
                process_batch(model, x, indices, sequence_states, sequence_log_likelihoods, sequence_lengths, λ, reg_type)
            end
            
            Flux.update!(opt, ps, grads)
            epoch_log_likelihood += loss
            total_sequences += length(unique(indices))
        end
        
        avg_log_likelihood, perplexity = compute_epoch_metrics(sequence_log_likelihoods, sequence_lengths)
        
        if epoch % 10 == 0 || epoch == 1
            log_message = "Epoch $epoch: Avg Log-Likelihood = $avg_log_likelihood, Perplexity = $perplexity"
            println(log_message)
            push!(logs, log_message)
        end

        # Clear sequence tracking at the end of each epoch
        empty!(sequence_states)
        empty!(sequence_log_likelihoods)
        empty!(sequence_lengths)
    end

    # return logs
end

"""
Evaluate the MLE-trained sequential model using log-likelihood and perplexity
"""
function evaluate_model_mle(model::Union{RNN, GRU, LSTM}, X::Matrix{Int32}, indices::Vector{Int32})::Dict{String, Float64}
    total_log_likelihood = 0f0
    total_tokens = 0

    for seq_id in unique(indices)
        seq_mask = indices .== seq_id
        seq_x = X[:, seq_mask]
        
        Flux.reset!(model)
        
        seq_log_likelihood = 0f0
        for i in 1:(size(seq_x, 2) - 1)  # We predict up to the second-to-last token
            probs = model(seq_x[:, i:i])
            true_next_token = argmax(seq_x[:, i+1])  # The next token is the "target"
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

"""
Function to generate sequence
"""
function generate_sequence(model::Union{RNN, GRU, LSTM}, start_token::Int, max_length::Int, vocab_size::Int, temperature::Float32)::Vector{Int}
    sequence = [start_token]
    Flux.reset!(model)
    
    for i in 2:max_length
        # Get the last token and convert to one-hot
        last_token = sequence[end]
        input = Flux.onehot(last_token, 1:vocab_size)
        
        # Get model's probs
        probs = vec(model(reshape(input, :, 1)))

        # Apply temperature scaling
        scaled_probs = probs .^ (1/temperature)
        scaled_probs ./= sum(scaled_probs)
        
        # Create a Categorical distribution and sample next token
        next_token = rand(Categorical(scaled_probs))
        
        push!(sequence, next_token)
    end
    
    return sequence
end

"""
Function to generate multiple sequences
"""
function generate_sequences(model::Union{RNN, GRU, LSTM}, num_sequences::Int, max_length::Int, vocab_size::Int, temperature::Float32)
    sequences = []
    for i in 1:num_sequences
        start_token = rand(1:vocab_size)
        seq = generate_sequence(model, start_token, max_length, vocab_size, temperature::Float32)
        push!(sequences, seq)
    end
    return sequences
end

"""
Function to convert number indices back to characters found in text (dataset)
"""
function convert_to_chars(sequences, char2id)
    id2char = Dict(id => char for (char, id) in char2id)
    char_sequences = []
    for seq in sequences
        char_seq = join([id2char[id] for id in seq])
        push!(char_sequences, char_seq)
    end
    return char_sequences
end

"""
Extract the learned transition matrix from a trained RNN, GRU, or LSTM model
"""
function extract_model_distribution(model::Union{RNN, GRU, LSTM}, vocab_size::Int)
    transition_matrix = zeros(Float64, vocab_size, vocab_size)
    
    Flux.reset!(model)
    for i in 1:vocab_size
        input = Flux.onehot(i, 1:vocab_size)
        probs = vec(model(reshape(input, :, 1)))
        transition_matrix[i, :] = probs
    end
    
    # Ensure probabilities are non-negative and sum to 1 for each row
    # softmax
    # všude použít log likelihood nebo probabilities
    #
    transition_matrix = max.(transition_matrix, 0)
    row_sums = sum(transition_matrix, dims=2)
    transition_matrix ./= row_sums
    
    return transition_matrix
end

"""
Extract the empirical transition matrix from the data
"""
function extract_empirical_distribution(X::Matrix{Int32}, vocab_size::Int)
    transition_counts = zeros(Int, vocab_size, vocab_size)
    
    for i in 1:(size(X, 2) - 1)
        from_state = argmax(X[:, i])
        to_state = argmax(X[:, i + 1])
        transition_counts[from_state, to_state] += 1
    end
    
    # Convert counts to probabilities
    transition_matrix = transition_counts ./ sum(transition_counts, dims=2)
    
    # Handle zero-count transitions
    transition_matrix[isnan.(transition_matrix)] .= 0
    transition_matrix = transition_matrix .+ 1e-10  # Add small value to avoid log(0)
    transition_matrix ./= sum(transition_matrix, dims=2)  # Renormalize
    
    return transition_matrix
end

"""
Compare the learned distribution of a model with the true distribution
"""
function compare_distributions(model::Union{RNN, GRU, LSTM}, true_transition_matrix, X::Matrix{Int32})
    vocab_size = size(true_transition_matrix, 1)
    model_transition_matrix = extract_model_distribution(model, vocab_size)
    empirical_transition_matrix = extract_empirical_distribution(X, vocab_size)

    kl_divergences_model_true = Float64[]
    kl_divergences_model_empirical = Float64[]
    kl_divergences_empirical_true = Float64[]

    total_kl_model_true = 0.0
    total_kl_model_empirical = 0.0
    total_kl_empirical_true = 0.0

    for i in 1:vocab_size
        true_probs = true_transition_matrix[i, :]
        model_probs = model_transition_matrix[i, :]
        empirical_probs = empirical_transition_matrix[i, :]

        # Create Categorical distribution from the probs values
        true_dist = Categorical(true_probs)
        model_dist = Categorical(model_probs)
        empirical_dist = Categorical(empirical_probs)

        kl_model_true = kldivergence(model_dist, true_dist)
        kl_model_empirical = kldivergence(model_dist, empirical_dist)
        kl_empirical_true = kldivergence(empirical_dist, true_dist)
        
        push!(kl_divergences_model_true, kl_model_true)
        push!(kl_divergences_model_empirical, kl_model_empirical)
        push!(kl_divergences_empirical_true, kl_empirical_true)
        
        total_kl_model_true += kl_model_true
        total_kl_model_empirical += kl_model_empirical
        total_kl_empirical_true += kl_empirical_true
    end
    
    avg_kl_model_true = total_kl_model_true / vocab_size
    avg_kl_model_empirical = total_kl_model_empirical / vocab_size
    avg_kl_empirical_true = total_kl_empirical_true / vocab_size

    return Dict(
        "kl_divergences_model_true" => round.(kl_divergences_model_true, digits=4),
        "kl_divergences_model_empirical" => round.(kl_divergences_model_empirical, digits=4),
        "kl_divergences_empirical_true" => round.(kl_divergences_empirical_true, digits=4),
        "avg_kl_model_true" => round(avg_kl_model_true, digits=4),
        "avg_kl_model_empirical" => round(avg_kl_model_empirical, digits=4),
        "avg_kl_empirical_true" => round(avg_kl_empirical_true, digits=4),
        "model_transition_matrix" => round.(model_transition_matrix, digits=3),
        "empirical_transition_matrix" => round.(empirical_transition_matrix, digits=3),
        "true_transition_matrix" => round.(true_transition_matrix, digits=3)
    )
end

"""
Function to print comparison results
"""
function print_comparison_results(results, n=10)
    println("\nDetailed comparison of transition probs by character obtained by different means (model, theoretical (true), empirical (sampled data):")
    for i in 1:min(n, size(results["true_transition_matrix"], 1))
        println("\nTransition probs for index $i:")
        println("  True probabilities:      ", results["true_transition_matrix"][i, :])
        println("  Empirical probabilities: ", results["empirical_transition_matrix"][i, :])
        println("  Model probabilities:     ", results["model_transition_matrix"][i, :])
        println("\n  KL divergence (Model vs True):      ", results["kl_divergences_model_true"][i])
        println("  KL divergence (Model vs Empirical): ", results["kl_divergences_model_empirical"][i])
        println("  KL divergence (Empirical vs True):  ", results["kl_divergences_empirical_true"][i])
    end
    
    println("\nAverage KL divergences:")
    println("  Model vs True:      ", results["avg_kl_model_true"])
    println("  Model vs Empirical: ", results["avg_kl_model_empirical"])
    println("  Empirical vs True:  ", results["avg_kl_empirical_true"])
end

function main()
    # Set random seed for reproducibility
    Random.seed!(36)

    # Parameters
    vocab_size = 5
    num_samples = 200
    max_length = 50
    hidden_size = 64
    epochs = 10
    batch_size = 30
    λ = 0.01
    reg_type = "L2"
    temperature = 1f1

    # Generate dataset
    initial_probs, transition_matrix = generate_probabilities(vocab_size)
    dataset = generate_dataset(initial_probs, transition_matrix, num_samples, max_length) #

    #= Load dataset
    dataset, vocab_size, char2id = load_tiny_shakespeare(max_length, num_samples) =#

    # Prepare data
    X, sequence_indices = prepare_data(dataset, vocab_size)

    println("Vocabulary size: ", vocab_size)
    println("Number of sequences: ", length(dataset))
    println("Total characters: ", size(X, 2))
    # display(char2id)

    # Split data into train and test sets
    split_ratio = 0.8
    split_idx = Int(floor(split_ratio * size(X, 2)))
    
    X_train = X[:, 1:split_idx]
    X_test = X[:, split_idx+1:end]
    sequence_indices_train = sequence_indices[1:split_idx]
    sequence_indices_test = sequence_indices[split_idx+1:end]

    # Adjust indices for test set
    unique_train_indices = unique(sequence_indices_train)
    sequence_indices_test = [findfirst(==(i), unique_train_indices) for i in sequence_indices_test]

    # Create models
    models = [
        RNN(vocab_size, hidden_size, vocab_size),
        # GRU(vocab_size, hidden_size, vocab_size),
        # LSTM(vocab_size, hidden_size, vocab_size)
    ]

    for (i, model) in enumerate(models)
        println("\nTraining $(typeof(model))...")
        
        # Train model
        @time logs = train_model_mle!(model, X_train, sequence_indices_train, epochs, batch_size, λ, reg_type)
        
        # Print logs
        # println("Training logs:")
        # foreach(println, logs)

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

        # Compare distributions
        comparison_results = compare_distributions(model, transition_matrix, X_train)
        print_comparison_results(comparison_results)

        # Sequence generation based on model-given probabilities
        generated_sequences = generate_sequences(model, 5, max_length, vocab_size, temperature)
        println("\nGenerated sequences:")
        for (i, seq) in enumerate(generated_sequences)
            println("Sequence $i: ", seq)
        end

        #= char_sequences = convert_to_chars(generated_sequences, char2id)
        for (i, seq) in enumerate(char_sequences)
            println("Sequence $i: ", seq)
        end =#
    end

    #= Compare with true distribution
    total_tokens = sum(length(seq) for seq in dataset)
    true_log_likelihood = sum(log(joint_probability(seq, initial_probs, transition_matrix)) for seq in dataset)
    true_avg_log_likelihood = true_log_likelihood / total_tokens
    true_perplexity = exp(-true_avg_log_likelihood)
    
    println("\nTrue Distribution Metrics:")
    println("Average Log-Likelihood: ", true_avg_log_likelihood)
    println("Perplexity: ", true_perplexity) =#
end

# Run the main function
main()