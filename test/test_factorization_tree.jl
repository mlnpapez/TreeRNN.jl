using Test
using Zygote
using Distributions

#include("../scripts/sequences/factorization_tree.jl")
include("../scripts/sequences/factor_tree_with_print.jl")
include("../scripts/models/rnn_model.jl")
include("../scripts/models/lstm_model.jl")
include("../scripts/models/gru_model.jl")
include("../scripts/models/stacked_model.jl")
include("../scripts/models/input_adapter.jl")

@testset "FactorizationTree Tests" begin
    # Test ConditionSet construction
    @testset "ConditionSet" begin
        conditions = ConditionSet()
        @test isempty(conditions.Tp)
        @test isempty(conditions.T_siblings)
        @test isempty(conditions.path)
    end

    # Test probability computation for array node
    @testset "Array Node Probability" begin
        array_data = reshape(rand(Float32, 5), :, 1)
        array_node = Mill.ArrayNode(array_data)
        model = RNN(5, 10, 1)
        
        tree = build_factorization_tree(array_node, model)
        prob, data = compute_probability(tree, tree.root)  # Use default direction
        println(prob)
        @test size(prob, 1) == 1
        @test data == array_data
        @test length(tree.Tp) == 1
    end

    # Test probability computation for bag node
    @testset "Bag Node Probability" begin
        # Create array nodes
        array_data1 = reshape(rand(Float32, 5), :, 1)
        array_data2 = reshape(rand(Float32, 5), :, 1)
        array_node1 = Mill.ArrayNode(array_data1)
        array_node2 = Mill.ArrayNode(array_data2)
        
        # Create bag node directly with array nodes
        bag_node = Mill.BagNode(array_node1, Mill.AlignedBags([1:1]))  # One bag, one array node

        
        model = RNN(5, 10, 1)
        tree = build_factorization_tree(bag_node, model)
        
        prob, data = compute_probability(tree, tree.root)
        @test size(prob, 1) == 1
        @test length(tree.Tp) == 1 
    end
    
    @testset "Product Node Probability" begin
        array_data1 = reshape(rand(Float32, 5), :, 1)
        array_data2 = reshape(rand(Float32, 5), :, 1)
        array_node1 = Mill.ArrayNode(array_data1)
        array_node2 = Mill.ArrayNode(array_data2)
        
        product_node = Mill.ProductNode((
            a = array_node1,
            b = array_node2
        ))
        
        input_size = 5
        model = RNN(input_size, 10, 1)
        
        # Test both directions
        tree_l2r = build_factorization_tree(product_node, model)
        log_prob_l2r, data_l2r = compute_probability(tree_l2r, tree_l2r.root; direction=:left_to_right)
        @test size(log_prob_l2r, 1) == 1
        @test length(tree_l2r.Tp) == 2

        tree_r2l = build_factorization_tree(product_node, model)
        log_prob_r2l, data_r2l = compute_probability(tree_r2l, tree_r2l.root; direction=:right_to_left)
        @test size(log_prob_r2l, 1) == 1
        @test length(tree_r2l.Tp) == 2
    end

    @testset "Direction-aware Probability Computation" begin
        array_data1 = reshape(rand(Float32, 5), :, 1)
        array_data2 = reshape(rand(Float32, 5), :, 1)
        array_node1 = Mill.ArrayNode(array_data1)
        array_node2 = Mill.ArrayNode(array_data2)
        
        product_node = Mill.ProductNode((
            a = array_node1,
            b = array_node2
        ))
        
        model = RNN(5, 10, 1)
        
        # Test left-to-right
        tree = build_factorization_tree(product_node, model)
        log_prob_l2r, data_l2r = compute_probability(tree, tree.root; direction=:left_to_right)
        @test length(tree.Tp) == 2
        
        # Test right-to-left
        tree = build_factorization_tree(product_node, model)
        log_prob_r2l, data_r2l = compute_probability(tree, tree.root; direction=:right_to_left)
        @test length(tree.Tp) == 2
    end

    @testset "Simple Hierarchical Structure" begin
        # Create leaf array nodes with random data
        array_dim = 5
        Tv1 = Mill.ArrayNode(reshape(rand(Float32, array_dim), :, 1))
        Tv2 = Mill.ArrayNode(reshape(rand(Float32, array_dim), :, 1))
        
        # Create array nodes for Tv3,1's children
        Tv31_1 = Mill.ArrayNode(reshape(rand(Float32, array_dim), :, 1))
        Tv32_1 = Mill.ArrayNode(reshape(rand(Float32, array_dim), :, 1))
        Tv33_1 = Mill.ArrayNode(reshape(rand(Float32, array_dim), :, 1))
        
        # Create product node Tv3,1
        Tv3_1 = Mill.ProductNode((
            Tv31 = Tv31_1,
            Tv32 = Tv32_1,
            Tv33 = Tv33_1
        ))
        
         # Create bag node Tv3 with single product child
        # We just pass the product node as data
        Tv3 = Mill.BagNode(Tv3_1, Mill.AlignedBags([1:1]))  # One bag with one product node
        
        # Create root product node
        root = Mill.ProductNode((
            Tv1 = Tv1,
            Tv2 = Tv2,
            Tv3 = Tv3
        ))
    
        # Create and test tree
        model = RNN(array_dim, 10, 1)
        tree = build_factorization_tree(root, model)
        
        # Test left-to-right traversal
        log_prob_l2r, data_l2r = compute_probability(tree, tree.root; direction=:left_to_right)
        
        # Verify Tp contents and order
        # For left-to-right traversal:
        # Tp should be [Tv1, Tv2, Tv31_1, Tv32_1, Tv33_1]
        @test length(tree.Tp) == 5
        @test tree.Tp[1] == Tv1  # First in DFS
        @test tree.Tp[2] == Tv2  # Second in DFS
        
        # The last three should be the array nodes from Tv3,1
        last_three = tree.Tp[3:5]
        @test Tv31_1 in last_three
        @test Tv32_1 in last_three
        @test Tv33_1 in last_three
        
        # Test right-to-left traversal
        tree_r2l = build_factorization_tree(root, model)
        log_prob_r2l, data_r2l = compute_probability(tree_r2l, tree_r2l.root; direction=:right_to_left)
        
        # Verify size is same (should still have all array nodes)
        @test length(tree_r2l.Tp) == 5
    
        # Verify probabilities are meaningful
        @test size(log_prob_l2r, 1) == 1
        @test size(log_prob_r2l, 1) == 1
        @test log_prob_l2r[1] ≤ 0
        @test log_prob_r2l[1] ≤ 0
    end

    @testset "Handling different inputs" begin
        base_model = RNN(5, 10, 1)
        adapted_model = InputAdapter(base_model, 5)


        array_data1 = reshape(rand(Float32, 5), :, 1)
        array_data2 = reshape(rand(Float32, 5), :, 1)
        array_node1 = Mill.ArrayNode(array_data1)
        array_node2 = Mill.ArrayNode(array_data2)
        
        sample = Mill.ProductNode((
            a = array_node1,
            b = array_node2
        ))

        # Create tree - automatically uses model's input size
        tree = build_factorization_tree(sample, adapted_model)
        prob, data = compute_probability(tree, tree.root; direction=:left_to_right)

        @testset "Variable size inputs" begin
            # Test different input sizes
            x = rand(Float32, rand(1:12), 1)
            
            output = tree.model(x)

            output_dim = size(output, 1)
            batch_dim = size(output, 2)
            
            @test output_dim == 1  # Check output dimension
            @test batch_dim == 1  # Check batch dimension
        end
    
        @testset "Vector vs Matrix inputs" begin
            # Vector input
            vector_input = rand(Float32, 7)
            output_v = tree.model(vector_input)
            @test length(output_v) == 1  # Should output scalar
            
            # Matrix input
            matrix_input = rand(Float32, 7, 3)
            output_m = tree.model(matrix_input)
            @test size(output_m, 1) == 1  # Check output dimension
            @test size(output_m, 2) == 3  # Check batch dimension preserved
        end
    
        @testset "Edge cases" begin
            # Single dimension
            x1 = rand(Float32, 1, 1)
            @test_nowarn tree.model(x1)
            
            # Large dimension
            x2 = rand(Float32, 100, 1)
            @test_nowarn tree.model(x2)
            
            # Multiple batches
            x3 = rand(Float32, 7, 10)
            output = tree.model(x3)
            @test size(output, 2) == 10
        end
    end
    
    @testset "Complex Hierarchical Structure" begin    
        
        # Helper function to generate data from a distribution
        function generate_from_dist(dist, rows, cols)
            return Float32.(rand(dist, rows, cols))
        end

        # Creating a synthetic MUTAG sample
        function create_test_sample()
            # Fixed number of observations at each level
            n_atoms = 3  # Instead of 26 as in real mutag sample
            n_bonds = 6  # Instead of 56 (total bonds across all atoms)
            
            # First create all bond products (they must have same n_bonds observations)
            function create_bond_products()
                # Create single product node with n_bonds observations
                return Mill.ProductNode((
                    element = Mill.ArrayNode(generate_from_dist(Normal(10.0, 3.0), 8, n_bonds)),
                    type_bond = Mill.ArrayNode(generate_from_dist(Normal(0.0, 1.0), 7, n_bonds)),
                    type_atom = Mill.ArrayNode(generate_from_dist(Normal(5.0, 2.0), 12, n_bonds)),
                    charge = Mill.ArrayNode(generate_from_dist(Normal(0.0, 1.5), 1, n_bonds))
                ))
            end
            
            # Create atoms (all must have n_atoms observations)
            # Each atom's bonds must be divided from total n_bonds
            bond_products = create_bond_products()
            
            # Create bag nodes for bonds - divide n_bonds among atoms
            # For n_atoms=3, n_bonds=4: [1:1, 2:3, 4:6] means:
            # - first atom has 1 bond
            # - second atom has 2 bonds
            # - third atom has 3 bonds
            bonds_bag = Mill.BagNode(bond_products, Mill.AlignedBags([1:1, 2:3, 4:6]))
            
            # Create main product node with n_atoms observations
            atoms_product = Mill.ProductNode((
                element = Mill.ArrayNode(generate_from_dist(Normal(10.0, 3.0), 8, n_atoms)),
                bonds = bonds_bag,  # This has n_atoms observations because of AlignedBags
                type_atom = Mill.ArrayNode(generate_from_dist(Normal(5.0, 2.0), 37, n_atoms)),
                charge = Mill.ArrayNode(generate_from_dist(Normal(0.0, 1.5), 1, n_atoms))
            ))
            
            # Create main atoms bag node
            atoms = Mill.BagNode(atoms_product, Mill.AlignedBags([1:n_atoms]))
            
            # Random size for top-level arrays
            random_size = rand(5:8)
            
            # Create root product node (must have 1 observation like atoms bag)
            root = Mill.ProductNode((
                lumo = Mill.ArrayNode(generate_from_dist(Normal(-5.0, 2.0), random_size, 1)),
                inda = Mill.ArrayNode(generate_from_dist(LogNormal(2.0, 0.5), random_size, 1)),
                logp = Mill.ArrayNode(generate_from_dist(Normal(-5.0, 2.0), random_size, 1)),
                ind1 = Mill.ArrayNode(rand(Float32, random_size, 1)),
                atoms = atoms  # 1 observation
            ))
            
            return root
        end
        
        sample1 = create_test_sample()
        sample2 = create_test_sample()

        # Create model with input adapter (to handle data vectors with diffrent input dimension)
        base_model = GRU(5, 10, 10)
        adapted_model = InputAdapter(base_model, 5)

        @testset "DFS Traversal From Left to Right" begin 
            println("Synthetic MUTAG sample (Mill structure): \n")
            printtree(sample1)

            # Create tree - automatically uses model's input size
            tree_l2r = build_factorization_tree(sample1, adapted_model)

            # Test left-to-right traversal
            log_prob_l2r, data_l2r = compute_probability(tree_l2r, tree_l2r.root; direction=:left_to_right)
            
            # Verify Tp contents (number of leaves/array nodes)
            expected_Tp_length = 37 # 4 top arrays + (3 atoms products obs * 3 array nodes types) + (6 bonds product obs * 4 array nodes type)
            @test length(tree_l2r.Tp) == expected_Tp_length
            
            # First two should be lumo and inda
            @test tree_l2r.Tp[1] == sample1[:lumo]
            @test tree_l2r.Tp[2] == sample1[:inda]

            # Verify probabilities
            @test size(log_prob_l2r, 1) == 1
            @test log_prob_l2r[1] ≤ 0

            println("\nProbability of observing values using dfs in l2r direction: ", log_prob_l2r[1])
        end
        
        @testset "DFS Traversal From Left to Right" begin
            println("Synthetic MUTAG sample (Mill structure): \n")
            printtree(sample1)

            # Test right-to-left traversal
            tree_r2l = build_factorization_tree(sample1, adapted_model)
            log_prob_r2l, data_r2l = compute_probability(tree_r2l, tree_r2l.root; direction=:right_to_left)
            
            # Should have same number of nodes in Tp
            # Verify Tp contents (number of leaves/array nodes)
            expected_Tp_length = 37 # 4 top arrays + (3 atoms products obs * 3 array nodes types) + (6 bonds product obs * 4 array nodes type)
            @test length(tree_r2l.Tp) == expected_Tp_length
            
            # Verify probabilities
            @test size(log_prob_r2l, 1) == 1
            @test log_prob_r2l[1] ≤ 0

            println("Probability of observing values using dfs in r2l direction: ", log_prob_r2l[1], "\n")
        end
    end
     # Potential additional tests:
    # 1. Order of nodes in Tp (based on traversal direction)
end