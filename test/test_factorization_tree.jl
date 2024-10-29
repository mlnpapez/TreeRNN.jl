using Test
using Zygote
include("../scripts/sequences/factorization_tree.jl")
include("../scripts/sequences/rnn_model.jl")
include("../scripts/sequences/lstm_model.jl")
include("../scripts/sequences/gru_model.jl")
include("../scripts/sequences/stacked_model.jl")

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
        prob_l2r, data_l2r = compute_probability(tree_l2r, tree_l2r.root; direction=:left_to_right)
        @test size(prob_l2r, 1) == 1
        @test length(tree_l2r.Tp) == 2

        tree_r2l = build_factorization_tree(product_node, model)
        prob_r2l, data_r2l = compute_probability(tree_r2l, tree_r2l.root; direction=:right_to_left)
        @test size(prob_r2l, 1) == 1
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
        prob_l2r, data_l2r = compute_probability(tree, tree.root; direction=:left_to_right)
        @test length(tree.Tp) == 2
        
        # Test right-to-left
        tree = build_factorization_tree(product_node, model)
        prob_r2l, data_r2l = compute_probability(tree, tree.root; direction=:right_to_left)
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
        prob_l2r, data_l2r = compute_probability(tree, tree.root; direction=:left_to_right)
        
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
        prob_r2l, data_r2l = compute_probability(tree_r2l, tree_r2l.root; direction=:right_to_left)
        
        # Verify size is same (should still have all array nodes)
        @test length(tree_r2l.Tp) == 5
    
        # Verify probabilities are meaningful
        @test size(prob_l2r, 1) == 1
        @test size(prob_r2l, 1) == 1
        #@test 0 ≤ prob_l2r[1] ≤ 1
        #@test 0 ≤ prob_r2l[1] ≤ 1
    end
    
    @testset "Complex Hierarchical Structure" begin    
        #=
        # Create multiple product children for bag node
        function create_product_child()
            # Create array nodes
            Tv31 = Mill.ArrayNode(reshape(rand(Float32, array_dim), :, 1))
            Tv32 = Mill.ArrayNode(reshape(rand(Float32, array_dim), :, 1))
            Tv33 = Mill.ArrayNode(reshape(rand(Float32, array_dim), :, 1))
            
            # Create product node
            return product_node = Mill.ProductNode((
                Tv31 = Tv31,
                Tv32 = Tv32,
                Tv33 = Tv33
            ))
            
        end
        
        Tv341 = Mill.ArrayNode(rand(Float32, array_dim, 2))
        Tv342 = Mill.ArrayNode(rand(Float32, array_dim, 2))
        Tv343 = Mill.ArrayNode(rand(Float32, array_dim, 2))

        product_node = Mill.ProductNode((
                Tv341 = Tv341,
                Tv342 = Tv342,
                Tv343 = Tv343
        )) 

        # Create multiple product children
        n_bag_children = 4  # Number of children in bag node
        product_children = [create_product_child() for _ in 1:n_bag_children] =#
        
        # Create leaf array nodes with random data
        array_dim = 5
        Tv1 = Mill.ArrayNode(rand(Float32, array_dim, 4))
        Tv2 = Mill.ArrayNode(rand(Float32, array_dim, 4))
        Tv4 = Mill.ArrayNode(rand(Float32, array_dim, 4))

        n_bag_children = 4  # Number of children/bags in bag node
        #  Create arrays and combine all arrays data for bag node
        all_data = hcat([rand(Float32, array_dim) for _ in 1:(3*n_bag_children)]...)
        data_node = Mill.ArrayNode(all_data)
        
        # Create bag node with proper bags
        # Each product node's data takes up 3 columns
        bags = Mill.AlignedBags([1:3, 4:6, 7:9, 10:12])  # Four bags, each with three columns
        Tv3 = Mill.BagNode(data_node, bags)
        
        # Create root product node
        root = Mill.ProductNode((
        Tv1 = Tv1,    # 3 Observations
        Tv2 = Tv2,    # 3 observations
        Tv3 = Tv3,    # 4 bags
        Tv4 = Tv4     # 3 observations
        ))

        # Create and test tree
        model = GRU(array_dim, 10, 1)
        # gru = GRU(array_dim, 10, 1)

        tree = build_factorization_tree(root, model)
        
        # Test left-to-right traversal
        prob_l2r, data_l2r = compute_probability(tree, tree.root; direction=:left_to_right)
        
        # Verify Tp contents
        # Expected: Tv1, Tv2, Tv4 and 9 array nodes (3 from each product child)
        expected_Tp_length = 3 + (3 * n_bag_children)
        @test length(tree.Tp) == expected_Tp_length
        
        # First two should be Tv1, Tv2
        @test tree.Tp[1] == Tv1
        @test tree.Tp[2] == Tv2
        
        # Test right-to-left traversal
        tree_r2l = build_factorization_tree(root, model)
        prob_r2l, data_r2l = compute_probability(tree_r2l, tree_r2l.root; direction=:right_to_left)
        
        # Should have same number of nodes in Tp
        @test length(tree_r2l.Tp) == expected_Tp_length
        
        # Verify probabilities
        @test size(prob_l2r, 1) == 1
        @test size(prob_r2l, 1) == 1 #
        #@test 0 ≤ prob_l2r[1] ≤ 1
        #@test 0 ≤ prob_r2l[1] ≤ 1

        println("Probability of Tv using dfs in l2r direction: ", prob_l2r[1])
        println("Probability of Tv using dfs in r2l direction: ", prob_r2l[1])

        # Additional tests for bag node independence
        # Each bag child should be independent but conditioned on Tp
        @test length(bags) == n_bag_children
    end

     # Potential additional tests:
    # 1. Order of nodes in Tp (based on traversal direction)
end