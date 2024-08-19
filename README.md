# TreeRNN

TreeRNN is a small experimental (unregistered) package implementing variants of recurrent neural networks (RNNs) for tree-structured data. The input tree-structured data are assumed to be in the JSON format. The package was used to compute the tree RNNs in the following paper: *Papež M, Rektoris M, Šmídl V, Pevný T. [Sum-Product-Set Networks: Deep Tractable Models for Tree-Structured Graphs](https://openreview.net/pdf?id=mF3cTns4pe). In The 12th International Conference on Learning Representations (ICLR2024).*

The package is built on top of [JSONGrinder.jl](https://github.com/CTUAvastLab/JsonGrinder.jl) and [Mill.jl](https://github.com/CTUAvastLab/Mill.jl).

To use the package, execute the following steps.

1. Clone this repository.
 ```
 git clone https://github.com/mlnpapez/TreeRNN.jl TreeRNN.jl
 ```
2. Go to the TreeRNN.jl repository.
 ```
 cd TreeRNN.jl
 ```
3. Open the Julia 1.10.2 console and write:
 ```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
 ```
4. Use the package:
 ``` julia
using TreeRNN
 ```

The repository contains 12 datasets in JSON format (listed in the table below). All these datasets were downloaded from [CTU Prague Relational Learning Repository](https://relational-data.org/) and converted into the JSON format.

| Dataset         | # of instances | # of classes |
| --------------- | -------------- | ------------ |
| mutagenesis     | 188            | 2            |
| genes           | 862            | 15           |
| cora            | 2708           | 7            |
| citeseer        | 3312           | 6            |
| webkp           | 877            | 5            |
| world           | 239            | 7            |
| chess           | 295            | 3            |
| uw_cse          | 278            | 4            |
| hepatitis       | 500            | 2            |
| ftp             | 30000          | 3            |
| ptc             | 343            | 2            |
| dallas          | 219            | 7            |

The repository currently contains three models (listed in the table below).

| Model    | Description                            | Reference |
| -------- | -------------------------------------- | --------- |
| TreeGRU  | tree-structured gated reccurent unit   | https://arxiv.org/abs/1610.02806 |
| TreeLSTM | tree-structured long short-term memory | https://arxiv.org/abs/1503.00075 |
| TreeMLP  | tree-structured multilayer perceptron  | - |


Execute the following steps to train one of these models on a selected dataset.

1. Go to the TreeRNN.jl repository.
 ```
 cd TreeRNN.jl
 ```
2. Open the Julia 1.10.2 console and write:
 ```julia
using Pkg
Pkg.activate("scripts")
Pkg.instantiate()
Pkg.develop(path=".")
 ```
3. Use the package:
 ``` julia
include("scripts/train.jl")
 ```
