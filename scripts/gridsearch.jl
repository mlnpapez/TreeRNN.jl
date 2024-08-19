#!/usr/bin/env sh
#SBATCH --array=1-60
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --exclude=n33
#SBATCH --out=/home/papezmil/logs/%x-%j.out
#=
srun julia tree_structures/gridsearch.jl --n $SLURM_ARRAY_TASK_ID --m $1
exit
# =#
using DrWatson
@quickactivate
using JSON3
using ArgParse
using DataFrames
using JsonGrinder

include("utils/utils.jl")
include("models/tree_mlp.jl")
include("models/tree_gru.jl")
include("models/tree_lstm.jl")


function commands()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--n"; arg_type = Int; default=1);
        ("--m"; arg_type = Int; default=1);
    end
    parse_args(s)
end

function slurm_rnns_acc()
    @unpack n, m = commands()
    dataset = datasets[m]
    ctype, no, ni, nepoc, bsize, ssize, seed_split, seed_init = collect(Iterators.product(
        [:tree_lstm], # run: done: tree_mlp tree_gru
        [10, 20, 30, 40],
        [10, 20, 30, 40],
        [200],
        [10],
        [1e-1, 1e-2, 1e-3],
        [1],
        collect(1:5)))[n]
    data = read("$(dirdata)/$(dataset.name).json", String)
    data = JSON3.read(data)
    x, y = data.x, data.y

    x = reduce(catobs, suggestextractor(schema(x), (; scalar_extractors = default_scalar_extractor())).(x))
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = split_data(x, y, seed_split)

    Random.seed!(seed_init)
    ne = length(unique(y))
    m = Dense(no, ne) ∘ cell_builder(ctype)(Float32, no, ni, x_trn)

    config_exp = (; seed_split, seed_init, dirdata, dataset=dataset.name, ctype, no, ni, nepoc, bsize, ssize)
    config_wat = (suffix="jld2", sort=false, ignores=(:dirdata, ), verbose=true, force=true)

    gd!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, Flux.Adam(ssize), nepoc, bsize, ne, config_exp, config_wat, "accuracy")
end
function slurm_hmil_acc()
    @unpack n, m = commands()
    dataset = datasets[m]
    ctype, no, nepoc, bsize, ssize, seed_split, seed_init = collect(Iterators.product(
        [:hmil],
        [10, 20, 30, 40],
        [200],
        [10],
        [1e-1, 1e-2, 1e-3],
        [1],
        collect(1:5)))[n]
    data = read("$(dirdata)/$(dataset.name).json", String)
    data = JSON3.read(data)
    x, y = data.x, data.y

    x = reduce(catobs, suggestextractor(schema(x)).(x))
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = split_data(x, y, seed_split)

    Random.seed!(seed_init)
    ne = length(unique(y))
    m = Dense(no, ne) ∘ Mill.reflectinmodel(x, d->Dense(d, no); all_imputing=true)

    config_exp = (; seed_split, seed_init, dirdata, dataset=dataset.name, ctype, no, nepoc, bsize, ssize)
    config_wat = (suffix="jld2", sort=false, ignores=(:dirdata, ), verbose=true, force=true)

    gd!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, Flux.Adam(ssize), nepoc, bsize, ne, config_exp, config_wat, "accuracy")
end
function slurm_rnns_mis()
    @unpack n, m = commands()
    dataset = datasets[m]

    df = collect_results("data/tree_structures/accuracy"; rinclude=[Regex(dataset.name)])
    df = groupby(df, [:dataset, :ctype, :no, :ni, :nepoc, :bsize, :ssize])
    df = combine(df, :acc_val=>mean, :acc_tst=>mean, renamecols=false)
	df = combine(df->df[argmax(df[!, :acc_val]), :], groupby(df, [:dataset, :ctype]))

    ctype, mrate, seed_split, seed_init = collect(Iterators.product(
        [:tree_mlp, :tree_gru, :tree_lstm],
        [1e-4],
        [1],
        collect(1:10)))[n]
    @unpack no, ni, nepoc, bsize, ssize = copy(df[df.ctype .== ctype, :][1, :])

    data = read("$(dirdata)/$(dataset.name).json", String)
    data = JSON3.read(data)
    x, y = data.x, data.y

    x = reduce(catobs, suggestextractor(schema(x), (; scalar_extractors = default_scalar_extractor())).(x))
    x = make_missing(x, mrate)
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = split_data(x, y, seed_split)

    Random.seed!(seed_init)
    ne = length(unique(y))
    m = Dense(no, ne) ∘ cell_builder(ctype)(Float32, no, ni, x_trn)

    config_exp = (; seed_split, seed_init, dirdata, dataset=dataset.name, ctype, no, ni, nepoc, bsize, ssize, mrate)
    config_wat = (suffix="jld2", sort=false, ignores=(:dirdata, ), verbose=true, force=true)

    gd!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, Flux.Adam(ssize), nepoc, bsize, ne, config_exp, config_wat, "missing")
end
function slurm_hmil_mis()
    @unpack n, m = commands()
    dataset = datasets[m]

    df = collect_results("data/tree_structures/accuracy"; rinclude=[Regex(dataset.name)])
    df = groupby(df, [:dataset, :ctype, :no, :nepoc, :bsize, :ssize])
    df = combine(df, :acc_val=>mean, :acc_tst=>mean, renamecols=false)
	df = combine(df->df[argmax(df[!, :acc_val]), :], groupby(df, [:dataset, :ctype]))

    ctype, mrate, seed_split, seed_init = collect(Iterators.product(
        [:hmil],
        [1e-4],
        [1],
        collect(1:10)))[n]
    @unpack no, nepoc, bsize, ssize = copy(df[df.ctype .== ctype, :][1, :])

    data = read("$(dirdata)/$(dataset.name).json", String)
    data = JSON3.read(data)
    x, y = data.x, data.y

    x = reduce(catobs, suggestextractor(schema(x)).(x))
    x = make_missing(x, mrate)
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = split_data(x, y, seed_split)

    Random.seed!(seed_init)
    ne = length(unique(y))
    m = Dense(no, ne) ∘ Mill.reflectinmodel(x, d->Dense(d, no); all_imputing=true)

    config_exp = (; seed_split, seed_init, dirdata, dataset=dataset.name, ctype, no, nepoc, bsize, ssize, mrate)
    config_wat = (suffix="jld2", sort=false, ignores=(:dirdata, ), verbose=true, force=true)

    gd!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, Flux.Adam(ssize), nepoc, bsize, ne, config_exp, config_wat, "missing")
end

# slurm_rnns_acc()
slurm_hmil_acc()
# slurm_rnns_mis()
# slurm_hmil_mis()

nothing
