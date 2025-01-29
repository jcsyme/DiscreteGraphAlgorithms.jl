###   ORDERED LOADING

module DiscreteGraphAlgorithms


using CSV,
      DataFrames,
      Distributed,
      DistributedArrays,
      Graphs,
      LazySets,
      LinearAlgebra,
      QuickHeaps,
      Random,
      SharedArrays,
      SparseArrays,
      Statistics,
      StatsBase,
      XLSX


##  CUSTOM LOADS - TEMPORARY

#=
# load IterativeHeaps
using Pkg
Pkg.develop(path = "/Users/jsyme/Documents/Projects/git_jbus/IterativeHeaps.jl")
using IterativeHeaps

# load GraphDistanceAlgorithms
Pkg.develop(path = "/Users/jsyme/Documents/Projects/git_jbus/GraphDistanceAlgorithms.jl")
using GraphDistanceAlgorithms

# get the graph fragments package
Pkg.develop(path = "/Users/jsyme/Documents/Projects/git_jbus/GraphFragments.jl/")
using GraphFragments
=#
using IterativeHeaps,
      GraphDistanceAlgorithms,
      GraphFragments  



# EXPORTS - start with Structs, then functions; organized by include

export all_dga_prefixes, # exports defined in this file
       GraphWrapper, # add from GraphWrappers
       df_to_graph_wrapper,
       get_distance_matrices,
       graph_to_graph_wrapper,
       read_egl,
       Iterand, # GraphOptimizationIterators
       IteratorParameters,
       OptimizationParameters,
       get_iterand_options,
       get_objective_from_removed_vertices,
       iterate,
       sample_for_swap,
       update_accepted_swap_params!, 
       ## ALGORITHMS - Start with AntColony
       Ant,
       AntColony,
       aco_continuation,
       aco_initialize_colony,
       aco_iterand,
       aco_iterand!,
       aco_log_iteration,
       aco_log_result,
       IterSpace, # Genetic
       Organism,
       OrganismPopulation,
       genetic_continuation,
       genetic_add_measures!,
       genetic_crossover,
       genetic_crossover_random_mapping,
       genetic_evolve,
       genetic_get_elites,
       genetic_get_num_elite,
       genetic_get_population_size,
       genetic_initialize_population,
       genetic_iterand,
       genetic_mutate!,
       genetic_mutation_exchange!,
       genetic_update_percentiles!,
       genetic_iterand!,
       genetic_log_iteration!,
       genetic_log_result!,
       graddesc_continuation, # Gradient Descent
       graddesc_iterand,
       graddesc_iterand!,
       graddesc_log_iteration!,
       graddesc_log_result!,
       gs_continuation, # Greedy Stochastic
       gs_iterand!,
       gs_log_iteration!,
       gs_log_result!,
       gs_iterand,
       pgraddesc_continuation, # Principle Gradient Descent
       # pgraddesc_iterand,
       # pgraddesc_iterand!,
       # pgraddesc_log_iteration!,
       # pgraddesc_log_result!,
       sann_continuation, # SimulatedAnnealing
       sann_iterand!,
       sann_log_iteration!,
       sann_log_result!,
       sann_iterand,
       BoolOrNoth,  # SupportFunctions
       FloatOrNoth,
       VecOrNoth,
       VMS,
       average_degree,
       bound_value,
       build_dict,
       build_dict_inverse,
       check_kwargs,
       check_path,
       check_valid_values!,
       get_crude_percentiles,
       null_func,
       parse_config,
       print_valid_values,
       printnames,
       rand_sample,
       random_unif_from_float_or_noth,
       read_csv,
       rm_edges_by_vertex!,
       str_replace,
       str_with_linewidth,
       try_parse_float,
       tuple_mean,
       write_csv_iterative!,
       zip_cols



##  INCLUDE

dir_load = @__DIR__


include(joinpath(dir_load, "Utilities", "SupportFunctions.jl"))
include(joinpath(dir_load, "Utilities", "GraphWrappers.jl"))
include(joinpath(dir_load, "Algorithms", "GraphOptimizationIterators.jl"))

# include the algorithms themselves
include(joinpath("Algorithms", "AntColony.jl"))
include(joinpath(dir_load, "Algorithms", "Genetic.jl"))
include(joinpath(dir_load, "Algorithms", "GradientDescent.jl"))
include(joinpath(dir_load, "Algorithms", "GreedyStochastic.jl"))
#include(joinpath(dir_load, "Algorithms", "PrincipleGradientDescent.jl"))
include(joinpath(dir_load, "Algorithms", "SimulatedAnnealing.jl"))

# export valid prefixes; init here, and push in each loaded algorithm
all_dga_prefixes = Vector{Symbol}(
       [
              :aco,
              :genetic,
              :graddesc,
              :gs,
              :sann
       ]
)


end