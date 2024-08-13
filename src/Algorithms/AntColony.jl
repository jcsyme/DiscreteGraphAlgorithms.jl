
"""
Implement ant colony optimization for combinatorial optimization on a graph.


## Key Functions

`aco_continuation`
`aco_iterand!`
`aco_log_iteration!`
`aco_log_result!`


##  Key Structures

`aco_iterand`: the GraphOptimizationIterators.Iterand object used inside of 
    heuristic optimization to implement ant colony optimization. This object 
    defines steps to take inside each iteration, ideal state swaps, stopping 
    conditions, acceptance thresholds, and more.


# setup to allow for local module load
dir_load = @__DIR__
!(dir_load in LOAD_PATH) ? push!(LOAD_PATH, dir_load) : nothing

using Graphs
using Random
using StatsBase

# custom modules
using .GraphFragments
using .GraphOptimizationIterators
using .SupportFunctions


##  EXPORTS

export continuation
export iterand!
export log_iteration!
export log_result!
#
export Ant1
export AntColony3

export IterSpace
""";

###########################
#    BEGIN DEFINITIONS    #
###########################

# some default parameters
default_parameter_aco_beta = 0.25
default_parameter_aco_num_elite = 0.5
default_parameter_aco_population_size = 0.1 # should be smaller with larger graphs
default_parameter_aco_rho = 0.1
default_parameter_aco_tau_0 = 1.0

"""
Create an Ant, used to converge toward a loally optimal solution. 


# Constructs 

```Ant1(path, measure)```

##  Initialization Arguments

- `characteristics`: vector of unique elements representing the set of vertices
    being analyzed


##  Optional Arguments

"""
struct Ant1
    k::Int64
    measure::VecOrNoth{Float64}
    path::Vector{Int64}

    function Ant1(
        path::Vector{Int64};
        measure::VecOrNoth{Float64} = nothing,
    )

        # setup the measure
        measure = isa(measure, Nothing) ? Vector{Float64}([]) : Vector{Float64}([measure])
        k = length(path)

        return new(
            k,
            measure,
            path,
        )

    end

end



"""

##  Initialization Arguments

- `ants`: vector of ants to use to setup the colony
- `graph`: AbstractGraph used to get properties and initialize the decision 
    heuristic ``𝞰_i`` for vertex ``i``

##  Optional Arguments

- `beta`: heuristic exponentiation in probability term for each ant.
- `heuristic`: heuristic to use for guiding probabilities in ants
- `rho`: evaporation rate of pheremones left behind by each ant; ``0 ≤ 𝞺 ≤ 1``
- `tau_0`: initial pheremone value ``𝞽_0``. Should be set with a consideration
    for the order of the objective function; for objectives ranging from 0 - 1, 
    a value of 0.5 can be a good starting point


##  Other Properties

- `n_ants`: number of ants to use in searching
- `n_vertices`: number of vertices in the graph (|V|)
- `k`: number of vertices to identify in optimal subset of graph
- `path_index`: vector of vectors ``v_i`` that store which ants visited node 
    ``i``. The vectors are dynamic and can increase in size if the number of
    ants exceeds the size of ``v_i``. A value of ``v_{ij} = 0`` implies that 
    the number of ants to visit ``i`` is ``< j``
- `path_index_size`: entry ``i`` stores the number of ants in path_index ``i``

"""
struct AntColony3
    ants::Vector{Ant1}
    beta::Real
    heuristic::Symbol
    heuristics::Vector{Float64}
    k::Int64
    n_ants::Int64
    n_vertices::Int64
    path_index::Vector{Vector{Int64}}
    path_index_size::Vector{Int64}
    percentiles::Vector{Float64}
    percentiles_index::Vector{Int64}
    pheremones::Vector{Float64}
    rho::Float64
    tau_0::Real
    

    function AntColony3(
        ants::Vector{Ant1},
        graph::AbstractGraph;
        beta::Real = default_parameter_aco_beta,
        heuristic::Symbol = :betweenness_centrality,
        rho::Float64 = default_parameter_aco_rho,
        tau_0::Real = default_parameter_aco_tau_0,
    )
        
        ##  CHECK ANTS

        # check specification of ants
        n_ants = length(ants)
        (n_ants == 0) && error("No ants found in AntColony3: stopping...")

        # check ant k
        k = nothing
        for (i, ant) in enumerate(ants)
            isa(k, Nothing) && (k = ant.k; continue)
            (k != ant.k) && error("Error trying to setup AntColony3: inconsistent `k` found in ants. Ensure values are uniform.")
        end
        

        ##  CHECK OTHER VALUES

        # check vertex sizes862533

        n_vertices = nv(graph)


        ##  INITIALIZE 

        # set heuristics
        if heuristic == :betweenness_centrality
            heuristics = betweenness_centrality(graph)
        elseif heuristic == :eigenvector_centrality
            heuristics = eigenvector_centrality(graph)
        end

        # initialize pheremones
        tau_0 = max(tau_0, 0.0000001)
        pheremones = tau_0 .* ones(Float64, n_vertices)

        # initialize percentiles for ants, used in tracking rankings of pheremone performance
        percentiles = zeros(Float64, n_ants)
        percentiles_index = collect(1:n_ants)
        
        # initialize the path index, which stores which ants visited the 
        vec_size = n_ants .* (heuristics ./ sum(heuristics))
        vec_size = max.(ceil.(Int64, vec_size), 1)
        path_index = [zeros(Int64, x) for x in vec_size]
        path_index_size = zeros(Int64, n_vertices)
        

        return new(
            ants,
            beta,
            heuristic,
            heuristics,
            k,
            n_ants,
            n_vertices,
            path_index,
            path_index_size,
            percentiles,
            percentiles_index,
            pheremones,
            rho,
            tau_0,
        )
    end
end



##################################
###                            ###
###    BEGIN CORE FUNCTIONS    ###
###                            ###
##################################

"""
Calculate the value of the objective function for each organism in the 
    population


# Constructs

```
aco_add_measures!(
    params_optimization::OptimizationParameters,
    colony::AntColony3;
    overwrite_existing_measure::Bool = true,
)
```


##  Function Arguments

- `params_optimization`: OptimizationParameters object
- `colony`: AntColony


##  Keyword Arguments

-  `overwrite_existing_measure`: Ants instantiate without a measure; the 
    default behavior is to overwrite the measure as ants go back and try new 
    paths.

"""
function aco_add_measures!(
    params_optimization::OptimizationParameters,
    colony::AntColony3;
    overwrite_existing_measure::Bool = true,
)

    # calculate metrics associated with each ant
    for (i, ant) in enumerate(colony.ants)

        # skip if a measure is in place and 
        (!overwrite_existing_measure & (length(ant.measure) == 1)) && continue

        objective = get_objective_from_removed_vertices(
            params_optimization,
            ant.path,
        )

        resize!(ant.measure, 1)
        ant.measure[1] = objective
    end
end



"""
Calculate the value of the objective function for each ant that visited a vertex


# Constructs

```
aco_update_pheremone_trails!(
    params_optimization::OptimizationParameters,
    colony::AntColony3,
    num_elite::Int64;
)
```


##  Function Arguments

- `params_optimization`: OptimizationParameters object
- `colony`: ant colony to update


##  Keyword Arguments

-  `overwrite_existing_measure`: Organisms instantiate without a measure; the 
    default behavior is to leave a measure in place once it is calculated. 

"""
function aco_update_pheremone_trails!(
    params_optimization::OptimizationParameters,
    colony::AntColony3,
    num_elite::Int64;
    #scalar_delta::
)

    # normalizing weight for ranked addition + max percentile; note that best gets rank num_elite +1
    denom = (num_elite + 1)*(num_elite + 2)/2 #1
    min_percentile = (num_elite > 0) ? (1 - (num_elite - 1)/colony.n_ants) : 0.0

    # calculate pheremones
    for (i, tau) in enumerate(colony.pheremones)
        
        # add in the best objective value if the current vertex is in S_best; then, add previous value
        tau_out = Int64(i in params_optimization.S_best)
        tau_out *= ((num_elite + 1)/denom)*aco_delta_tau(params_optimization.obj_best)
        tau_out += tau*(1 - colony.rho)
        
        # add component ants that visited this vertex
        num_ants = colony.path_index_size[i]

        if num_ants > 0

            for j in 1:num_ants

                # get index and check rank/percentile
                ant_index = colony.path_index[i][j]
                perc = colony.percentiles[ant_index]
                (perc < min_percentile) && continue

                # if the percentile is high enough, get the weight from rank; then, weight the pheremone deposit
                rank = (1 - perc)*colony.n_ants + 1
                weight = (num_elite - rank + 1)/denom
                
                measure_cur = weight*aco_delta_tau(colony.ants[ant_index])#.measure[1]
                tau_out += measure_cur
            end
        end
        
        colony.pheremones[i] = tau_out
        
    end
end

"""
Function that determines ∆𝞽(x) for a measure x

# Constructs



"""
function aco_delta_tau(
    measure::Real;
    log_shift::Real = 4,
)

    #out = measure
    out = log((measure + log_shift)/log_shift)

    return out
end

function aco_delta_tau(
    ant::Ant1,
)   
    meas = ant.measure
    out = (length(meas) > 0) ? meas[1] : 0.0
    out = aco_delta_tau(out)

    return out
end


"""
Support function to streamline calculating the population size based on the 
    input `population_size`


# Constructs

```
aco_get_num_elite(
    colony::AntColony,
    num_elite::Real;
    max_frac::Float64 = 0.5,
)
```


##  Function Arguments

- `colony`: AntColony object containing information about the population
- `num_elite`: specification of number of high performers to pass from
    generation to generation. Similar to population size, can be specified as a 
    number or a fraction. 

    NOTE: if > 0, always passes at least one. Set to 0 to turn elitism off.

    * If entered as a floating point 0 < s < 1, assumes that it represents the 
        top fraction of the population

      For example, with a population of 50 and a value of 0.05, then the top 
        2.5--rounded up to 3--performers will be passed to the next generation.

    * If entered as an integer > 0, then assumes that this is the number of
        elites to pass. Always capped at 50% of the population.

##  Keyword Arguments

- `max_frac`: maximum allowable fraction of the population that can be specified
    as elite
"""
function aco_get_num_elite(
    colony::AntColony3,
    num_elite::Real;
    max_frac::Float64 = 0.5,
)

    max_size = floor(Int64, colony.n_ants*max_frac)

    if num_elite >= 1

        num_elite = ceil(Int64, num_elite)
        num_elite = min(num_elite, max_size)

    else

        frac = max(min(num_elite, max_frac), 0.0)
        num_elite = ceil(colony.n_ants*frac)
        num_elite = min(num_elite, max_size)
    end

    num_elite = Int64(num_elite)

    return num_elite
end



"""

Get the AbstractWeights used to guide sampling for ants

# Constructs

```
aco_get_path_weights(
    colony::AntColony3,
)
```


##  Function Arguments

- `colony`: the ant colony to use to calculate weights
"""
function aco_get_path_weights(
    colony::AntColony3,
)
    # 
    vec_eta = colony.heuristics
    vec_tau = colony.pheremones
    
    # convert to StatsBase weights
    weights = vec_tau .* (vec_eta .^ colony.beta)
    weights = Weights(weights)

    return weights
end


"""

Sample new paths for ants in the colony

# Constructs

```
aco_update_ant_paths!(
    colony::AntColony3,
    sample_space::Vector{Int64},
    k::Int64,
)
```

```
aco_update_ant_paths!(
    params_optimization::OptimizationParameters,
    colony::AntColony3,
)
```


##  Function Arguments

- `colony`: the ant colony to use to calculate weights
- `sample_space`: the sample space from which to build paths
- `k`: number of points in the path (size of discrete node pool)
- `params_optimization`: specify the OptimizationParameters object to call the
    sample space and number of vertices to remove 

##  Keyword Arguments

- `include_s0`: include params_optimization.S in the sample? Used for 
    initialization
"""
function aco_update_ant_paths!(
    colony::AntColony3,
    sample_space::Vector{Int64},
    k::Int64;
    include_s0::Bool = false,
)

    # clear path index
    reset_path_index!(colony, )

    # get sampling weights for each vertex based on the heuristic and current value of pheremones
    weights = aco_get_path_weights(colony)

    # for each ant, get a new path based on weights, then update the path index
    for (i, ant) in enumerate(colony.ants)
        
        if (i == 1) & include_s0
            s = copy(params_optimization.S)
        else 
            s = StatsBase.sample(
                sample_space,
                weights,
                k;
                replace = false,
            )
        end
        
        sort!(s)
        
        # update path and path index
        ant.path .= s

        push_to_path_index_unsafe!(colony, i, s, )
    end
end



function aco_update_ant_paths!(
    params_optimization::OptimizationParameters,
    colony::AntColony3;
    kwargs...
)

    aco_update_ant_paths!(
        colony,
        params_optimization.all_vertices,
        params_optimization.n_nodes;
        kwargs...
    )
end



"""
Update the ant colony at an iteration by doing the following actions:
    
    1. update paths, by sampling new paths and updating the path indices
    2. add measures associated with each ant
    3. update percentiles and ranks
    4. update pheremones


# Constructs

```
aco_update_colony!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
    colony::AntColony3,
    num_elite::Real;
    include_s0::Bool = false,
    msg::Union{String, Nothing}
)
```


##  Function Arguments

- `params_iterator`: IteratorParameters object used to track best values, 
    iteration index, etc.
- `params_optimization`: OptimizationParameters object
- `colony`: AntColony3 object to operate on
- `num_elite`: specification of number of elite, can be float 
    (0 ≤ `num_elite` < 1) or an integer (≥ 1). If Nothing, elitism is not used

##  Keyword Arguments

- `include_s0`: include params_optimization.S in the sample? Used for 
    initialization
- `msg`: optional message to pass in update
"""
function aco_update_colony!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
    colony::AntColony3,
    num_elite::Real;
    include_s0::Bool = false,
    msg::Union{String, Nothing} = nothing,
)

    # update paths, add measures, and update percentiles
    aco_update_ant_paths!(
        params_optimization,
        colony;
        include_s0 = include_s0,
    )

    aco_add_measures!(
        params_optimization, 
        colony,
    )

    aco_update_percentiles!(
        colony,
    )

    # update the iteration parameters parameters--must happen before updating pheremones
    aco_update_params!(
        params_iterator,
        params_optimization,
        colony;
        msg = msg,
    )

    # get number of elite and update the trails
    n_elite = aco_get_num_elite(colony, num_elite)
    aco_update_pheremone_trails!(
        params_optimization,
        colony,
        n_elite;
    )

end



"""
Support function to streamline calculating the population size based on the 
    input `population_size`


# Constructs

```
aco_get_population_size(
    params_optimization::OptimizationParameters,
    population_size::Int64;
)
```


##  Function Arguments

- `params_optimization`: OptimizationParameters object
- `population_size`: population size to instantiate
    * If entered as a floating point 0 < s < 1, assumes that it represents an 
        expected total proportion of the vertices as coverage 

    For example, with 1000 vertices and a fragmentation target removal of 4 
        vertices, a value of 0.05 => that the expected coverage should be 50
        vertices => a population of 12.5, which rounds up to 13.

    * If entered as an integer > 0, then assumes that this is the population 
        size

##  Keyword Arguments

"""
function aco_get_population_size(
    params_optimization::OptimizationParameters,
    population_size::Real;
)

    if population_size >= 1
        population_size = round(Int64, population_size)

    else
        target_size = max(population_size, 0.0)
        target_size *= params_optimization.graph_wrapper.dims[1]

        population_size = round(Int64, target_size/params_optimization.n_nodes)
        population_size = max(population_size, 1)
    end

    return population_size
end



"""
Initialize the population of organisms and populate their genome (random sample)

# Constructs

```
aco_initialize_colony(
    params_optimization::OptimizationParameters,
    population_size::Int64;
)
```


##  Function Arguments

- `params_optimization`: OptimizationParameters object
- `population_size`: population size to instantiate
    * If entered as a floating point 0 < s < 1, assumes that it represents an 
        expected total proportion of the vertices as coverage 

    For example, with 1000 vertices and a fragmentation target removal of 4 
        vertices, a value of 0.05 => that the expected coverage should be 50
        vertices => a population of 12.5, which rounds up to 13.

    * If entered as an integer > 0, then assumes that this is the population 
        size

##  Keyword Arguments

- `kwargs...`: passed to AntColony3 on initialization
"""
function aco_initialize_colony(
    params_optimization::OptimizationParameters,
    population_size::Real;
    kwargs...
)

    population_size = aco_get_population_size(
        params_optimization,
        population_size,
    )

    # initialize the storage vector (note that resize! and fill! point to same object)
    vec_population = Vector{Ant1}(
        [
            Ant1(collect(1:params_optimization.n_nodes))
            for x in 1:population_size
        ]
    )

    # initialize the colony; updated afterwards 
    colony = AntColony3(
        vec_population,
        params_optimization.graph;
        kwargs...
    )

    return colony
    
end



"""
Update percentiles in an OrganismPopulation

# Constructs

```
aco_update_percentiles!(
    population::OrganismPopulation5,
)
```


##  Function Arguments

- `population`: OrganismPopulation to update


##  Keyword Arguments

"""
function aco_update_percentiles!(
    colony::AntColony3,
)
    # calculate percentiles and update in the object
    vec_percs = get_crude_percentiles([x.measure[1] for x in colony.ants])

    # get indices
    percsort = collect(zip(vec_percs, 1:length(vec_percs)))
    sort!(percsort, rev = true)

    # update percentiles and percentiles indices
    for (i, k) in enumerate(vec_percs)
        colony.percentiles[i] = k
        colony.percentiles_index[i] = percsort[i][2]
    end
end



"""
Following a 

# Constructs

```
aco_update_params!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
    population::OrganismPopulation5;
    msg::Union{String, Nothing} = nothing,
)
```

##  Function Arguments

- `params_iterator`: IteratorParameters object used to track iteration and store
    objective function value
- `params_optimization`: OptimizationParameters struct containing the current
    set of candidate vertices to remove (S) 
- `population`: OrganismPopulation storing percentiles etc.


##  Keyword Arguments

- `msg`: optional message to pass for logging (must be a string)

"""
function aco_update_params!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
    colony::AntColony3;
    msg::Union{String, Nothing} = nothing,
)
    # log the event?
    log = params_optimization.log_info & isa(msg, String)
    log && @info(msg)

    ind_best = colony.percentiles_index[1]
    ant_best = colony.ants[ind_best]
    obj_best = ant_best.measure[1]
    S_best = ant_best.path
     
    # update best value
    scal = (params_optimization.objective_direction == :maximize) ? 1 : -1
    update_best = isa(params_optimization.obj_best, Nothing)
    update_best |= !update_best ? (obj_best*scal > params_optimization.obj_best*scal) : false

    if update_best
        params_iterator.obj_best = obj_best
        params_optimization.obj_best = obj_best
        params_optimization.S_best .= S_best

        # reset count from no improvement
        params_iterator.i_no_improvement = -1
    end
    
    # update set states
    params_optimization.S = S_best
    params_optimization.S_comp = collect(
        setdiff(
            Set(params_optimization.all_vertices), 
            Set(params_optimization.S)
        )
    )

end




###########################
#    SUPPORT FUNCTIONS    #
###########################

"""
Push ant `ant_index` to the path index for vertex `vertex` in colony `colony`. 
"""
function push_to_path_index_unsafe!(
    colony::AntColony3,
    ant_index::Int64,
    vertex::Int64,
)
    @inbounds begin

        s = colony.path_index_size[vertex]
        ind = colony.path_index[vertex]

        # double the size of the index vector; this operation should be rare
        (s == length(ind)) && resize!(ind, 2*s)

        colony.path_index[vertex][s + 1] = ant_index
        colony.path_index_size[vertex] += 1
    end
end

function push_to_path_index_unsafe!(
    colony::AntColony3,
    ant_index::Int64,
    vertices::Vector{Int64},
)
    for v in vertices
        push_to_path_index_unsafe!(
            colony,
            ant_index,
            v,
        )
    end
end



"""
Support function for Ant Colony optimization; cleat the path indices
"""
function reset_path_index!(
    colony::AntColony3,
)
    # reset 
    fill!.(colony.path_index, 0)
    fill!(colony.path_index_size, 0)
end





##################################
###                            ###
###    ITERAND CONSTRUCTORS    ###
###                            ###
##################################

"""
Define conditions under when to break iteration

# Constructs

```
continuation(
    ip::IteratorParameters,
    op::OptimizationParameters
)::Bool
```

##  Function Arguments

- `ip`: parameters used in iteration
- `op`: parameters used to set up the algorithm run


# Returns

Returns a `Bool` defining whether or not to continue iterating
"""
function aco_continuation(
    ip::IteratorParameters,
    op::OptimizationParameters
)::Bool

    out = (ip.i < op.max_iter) 
    #out &= (ip.eps >= op.epsilon) 
    out &= (ip.i_no_improvement < op.max_iter_no_improvement)
    out &= ip.cont

    return out
end





"""
Iterand function for greedy optimization of graph fragmentation

# Constructs

```
iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```


##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run


##  Keyword Arguments

##  Function Arguments

- `beta`: heuristic exponentiation in probability term for each ant.
- `heuristic`: heuristic to use for guiding probabilities in ants
- `num_elite`: specification of number of high performers to pass from
    generation to generation. Similar to population size, can be specified as a 
    number or a fraction. 

    NOTE: if > 0, always passes at least one. Set to 0 to turn elitism off.

    * If entered as a floating point 0 < s < 1, assumes that it represents the 
        top fraction of the population

      For example, with a population of 50 and a value of 0.05, then the top 
        2.5--rounded up to 3--performers will be passed to the next generation.

    * If entered as an integer > 0, then assumes that this is the number of
        elites to pass. Always capped at 50% of the population.

- `population_size`: size of the population
    * If entered as a floating point 0 < s < 1, assumes that it represents an 
        expected total proportion of the vertices as coverage 

      For example, with 1000 vertices and a fragmentation target removal of 4 
        vertices, a value of 0.05 => that the expected coverage should be 50
        vertices => a population of 12.5, which rounds up to 13.

    * If entered as an integer > 0, then assumes that this is the population 
        size
- `rho`: evaporation rate of pheremones left behind by each ant; ``0 ≤ 𝞺 ≤ 1``
- `tau_0`: initial pheremone value ``𝞽_0``. Should be set with a consideration
    for the order of the objective function; for objectives ranging from 0 - 1, 
    a value of 0.5 can be a good starting point
"""
function aco_iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters;
    beta::Real = default_parameter_aco_beta,
    heuristic::Symbol = :betweenness_centrality,
    init_colony_with_op_s::Bool = false,
    key_colony::Symbol = :colony,
    num_elite::Real = default_parameter_aco_num_elite,
    population_size::Real = default_parameter_aco_population_size,
    rho::Float64 = default_parameter_aco_rho,
    tau_0::Real = default_parameter_aco_tau_0,
)

    # initialize population if necessary
    if params_iterator.i == 0
        params_iterator.dict_optional_use[key_colony] = aco_initialize_colony(
            params_optimization,
            population_size;
            beta = beta,
            heuristic = heuristic,
            rho = rho,
            tau_0 = tau_0,
        )
    end

    # try getting previous state
    colony = get(
        params_iterator.dict_optional_use, 
        key_colony, 
        nothing
    )

    if isa(colony, Nothing)
        msg = "key $(key_colony) not found in the iterator dictionary. Stopping Ant Colony Algorithm algorithm."
        error(msg)
    end

    # force the optimization parameters .S to be included in the colony?
    include_s0 = init_colony_with_op_s & (params_iterator.i == 0)

    # update the colony
    aco_update_colony!(
        params_iterator,
        params_optimization,
        colony,
        num_elite;
        include_s0 = include_s0,
    )

    # update dictionary to pass to next iteration 
    params_iterator.dict_optional_use[key_colony] = colony

end



"""
Iteration logging function (within iteration)

# Constructs

```
log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function aco_log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("$(params_iterator.i) iterations complete with value $(params_iterator.obj_best)")
end



"""
Iteration logging function (after iteration)

# Constructs

```
log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function aco_log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("Ant Colony algorithm complete in $(params_iterator.i) iterations.") 
end



"""
Iterand for the greedy optimization approach
"""
aco_iterand = Iterand(
    aco_continuation,
    aco_iterand!;
    log_iteration! = aco_log_iteration!,
    log_result! = aco_log_result!,
    opts_prefix = :aco,
)

