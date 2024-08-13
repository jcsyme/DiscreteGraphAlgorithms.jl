"""
Implement a genetic annealing algorithm for optimizations focused on selecting
    a subsset of vertices.


## Key Functions

`genetic_continuation`
`genetic_iterand!`
`genetic_log_iteration!`
`genetic_log_result!`


##  Key Structures

`genetic_iterand`: the GraphOptimizationIterators.Iterand object used inside of 
    heuristic optimization to implement simulated annealing. This object 
    defines steps to take inside each iteration, ideal state swaps, stopping 
    conditions, acceptance thresholds, and more.
"""


###########################
#    BEGIN DEFINITIONS    #
###########################

# some types
FloatOrNoth = Union{Float64, Nothing}
VecOrNoth{T} = Union{Vector{T}, Nothing}
IterSpace{T} = Union{Vector{T}, UnitRange{T}, Base.OneTo{T}}

"""
Create an immutable orgnism, used to pass to other generations

# Constructs 

##  Initialization Arguments

- `characteristics`: vector of unique elements representing the set of vertices
    being analyzed


##  Optional Arguments

"""
struct Organism
    genome::Vector{Int64}
    measure::VecOrNoth{Float64}
    p_automatic::FloatOrNoth
    p_crossover::FloatOrNoth
    p_mutate::FloatOrNoth
    p_parent::FloatOrNoth

    function Organism(
        genome::Vector{Int64};
        measure::Union{Float64, Nothing} = nothing,
        p_automatic::FloatOrNoth = nothing,
        p_crossover::FloatOrNoth = nothing,
        p_mutate::FloatOrNoth = nothing,
        p_parent::FloatOrNoth = nothing,
    )

        sort!(genome)

        # call SupportFunctions utility
        p_automatic = random_unif_from_float_or_noth(p_automatic)
        p_crossover = random_unif_from_float_or_noth(p_crossover)
        p_mutate = random_unif_from_float_or_noth(p_mutate)
        p_parent = random_unif_from_float_or_noth(p_parent)

        measure = isa(measure, Nothing) ? Vector{Float64}([]) : Vector{Float64}([measure])

        return new(
            genome,
            measure,
            p_automatic,
            p_crossover,
            p_mutate,
            p_parent,
        )

    end

end



"""

- `percentiles_index`: stores the index of element with rank i in 
    OrganismPopulation.percentiles, 1 <= i <= size
"""
struct OrganismPopulation
    population::Vector{Organism}
    percentiles::Vector{Float64}
    percentiles_index::Vector{Int64}
    size::Vector{Int64}
    

    function OrganismPopulation(
        population::Vector{Organism}
    )
        
        size = [length(population)]
        percentiles = zeros(Float64, size[1])
        percentiles_index = collect(1:size[1])

        return new(
            population,
            percentiles,
            percentiles_index,
            size,
        )
    end
end

VecOrPop = Union{Vector{Organism}, OrganismPopulation}



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
genetic_add_measures!(
    params_optimization::OptimizationParameters,
    vec_population::Vector{Organism};
)
```


##  Function Arguments

- `params_optimization`: OptimizationParameters object
- `vec_population`: vector of


##  Keyword Arguments

-  `overwrite_existing_measure`: Organisms instantiate without a measure; the 
    default behavior is to leave a measure in place once it is calculated. 

"""
function genetic_add_measures!(
    params_optimization::OptimizationParameters,
    population::VecOrPop;
    overwrite_existing_measure::Bool = false,
)

    vec_population = isa(population, OrganismPopulation) ? population.population : population

    # calculate metrics associated with 
    for (i, org) in enumerate(vec_population)

        # skip if a measure is in place and 
        (!overwrite_existing_measure & (length(org.measure) == 1)) && continue

        objective = get_objective_from_removed_vertices(
            params_optimization,
            org.genome,
        )
        resize!(org.measure, 1)
        org.measure[1] = objective
        
    end
end


"""
Perform a crossover between two organisms--calls the appropriate function.

# Constructs

```
genetic_crossover(
    org_a::Organism,
    org_b::Organism,
    crossover_function::Symbol
)
```


##  Returns

- `Organism`: if successful, returns crossover organism
- `nothing`: if unsuccesful or if `crossover_function` is invalidly specified


##  Function Arguments

- `org_a`: first parent organism
- `org_b`: second parent organism
- `crossover_function`: crossover function to use. Valid options are:
    * `:random_mapping`: randomly pull from parent genomes, weighting selection
        probabilties by the value in org.measure[1] (calls 
        genetic_crossover_random_mapping)


##  Keyword Arguments

- `kwargs`: passed to the crossover function

"""
function genetic_crossover(
    org_a::Organism,
    org_b::Organism,
    crossover_function::Symbol;
    kwargs...
)
    out = nothing

    if crossover_function == :random_mapping
        out = genetic_crossover_random_mapping(org_a, org_b; kwargs...)
    end
  
    
    return out
end



"""
Implement a GA crossover using a random mapping


# Constructs

```
genetic_crossover_random_mapping(
    org_a::Organism,
    org_b::Organism;
    stop_on_no_measure::Bool = false,
    weight_params::Bool = false,
)
```


##  Returns

- `Organism`: if successful, returns crossover organism
- `nothing`: if `stop_on_no_measure` and either `org_a` or `org_b` is missing a 
    meausre


##  Function Arguments

- `org_a`: first parent organism
- `org_b`: second parent organism


##  Keyword Arguments

- `stop_on_no_measure`: if a measure value is not found for either organism, 
    stop the estimation? Otherwise, if false, assign probability randomly
- `weight_params`: set to `true` to use the relative measure weights to weight 
    parameters for the child
"""
function genetic_crossover_random_mapping(
    org_a::Organism,
    org_b::Organism;
    stop_on_no_measure::Bool = false,
    weight_params::Bool = false,
)
    
    ##  GET SHARED TRAITS/COMPLEMENTS
    
    shared_traits = intersect(org_a.genome, org_b.genome)
    n_shared = length(shared_traits)
    compl_a = setdiff(org_a.genome, shared_traits)
    compl_b = setdiff(org_b.genome, shared_traits)
    
    # identify the number of shared elements
    n_a = length(compl_a)
    n_b = length(compl_b)
    n_compl = min(length(compl_a), length(compl_b))
    
    # shuffle and make an implicit mapping
    shuff_a = shuffle(compl_a)
    shuff_b = shuffle(compl_b)
    
    
    ##  GET RELATIVE PROBABILITIES AND ASSIGN
    
    # check if each organism has a valid measure; if conditions are right, return nothing
    valid_a = (length(org_a.measure) == 1)
    valid_b = (length(org_b.measure) == 1)
    (!(valid_a & valid_b) & stop_on_no_measure) && (return nothing)

    p_a = valid_a ? org_a.measure[1] : Random.rand()
    p_b = valid_b ? org_b.measure[1] : Random.rand()
    denom = p_a + p_b
    (denom != 0) && (p_a /= denom; p_b /= denom;)
    

    # if there are extra, add them randomly based on the probability 
    n_extra_a = n_a - n_compl 
    n_extra_b = n_b - n_compl
    vec_sample = Vector{Int64}()
    
    if n_extra_a != n_extra_b
        vec_sample = (n_extra_a > n_extra_b) ? shuff_a : shuff_b
        p = (n_extra_a > n_extra_b) ? p_a : p_b
        vec_sample = [x for x in vec_sample[(n_compl + 1):end] if Random.rand() < p]
    end

    
    # next, use random sampling to assign
    vec_genome = [shared_traits; ones(Int64, n_compl); vec_sample]
    if n_compl > 0
        for i in 1:n_compl
            ind = n_shared + i
            vec_genome[ind] = (Random.rand() <= p_a) ? shuff_a[i] : shuff_b[i]
        end
    end
    
    sort!(vec_genome)
    
    
    ##  FINALLY, ASSIGN PROBABILITIES TO THE NEW ORGANISM
    
    p_1 = weight_params ? p_a : Random.rand()
    p_2 = 1 - p_1
    p_automatic = p_1*org_a.p_automatic + p_2*org_b.p_automatic
    p_crossover = p_1*org_a.p_crossover + p_2*org_b.p_crossover
    p_mutate = p_1*org_a.p_mutate + p_2*org_b.p_mutate
    p_parent = p_1*org_a.p_parent + p_2*org_b.p_parent
    
    
    # set output
    org_out = Organism(
        vec_genome;
        p_automatic = p_automatic,
        p_crossover = p_crossover,
        p_mutate = p_mutate,
        p_parent = p_parent,
    )
    
    return org_out
end



"""
Implement the evolutionary process for a 

# Constructs

```
genetic_evolve(
    params_optimization::OptimizationParameters,
    population::OrganismPopulation,
    num_elites::Real;
    crossover_function::Symbol = :random_mapping,
    mutation_transformation::Function = identity,
)
```


##  Function Arguments

- `params_optimization`: OptimizationParameters object
- `population`: OrganismPopulation to evolve
- `num_elites`: specification of number of elites (see ?genetic_get_num_elite) 
    for more information on this input

##  Keyword Arguments

- `crossover_function`: crossover function to use. See ?genetic_crossover for 
    valid options
- `mutation_transformation`: probability transformation to apply to organisms'
    mutation probabilities. Should be a mapping f:[0, 1] -> [0, 1].
"""
function genetic_evolve(
    params_optimization::OptimizationParameters,
    population::OrganismPopulation,
    num_elites::Real;
    crossover_function::Symbol = :random_mapping,
    mutation_transformation::Function = identity,
)
    
    # start by getting elites
    population_new = genetic_get_elites(
        population,
        num_elites,
    )
    
    # generate a set of indices to ensure there's no repetition
    set_pop = Set(
        Tuple(x for x in org.genome)
        for org in population_new
    )
    
    # vector to use to add new children
    n_new = population.size[1] - num_elites
    population_add = Vector{Organism}([population.population[1] for x in 1:n_new])
    
    
    # weigh base on probability of being a parent
    # weights_sample = pop.percentiles .* [x.p_parent for x in pop.population]
    weights = population.percentiles
    weights = Weights(weights)
    verts = Graphs.vertices(params_optimization.graph)
    
    # next, iterate
    i = 1
    
    while i <= n_new
        
        # get parents and implement crossover
        parents = StatsBase.sample(
            1:population.size[1],
            weights,
            2,
        )
        
        # get child
        child = genetic_crossover(
            population.population[parents[1]],
            population.population[parents[2]],
            crossover_function,
        )
        
        # mutate
        genetic_mutate!(
            child,
            verts,
            :exchange;
            clear_measure = true,
            probability_transformation = mutation_transformation,
        )
        
        
        tup_genome = Tuple(x for x in child.genome)
        (tup_genome in set_pop) && continue
        
        # add to the set and the vector of new populations
        push!(set_pop, tup_genome)
        population_add[i] = child
        
        i += 1
    end
    
    
    # conver to an organism population, add measures, and generate percentiles
    population_out = OrganismPopulation(
        [population_new; population_add]
    )

    genetic_add_measures!(params_optimization, population_out)
    genetic_update_percentiles!(population_out)
    
    return population_out
    
end



"""
Support function to streamline calculating the population size based on the 
    input `population_size`

    
# Constructs

```
genetic_get_elites(
    population::OrganismPopulation,
    num_elite::Int64;
)
```

##  Function Arguments

- `population`: OrganismPopulation object containing information about the
    population
- `num_elite`: actual number (would be output of genetic_get_num_elite()) of 
    elites to pass


##  Keyword Arguments


"""
function genetic_get_elites(
    population::OrganismPopulation,
    num_elite::Int64;
)
    
    elite_indices = population.percentiles_index[1:num_elite]
    
    return population.population[elite_indices]
    
end



"""
Support function to streamline calculating the population size based on the 
    input `population_size`


# Constructs

```
genetic_get_num_elite(
    population::OrganismPopulation,
    num_elite::Real;
    max_frac::Float64 = 0.5,
)
```


##  Function Arguments

- `population`: OrganismPopulation object containing information about the
    population
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
function genetic_get_num_elite(
    population::OrganismPopulation,
    num_elite::Real;
    max_frac::Float64 = 0.5,
)

    max_size = floor(Int64, population.size[1]*max_frac)

    if num_elite >= 1

        num_elite = ceil(Int64, num_elite)
        num_elite = min(num_elite, max_size)

    else

        frac = max(min(num_elite, max_frac), 0.0)
        num_elite = ceil(population.size[1]*frac)
        num_elite = min(num_elite, max_size)
    end

    num_elite = Int64(num_elite)

    return num_elite
end



"""
Support function to streamline calculating the population size based on the 
    input `population_size`


# Constructs

```
genetic_get_population_size(
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
function genetic_get_population_size(
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
genetic_initialize_population(
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

- `init_pop_with_op_s`: initialize the population with params_optimization.S in
    the initial population? Set to `true` to ensure that params_optimization.S
    is in the starting state.
"""
function genetic_initialize_population(
    params_optimization::OptimizationParameters,
    population_size::Real;
    init_pop_with_op_s::Bool = false,
)

    population_size = genetic_get_population_size(
        params_optimization,
        population_size,
    )
    

    # initialize the storage vector (note that resize! and fill! point to same object)
    vec_population = Vector{Organism}(
        [
            Organism(collect(1:params_optimization.n_nodes))
            for x in 1:population_size
        ]
    )

    # initialize organisms
    for k in 1:population_size
        
        # 
        if (k == 1) & init_pop_with_op_s
            s = copy(params_optimization.S)
        else 
            s = StatsBase.sample(
                params_optimization.all_vertices, 
                params_optimization.n_nodes; 
                replace = false
            )
        end

        org = Organism(s)
        vec_population[k] = org
    end

    pop = OrganismPopulation(vec_population)

    return pop
    
end



"""
Mutate an offspring using an specified mutation function

# Constructs

```
genetic_mutate!(
    org::Organism,
    space::Vector{Int64};
    clear_measure::Bool = true,
    probability_transformation::Function = identity,
    kwargs...
)
```


##  Function Arguments

- `org`: organism to mutate
- `space`: vector of options to sample from
- `mutation_function`: function to call. Valid options are
    * `exchange`: randomly exchange elements with others in the sample space 
        using the probability of mutation for each element


##  Keyword Arguments

- `kwargs`: passed to mutation function; includes some of the following:

- `clear_measure`: remove the measure value stored in the organism if mutation 
    is succesful?
- `probability_transformation` (optional): function applied to the mutation 
    probability. Can be used to reduce the scalar if relative population traits 
    are favorable, e.g.
    NOTE: should be a map f:[0, 1] -> [0, 1]

"""
function genetic_mutate!(
    org::Organism,
    space::IterSpace{Int64},
    mutation_function::Symbol;
    kwargs...
)

    if mutation_function == :exchange
        genetic_mutation_exchange!(
            org,
            space;
            kwargs...
        )
    end

    return nothing
end



"""
Mutate an offspring using an exchange from a space

# Constructs

```
genetic_mutation_exchange!(
    org::Organism,
    space::Vector{Int64};
    clear_measure::Bool = true,
    probability_transformation::Function = identity,
)
```


##  Function Arguments

- `org`: organism to mutate
- `space`: vector of options to sample from


##  Keyword Arguments

- `clear_measure`: remove the measure value stored in the organism if mutation 
    is succesful?
- `probability_transformation`: function applied to the mutation 
    probability. Can be used to reduce the scalar if relative population traits 
    are favorable, e.g.
    NOTE: should be a map f:[0, 1] -> [0, 1]
"""
function genetic_mutation_exchange!(
    org::Organism,
    space::IterSpace{Int64};
    clear_measure::Bool = true,
    probability_transformation::Function = identity,
)
    
    # maximal mutation--do this to reduce comptuational load rather than sampling & deleting etc.
    mutation_full = StatsBase.sample(
        setdiff(space, org.genome),
        length(org.genome); 
        replace = false
    )
    
    j = 1

    @inbounds begin
        for (i, v) in enumerate(org.genome)
            (Random.rand() > probability_transformation(org.p_mutate)) && continue
            org.genome[i] = mutation_full[j]
            j += 1
        end
    end
    
    # only remove the measure if we know it changed
    (clear_measure & (j > 1)) && resize!(org.measure, 0);
    
    return nothing
end



"""
Update percentiles in an OrganismPopulation

# Constructs

```
genetic_update_percentiles!(
    population::OrganismPopulation,
)
```


##  Function Arguments

- `population`: OrganismPopulation to update


##  Keyword Arguments

"""
function genetic_update_percentiles!(
    population::OrganismPopulation,
)
    # calculate percentiles and update in the object
    vec_percs = get_crude_percentiles([x.measure[1] for x in population.population])

    # get indices
    percsort = collect(zip(vec_percs, 1:length(vec_percs)))
    sort!(percsort, rev = true)


    # update percentiles and percentiles indices
    for (i, k) in enumerate(vec_percs)
        population.percentiles[i] = k
        population.percentiles_index[i] = percsort[i][2]
    end
end



"""
Following a 

# Constructs

```
genetic_update_params!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
    population::OrganismPopulation;
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
function genetic_update_params!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
    population::OrganismPopulation;
    msg::Union{String, Nothing} = nothing,
)
    # log the event?
    log = params_optimization.log_info & isa(msg, String)
    log && @info(msg)

    ind_best = population.percentiles_index[1]
    org_best = population.population[ind_best]
    obj_best = org_best.measure[1]
    S_best = org_best.genome
     
    # update best value
    scal = (params_optimization.objective_direction == :maximize) ? 1 : -1
    update_best = isa(params_optimization.obj_best, Nothing)
    update_best |= !update_best ? (obj_best*scal > params_optimization.obj_best*scal) : false

    if update_best
        params_iterator.obj_best = obj_best
        params_optimization.obj_best = obj_best
        params_optimization.S_best .= S_best

        # reset no improvement count
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




"""
Define conditions under when to break iteration

# Constructs

```
genetic_continuation(
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
function genetic_continuation(
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
genetic_iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```


##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run


##  Keyword Arguments

- `init_pop_with_op_s`: initialize the population with params_optimization.S in
    the initial population? Set to `true` to ensure that params_optimization.S
    is in the starting state.
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
"""
function genetic_iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters;
    init_pop_with_op_s::Bool = false,
    key_population::Symbol = :population,
    num_elite::Real = 0.05,
    population_size::Real = 0.1,
)

    # initialize population if necessary
    if params_iterator.i == 0

        #println("num_elite:\t$(num_elite)")
        #println("population_size:\t$(population_size)")

        params_iterator.dict_optional_use[key_population] = genetic_initialize_population(
            params_optimization,
            population_size;
            init_pop_with_op_s = init_pop_with_op_s,
        )

        genetic_add_measures!(
            params_optimization, 
            params_iterator.dict_optional_use[key_population]
        )

        genetic_update_percentiles!(
            params_iterator.dict_optional_use[key_population]
        )
    end

    
    # try getting previous state
    org_population = get(
        params_iterator.dict_optional_use, 
        key_population, 
        nothing
    )

    if isa(org_population, Nothing)
        msg = "key $(key_population) not found in the iterator dictionary. Stopping Genetic algorithm."
        error(msg)
    end


    n_elite = genetic_get_num_elite(org_population, num_elite)
    org_population_new = genetic_evolve(
        params_optimization,
        org_population,
        n_elite,
    )

    # update 
    params_iterator.dict_optional_use[key_population] = org_population_new
    
    
    msg = "\n\n** keeping swap at iteration $(params_iterator.i) - F = $(params_iterator.obj_best)\n"
    genetic_update_params!(
        params_iterator,
        params_optimization,
        org_population_new;
        #msg = msg,
    )
    

end



"""
Iteration logging function (within iteration)

# Constructs

```
genetic_log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function genetic_log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("$(params_iterator.i) iterations complete with value $(params_iterator.obj_best)")
end



"""
Iteration logging function (after iteration)

# Constructs

```
genetic_log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function genetic_log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("Genetic algorithm complete in $(params_iterator.i) iterations.") 
end



"""
Iterand for the greedy optimization approach
"""
genetic_iterand = Iterand(
    genetic_continuation,
    genetic_iterand!;
    log_iteration! = genetic_log_iteration!,
    log_result! = genetic_log_result!,
    opts_prefix = :genetic,
)
