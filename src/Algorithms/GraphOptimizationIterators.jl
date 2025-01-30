

##############################
###                        ###
###    BUILD STRUCTURES    ###
###                        ###
##############################
"""
Optimization parameters used to facilitate modular algorithm construction for
    DiscreteGraphAlgorithms.

# Constructs

```
OptimizationParameters
```


##  Initialization Arguments

- `n_nodes`: number of vertices to identify 
- `graph_wrapper`: GraphWrapper or Graph to use for graph access--if input as a 
    graph, will convert to GraphWrapper. 


##  Optional Arguments

- `benchmark_algorithms_to_try`: optional list of benchmarking algorithms to try
    for auto-selection of distance algorithms. If nothing, selects to all 
    available (default)
- `benchmark_n_runs`: number of runs to use in benchmarking to auto-select an 
    algorithm
- `benchmark_selection_metric`: metric to use for selection of distance 
    algorithm. Options include: 
        
- `dict_arrays_distance`: optional dictionary storing cache arrays 
    (pre-allocation) for distance algorithms. Should be used only if specifying
    distance_algorithm as well.
- `distance_algorithm`: iterative distance algorithm to use. If `auto`, then a 
    routine will be used to select a distance algorithm based on empircal 
    performance on the graph

    * if using fragmentation and `distance_algorithm = :auto`:

        * `benchmark_algorithms_to_try`: specify algorithms to try for selection
        * `benchmark_n_runs`: number of runs to benchmark
        * `benchmark_selection_metric`: metric used to select
    
        see `?select_algorithm_from_benchmark_for_iteration()` for more 
        information on these arguments

- `log_info`: log information as iteration proceeds?
- `max_iter`: Maximum number of iterations to perform
- `max_iter_no_improvement`: Optional specification of a maximum number of
    iterations to allow without improvement in the objective function
- `objective_direction`: `:maximize` or `:minimize`
- `objective_function_name`: objective function to use. Currently supported 
    methods are:

    * :fragmentation

- `opts`: dictionary including options to pass to the algorithm. To pass a
    keyword argument to the algorithm's iterand, you must prepend it with the
    algorithm's opts_prefix, which can be accessed using `Iterand.opts_prefix`.
    
    For example, Simulated Annuealing uses `:sann`. To pass the 
    `max_exploration_error` keyword argument to sann_iterand!, you would use

        opts = Dict(:sann_max_exploration_error => 0.3)

    Similarly, to pass the `init_pop_with_op_s` and `num_elite` keyword 
    arguments to the genetic algorithm (with example values), use

        opts = Dict(
            :genetic_init_pop_with_op_s => true,
            :genetic_num_elite => 0.1,
        )
- `parallel_approach`: approach to parallelize. See ?fragmentation for more. One 
    of the following options:
    * `auto`: choose based on size of graph
    * `parallel`: force parallel
    * `serial`: force serial

- `S`: Initial set `S` to use for graph algorithms. 
"""
mutable struct OptimizationParameters
    n_nodes::Int64
    graph_wrapper::GraphWrapper
    dict_arrays_distance::Union{Dict, Nothing}
    distance_algorithm::Symbol
    log_info::Bool
    max_iter::Int64
    max_iter_no_improvement::Int64
    objective_direction::Symbol
    objective_function_name::Symbol
    opts::Union{Dict, Nothing}
    S::Union{Vector{Int64}, UnitRange{Int64}, Nothing}
    all_vertices::Vector{Int64}
    graph::Union{SimpleGraph, SimpleDiGraph}
    obj_best::Union{Real, Nothing}
    objective_function::Function
    parallel_approach::Symbol
    S_0::Vector{Int64}
    S_best::Vector{Int64}
    S_comp::Union{Vector{Int64}, UnitRange{Int64}, Nothing}
    S_tup::Tuple
    
    function OptimizationParameters(
        n_nodes::Int64,
        graph_wrapper::Union{GraphWrapper, AbstractGraph};
        benchmark_algorithms_to_try::Union{Vector{Symbol}, Nothing} = nothing,
        benchmark_n_runs::Int64 = 10,
        benchmark_selection_metric::Symbol = :mean,
        dict_arrays_distance::Union{Dict, Nothing} = nothing,
        distance_algorithm::Symbol = :auto,
        log_info::Bool = false,
        max_iter::Int64 = 1000,
        max_iter_no_improvement::Int64 = 200,
        objective_direction::Symbol = :maximize,
        objective_function_name::Symbol = :fragmentation,
        opts::Union{Dict, Nothing} = nothing,
        parallel_approach::Symbol = :auto,
        S::Union{Vector{Int64}, UnitRange{Int64}, Nothing} = nothing,
        kwargs...
    )
        
        ##  CHECK INSTANTIATION  
        
        # check that input is specified correctly
        isa(graph_wrapper, AbstractGraph) && (graph_wrapper = graph_to_graph_wrapper(graph_wrapper, ))
        graph = copy(graph_wrapper.graph)
        m, n = size(graph)
        max_iter_no_improvement = minimum([max_iter_no_improvement, max_iter])

        # conditions to exit on
        return_nothing = (m < n_nodes)
        return_nothing |= (max_iter_no_improvement < 0)
        return_nothing |= (max_iter < 0)

        if return_nothing
            return nothing
        end

        

        
        ##  OPTIONS AND FUNCTION INIT

        # check options
        opts = isa(opts, Nothing) ? Dict{Symbol, Any}() : opts
        opts = Dict{Symbol, Any}((Symbol(k), v) for (k, v) in opts)
        
        # check objective direction
        valid_directions = [:maximize, :minimize]
        objective_direction = !(objective_direction in valid_directions) ? :maximize : objective_direction

        # check objective function
        valid_functions = [:fragmentation]
        objective_function_name = (
            (objective_function_name in valid_functions) 
            ? :fragmentation 
            : objective_function_name
        )
        
        # initialize as nothing, can update on a function by function basis
        dict_arrays = nothing

        # 
        if objective_function_name == :fragmentation
            
            # if auto-selecting, use an iteration benchmark on fragmentation to select
            if distance_algorithm == :auto
                @info "Calling select_algorithm_from_benchmark_for_iteration() to select distance algorithm on graph..."

                distance_algorithm = GraphFragments.select_algorithm_from_benchmark_for_iteration(
                    graph;
                    algorithms_to_try = benchmark_algorithms_to_try,
                    n_runs = benchmark_n_runs,
                    selection_metric = benchmark_selection_metric,
                )
                
                @info "Selected algorithm '$(distance_algorithm)'"
            end


            ##  SPAWN PRE-ALLOCATED DISTANCE ARRAYS?
            
            try_par = try_parallel(
                graph; 
                parallel_approach = parallel_approach, 
            )


            spawn_q = check_arrays_for_algorithm(
                distance_algorithm, 
                dict_arrays_distance,
            )

            if spawn_q

                spawn_type = try_par ? :DistribuatedArray : :Vector

                dict_arrays_distance = spawn_arrays(
                    graph_wrapper.graph,
                    distance_algorithm;
                    type = spawn_type,
                )
                
            end

            # set the objective function
            function objective_function(
                graph_input::AbstractGraph;
                kwargs...
            )
                
                out = fragmentation(
                    graph_input,
                    dict_arrays_distance;
                    distance_algorithm = distance_algorithm,
                    parallel_approach = parallel_approach,
                    kwargs...
                )

                return out
            end

            function objective_function(; kwargs...)
                out = fragmentation(
                    graph,
                    dict_arrays_distance;
                    distance_algorithm = distance_algorithm,
                    parallel_approach = parallel_approach,
                    kwargs...
                )

                return out
            end
            
        end
        



        ##  GRAPH AND INITIAL FRAGMENTATION SET INIT

        # some basic init
        all_vertices = collect(1:m)

        # initialize set of states
        S = isa(S, Nothing) ? StatsBase.sample(all_vertices, n_nodes; replace = false) : S
        S = issubset(Set(S), Set(all_vertices)) ? S : StatsBase.sample(all_vertices, n_nodes; replace = false)
        S_comp = collect(setdiff(Set(all_vertices), Set(S)))
        
        # get some offshoots
        sort!(S)
        sort!(S_comp)
        S_tup = Tuple(S)
        
        # set of states used to initialize object; S_0 does not change, but S_best stores the set S with the best value found
        S_0 = copy(S) 
        S_best = copy(S) 
        obj_best = nothing # optionally set

        
        return new(
            n_nodes,
            graph_wrapper,
            dict_arrays_distance,
            distance_algorithm,
            log_info,
            max_iter,
            max_iter_no_improvement,
            objective_direction,
            objective_function_name,
            opts,
            S,
            all_vertices,
            graph,
            obj_best,
            objective_function,
            parallel_approach,
            S_0,
            S_best,
            S_comp,
            S_tup
        )
    end
end



"""
# Summary

`IteratorParameters` is used to store iteration parameters that are passed to
    iterative functions in DiscreteGraphAlgorithms.


# Constructs

```
IteratorParameters
```


##  Initialization Arguments

- `obj`: best value of objective function


##  Properties (Mutable)

- `cont`: continue iterating
- `eps`: acceptable threshold for convergence. 
- `i`: iteration index
- `i_no_improvement`: number of iterations without improvement in the objective
    function
- `obj_0`: initial objective value (maximized)
- `obj_try`: candidate objective value
- `s`: 

"""
mutable struct IteratorParameters
    obj::Real
    cont::Bool
    dict_optional_use::Dict
    eps::Real
    i::Int64 
    i_no_improvement::Int64 
    obj_0::Real
    obj_best::Real
    obj_try::Real
    s::Int64 
    
    
    function IteratorParameters(
        obj::Real;
        cont::Bool = true,
        dict_optional_use::Dict = Dict(),
        eps::Real = 1.0,
        i::Int64 = 0,
        i_no_improvement::Int64 = 0,
        obj_0::Real = 0.0,
        obj_best::Real = 0.0,
        obj_try::Real = 0.0,
        s::Int64 = 0,
    )
        
        nothing

        return new(
            obj,
            cont,
            dict_optional_use,
            eps,
            i,
            i_no_improvement,
            obj_0,
            obj_best,
            obj_try,
            s
        )
    end
end



"""
Construct an iterand that can be called from the iterator function


- `log_iteration!`: a function used to log information within each iteration
    based on state variables stored in the objects `ip` and `op` and iteration 
    `iteration`. The function should have the following structure:

    ```
    log_iteration!( 
        ip::IteratorParameters,
        op::OptimizationParameters;
        iteration::Union{Int64, Nothing} = VALUE,
    )
    ```
- `log_result!`: an optional function used to log information following the 
    end of iteration that is based on state variables stored in the objects `ip` 
    and `op`. The function should have the following structure:

    ```
    log_result!( 
        ip::IteratorParameters,
        op::OptimizationParameters;
    )
    ```
- `opts_prefix`: optional prefix to provide to map keys in 
    OptimizationParameters.opts to this iterand; used for passing keyword 
    arguments. When searching for keyword arguments to pass, the iterate() 
    routine checks first for keys that match Iterand.opts_prefix. 

    Iterand specific keys in OptimizationParameters.opts will have the following
        form:

        PREFIX_key_generic

        where 
        * opts_prefix = :PREFIX and
        * key_generic = the generic parameter value to pass

""";
struct Iterand
    continuation::Function
    update!::Function
    log_iteration!::Function
    log_result!::Function
    opts_prefix::Union{Symbol, Nothing}

    function Iterand(
        continuation::Function,
        update!::Function;
        log_iteration!::Function = null_func,
        log_result!::Function = null_func,
        opts_prefix::Union{Symbol, Nothing} = nothing,
    )
      
        return new(
            continuation,
            update!,
            log_iteration!,
            log_result!,
            opts_prefix,
        )
    end
end




#######################
#    KEY FUNCTIONS    #
#######################


"""
Return options, passed from params_optimization.opts, for the Iterand.update!
    function. 

# Constructs

```
get_iterand_options(
    iterand::Iterand,
    params_optimization::OptimizationParameters;
)
```

##  Function Arguments

- `iterand`: algorithm iterand implemented as Iterand structure
- `params_optimization`: OptimizationParameters structure


##  Keyword Arguments

- `function_type`: Either `:update` or `:continuation`' 
"""
function get_iterand_options(
    iterand::Iterand,
    params_optimization::OptimizationParameters;
    function_type::Symbol = :update,
)
    # ensure function_type is valid
    !(function_type in [:update, :continuation]) && (function_type = :update)

    # first, check Iterand specific parameters
    dict_specific = (
        !isa(iterand.opts_prefix, Nothing)
        ? get_opts_subdict_from_prefix(
            params_optimization.opts,
            iterand.opts_prefix,
        )
        : Dict{Symbol, Any}()
    )

    # get applicable generic arguments, then overwrite with method-specific
    func = (function_type == :update) ? iterand.update! : iterand.continuation
    dict_generic = check_kwargs(func; params_optimization.opts...)
    dict_specific = check_kwargs(func; dict_specific...)
    merge!(dict_generic, dict_specific)

    return dict_generic
end



"""
Calculate the objective function on the base graph after removing vertices.

# Constructs

```
get_objective_from_removed_vertices(
    params_optimization::OptimizationParameters,
    S_try::Vector{Int64};
    ...kwargs,
)
```


##  Function Arguments

- `params_optimization`: OptimizationParameters used to provide objective
    function
- `S_try`: set of vertices (removed) to calculate objective function for


##  Keyword Arguments

- `kwargs`: passed to `params_optimization.objective_function`

"""
function get_objective_from_removed_vertices(
    params_optimization::OptimizationParameters,
    S_try::Vector{Int64};
    kwargs...
)
    #graph_try = copy(params_optimization.graph)
    rem_vertices!(params_optimization.graph, S_try)
    obj_try = params_optimization.objective_function(
        params_optimization.graph;
        kwargs...
    )
    
    reset_optimization_graph!(params_optimization)

    return obj_try
end



"""
Using the iterand's prefix, retrieve any options specified in dict_opts

# Constructs

```
get_opts_subdict_from_prefix(
    dict_ops::Dict,
    iterand_prefix::Symbol,
)
```

##  Function Arguments

- `dict_ops`: dictionary containing opts with algorithm-specific prefixes in 
    keys
- `iterand_prefix`: the iterand prefix to use

##  Keyword Arguments
"""
function get_opts_subdict_from_prefix(
    dict_ops::Dict,
    iterand_prefix::Symbol,
)
    # get the 
    prefix = "$(iterand_prefix)_"
    dict_specific = Dict{Symbol, Any}(
        (Symbol(replace(String(k), prefix => "")), v) 
        for (k, v) in dict_ops
        if startswith(String(k), prefix)
    )

    return dict_specific
end



"""
Call an iterand, or internal algorithmic structure

# Constructs

```
iterate(
    iterand::Iterand,
    params_optimization::OptimizationParameters;
    log_interval::Int64 = 10,
    kwargs...
)
```

##  Function Arguments

- `iterand`: optimization iterand, which governs the set of instructions to
    undertake at each iteration
- `params_optimization`: optimization parameters (including the set of outcomes)
    to keep track of


##  Keyword Arguments

- `log_interval`: iteration interval for logging (calls iterand.log_iteration!).
    Set to nothing to suppress logging.
- `max_n_restarts`: maximum number of restarts to allow. Set to 0 to accept the 
    best result after one pass.
- `return_best`: by default, return the best value of the objective found by the
    algorithm; sometimes, this is not the state in which the algorithm
    terminates (e.g., in simualted annealing)
- `kwargs...`: passed to `IteratorParameters` on setup 


##  Returns

- Returns a tuple of the following form:

    (
        obj_best,
        S_best,
        graph_best,
    )

    where:

        * obj_return: objective value associated with the set S_best
        * S_best: set of vertices associated with the best objective value
        * graph_best: graph that has removed vertices S_best
"""
function iterate(
    iterand::Iterand,
    params_optimization::OptimizationParameters;
    log_interval::Union{Int64, Nothing} = 10,
    max_n_restarts::Int64 = 0,
    return_best::Bool = true,
    kwargs...
)
    
    ##  CHECKS AND INITIALIZATION

    log_q = !isa(log_interval, Nothing)

    # check specification of inputs
    isa(params_optimization, Nothing) && (return nothing)
    log_interval = log_q ? max(1, log_interval) : nothing
    params_optimization.obj_best = nothing # erase previous value

    # select options to be passed to iterand
    dict_continuation_opts = get_iterand_options(
        iterand, 
        params_optimization;
        function_type = :continuation,
    )
    dict_update_opts = get_iterand_options(
        iterand, 
        params_optimization;
        function_type = :update,
    )
    
    # get current value of objective function (e.g., fragmentation)
    F = params_optimization.objective_function(
        params_optimization.graph;
    )

    # initialize iteration parameters and
    params_iterator = IteratorParameters(
        F; 
        kwargs...
    )


    ##  ITERATE USING THE HEURISTIC ITERAND

    while iterand.continuation(
        params_iterator, 
        params_optimization; 
        dict_continuation_opts...
    )

        iterand.update!(
            params_iterator, 
            params_optimization;
            dict_update_opts...
        )
        
        # update key parameters
        params_iterator.i += 1
        params_iterator.i_no_improvement += 1
        
        # log the iteration?
        if log_q
            ((params_iterator.i - 1)%log_interval == 0) && iterand.log_iteration!(params_iterator, params_optimization,)
        end
        
        # temporary--call from iterand
        (params_iterator.i + 1%500 == 0) && GC.gc()
    end
    
    log_q && iterand.log_result!(params_iterator, params_optimization,)

    # by default, return the best value found
    S_return = return_best ? params_optimization.S_best : params_optimization.S
    obj_return = return_best ? params_iterator.obj_best : params_iterator.obj

    # setup graph as part of return
    graph_out = copy(params_optimization.graph)
    rem_vertices!(graph_out, S_return)
    
    tup_out = (
        obj_return,
        S_return,
        graph_out
    )

    # call garbage collection to close it out
    GC.gc()
    
    return tup_out
end



"""
Reset params_optimization.graph based on the graph_wrapper

# Constructs

```
reset_optimization_graph!(
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_optimization`: OptimizationParameters structure to update

"""
function reset_optimization_graph!(
    params_optimization::OptimizationParameters,
)
    params_optimization.graph = copy(params_optimization.graph_wrapper.graph)
end



"""
Sample from an OptimizationParameters pool of potential vertices.

# Constructs


##  Function Arguments

- `params_optimization`: OptimizationParameters struct containing the current
    set of candidate vertices to remove (S) as well as the pool of all vertices
    (V)
- `swap_size`: number of vertices to swap from V to S


##  Keyword Arguments

- `sample_from_neighbors`: sample only from neighbors? If true, samples a first
    vertex, then successively samples neighbors to sample from. Can 
"""
function sample_for_swap(
    params_optimization::OptimizationParameters,
    swap_size::Int64;
    sample_from_neighbors::Bool = false,
)
    # get vertices leaving current state
    S_out = Vector{Int64}(
        StatsBase.sample(
            params_optimization.S, 
            swap_size; 
            replace = false
        )
    )

    # new vertices entering state
    S_in = Vector{Int64}(
        StatsBase.sample(
            setdiff(
                params_optimization.all_vertices, 
                params_optimization.S
            ), 
            swap_size; 
            replace = false
        )
    )

    # get new, proposed state
    S_try = Vector{Int64}(
        sort(
            union(
                setdiff(
                    params_optimization.S, 
                    S_out
                ), 
                S_in
            )
        )
    )

    tup = (S_try, S_out, S_in)

    return tup
end



"""
Following a swap, update the IteratorParameters object with new information

# Constructs

```
update_accepted_swap_params!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
    obj_new::Float64, 
    S_new::Vector{Int64};
    msg::Union{String, Nothing} = nothing,
)
```

##  Function Arguments

- `params_iterator`: IteratorParameters object used to track iteration and store
    objective function value
- `params_optimization`: OptimizationParameters struct containing the current
    set of candidate vertices to remove (S) 
- `obj_new`: new value of the objective function to take
- `S_new`: new candidate set of vertices to act on


##  Keyword Arguments

- `msg`: optional message to pass for logging (must be a string)

"""
function update_accepted_swap_params!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
    obj_new::Float64, 
    S_new::Vector{Int64};
    msg::Union{String, Nothing} = nothing,
)
    # log the event?
    log = params_optimization.log_info & isa(msg, String)
    log && @info(msg)

    # get a scalar based on whether or not you're maximizing or minimizing
    scal = (params_optimization.objective_direction == :maximize) ? 1 : -1

    # update objective function and convergence
    params_iterator.eps = (obj_new - params_iterator.obj)*scal
    params_iterator.obj_0 = params_iterator.obj
    params_iterator.obj = obj_new

    # update best value
    update_best = isa(params_optimization.obj_best, Nothing)
    update_best |= !update_best ? (obj_new*scal > params_optimization.obj_best*scal) : false

    if update_best
        params_iterator.obj_best = obj_new
        params_optimization.obj_best = obj_new
        params_optimization.S_best = S_new
    end
    
    # update set states
    params_optimization.S = S_new
    params_optimization.S_comp = collect(
        setdiff(
            Set(params_optimization.all_vertices), 
            Set(params_optimization.S)
        )
    )

    # reset count from no improvement
    params_iterator.i_no_improvement = -1
end
