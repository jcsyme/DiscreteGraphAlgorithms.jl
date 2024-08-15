"""
Implement a stochastic gradient descent algorithm for KPP-Negative (maximize 
        fragmentation).


## Key Functions

`gs_continuation`
`gs_iterand!`
`gs_log_iteration!`
`gs_log_result!`


##  Key Structures

`gs_iterand`: the GraphOptimizationIterators.Iterand object used inside of 
    heuristic optimization to implement stochasic gradient descent. This object 
    defines steps to take inside each iteration, ideal state swaps, stopping 
    conditions, acceptance thresholds, and more.
"""



###########################
#    BEGIN DEFINITIONS    #
###########################

prefix_gs = :gs

"""
Define conditions under when to break iteration

# Constructs

```
gs_continuation(
    ip::IteratorParameters,
    op::OptimizationParameters
)::Bool
```

##  Function Arguments

- `ip`: parameters used in iteration
- `op`: parameters used to set up the algorithm run


##  Keyword Arguments

- `epsilon`: threshold used to denote convergence. If the directional value of 
    the change in the objective function is less than epsilon, then the 
    algorithm will terminate.


# Returns

Returns a `Bool` defining whether or not to continue iterating
"""
function gs_continuation(
    ip::IteratorParameters,
    op::OptimizationParameters;
    epsilon::Float64 = 10.0^(-6.0),
    kwargs...
)::Bool

    out = (ip.i < op.max_iter) 
    out &= (ip.eps >= epsilon) 
    out &= (ip.i_no_improvement < op.max_iter_no_improvement)
    out &= ip.cont

    return out
end



"""
Iterand function for greedy optimization of graph fragmentation

# Constructs

```
gs_iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run


##  Keyword Arguments

- `randomize_swap_count`: randomize the number of vertices that are swapped at
    each iteration?
"""
function gs_iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters;
    randomize_swap_count::Bool = false,
)
    if ((params_iterator.i == 0) & !isa(params_optimization.S, Nothing))
        swap_size = 0
    else
        swap_size = (
            randomize_swap_count 
            ? StatsBase.sample(1:n_nodes, 1; replace = false)[1] 
            : 1
        )
    end
    
    (S_try, S_out, S_in) = sample_for_swap(params_optimization, swap_size)


    # recalculate distances
    obj_try = get_objective_from_removed_vertices(
        params_optimization, 
        S_try,
    )

    if obj_try > params_iterator.obj

        msg = "\n\n** keeping swap at iteration $(params_iterator.i) - F = $(params_iterator.obj), F_0 = $(params_iterator.obj_0), F_try = $(obj_try)\n"

        update_accepted_swap_params!(
            params_iterator,
            params_optimization,
            obj_try,
            S_try;
            msg = msg,
        )
        """
        params_optimization.log_info ? @info(msg) : nothing
        params_iterator.eps = obj_try - params_iterator.obj
        params_iterator.obj_0 = params_iterator.obj
        params_iterator.obj = obj_try
        params_optimization.S = S_try
        params_iterator.i_no_improvement = -1
        """
    end

end



"""
Iteration logging function (within iteration)

# Constructs

```
gs_log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function gs_log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("$(params_iterator.i) iterations complete with value $(params_iterator.obj)")
end



"""
Iteration logging function (after iteration)

# Constructs

```
gs_log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function gs_log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("Stochastic gradiest descent complete in $(params_iterator.i) iterations.") 
end



"""
Iterand for the greedy optimization approach
"""
gs_iterand = Iterand(
    gs_continuation,
    gs_iterand!;
    log_iteration! = gs_log_iteration!,
    log_result! = gs_log_result!,
    opts_prefix = prefix_gs,
)
