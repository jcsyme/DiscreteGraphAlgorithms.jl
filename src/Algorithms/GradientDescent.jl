"""
Implement a gradient descent optimization algorithm for KPP-Negative (maximize 
        fragmentation).


## Key Functions

`graddesc_continuation`
`graddesc_iterand!`
`graddesc_log_iteration!`
`graddesc_log_result!`


##  Key Structures

`graddesc_iterand`: the GraphOptimizationIterators.Iterand object used inside of 
    heuristic optimization to implement greedy optimization. This object defines 
    steps to take inside each iteration, ideal state swaps, stopping conditions, 
    acceptance thresholds, and more.
"""


"""
Define conditions under when to break iteration

# Constructs

```
graddesc_continuation(
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
function graddesc_continuation(
    ip::IteratorParameters,
    op::OptimizationParameters
)::Bool
    out = ((ip.i < op.max_iter) & ip.cont)
    return out
end



"""
Iterand function for greedy optimization of graph fragmentation

# Constructs

```
graddesc_iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments
- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function graddesc_iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
    
    obj_best_inner_iter = 0 # UPDATE
    S_f_inner_iter = copy(params_optimization.S)
    S_eval_out = copy(params_optimization.S)
    # can copy S_eval from params_optimizaion.S as well
 
    # need to add while loop here
    for (ind, s_in) in enumerate(params_optimization.S)

        for s_out in params_optimization.S_comp
            
            #S_eval_out = union(Set(S_eval), Set([s_out]))
            #S_eval_out = sort(collect(S_eval_out))
            for (k, v) in enumerate(params_optimization.S) 
                @inbounds S_eval_out[k] = (k != ind) ? v : s_out
            end
            sort!(S_eval_out)

            S_eval_out_tup = Tuple(S_eval_out)

            obj_try = get(
                params_iterator.dict_optional_use, 
                S_eval_out_tup, 
                nothing
            )

            if !isa(obj_try, Nothing)
                if obj_try > obj_best_inner_iter
                    #S_f_inner_iter .= S_eval_out
                    copyto!(S_f_inner_iter, S_eval_out)
                    obj_best_inner_iter = obj_try
                end

                continue
            end

            # recalculate distances
            obj_try = get_objective_from_removed_vertices(
                params_optimization, 
                S_eval_out
            )

            # add to dictionary of explored values and update current maximum fragmentation/value
            params_iterator.dict_optional_use[S_eval_out_tup] = obj_try
            if obj_try > obj_best_inner_iter
                #S_f_inner_iter .= S_eval_out
                copyto!(S_f_inner_iter, S_eval_out)
                obj_best_inner_iter = obj_try
            end
        end
    end

 
    if obj_best_inner_iter > params_iterator.obj
        msg = "\n\n** Updating iteration - F = $(params_iterator.obj), F_0 = $(params_iterator.obj_0), obj_best_inner_iter = $(obj_best_inner_iter)\n"
        
        update_accepted_swap_params!(
            params_iterator,
            params_optimization,
            obj_best_inner_iter,
            S_f_inner_iter;
            msg = msg,
        )

    else 
        params_iterator.cont = false
    end
end



"""
Iteration logging function (within iteration)

# Constructs

```
graddesc_log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function graddesc_log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("$(params_iterator.i) iterations complete with value $(params_iterator.obj)")
end



"""
Iteration logging function (after iteration)

# Constructs

```
graddesc_log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function graddesc_log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("Greedy optimization complete in $(params_iterator.i) iterations.") 
end



"""
Iterand for the greedy optimization approach
"""
graddesc_iterand = Iterand(
    graddesc_continuation,
    graddesc_iterand!;
    log_iteration! = graddesc_log_iteration!,
    log_result! = graddesc_log_result!,
)
