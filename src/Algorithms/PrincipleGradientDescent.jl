"""
Implement a greedy optimization algorithm for KPP-Negative (maximize 
        fragmentation).


## Key Functions

`pgraddesc_continuation`
`pgraddesc_iterand!`
`pgraddesc_log_iteration!`
`pgraddesc_log_result!`


##  Key Structures

`pgraddesc_iterand`: the GraphOptimizationIterators.Iterand object used inside of 
    heuristic optimization to implement greedy optimization. This object defines 
    steps to take inside each iteration, ideal state swaps, stopping conditions, 
    acceptance thresholds, and more.
"""


"""
Define conditions under when to break iteration

# Constructs

```
pgraddesc_continuation(
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
function pgraddesc_continuation(
    ip::IteratorParameters,
    op::OptimizationParameters
)::Bool
    out = ((ip.i < op.max_iter) & ip.cont)
    return out
end


"""
Initialize/fill a gradiant vector
"""
function pgraddesc_set_gradient!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters;
    key_vec_grad::Symbol = :vec_grad,
)
    vec = get(
        params_iterator.dict_optional_use, 
        key_vec_grad,
        nothing
    )

    if isa(vec, Nothing)
        vec = zeros(Float64, params_optimization.graph_wrapper.dims[1])
        params_iterator.dict_optional_use[key_vec_grad] = vec
    end

    fill!(vec, 0)

    return nothing
end


"""
Initialize/fill a sign vector
"""
function pgraddesc_set_sign!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters;
    key_vec_sign::Symbol = :vec_sign,
)
    vec = get(
        params_iterator.dict_optional_use, 
        key_vec_sign,
        nothing
    )

    # initialize as empty except for S
    if isa(vec, Nothing)
        vec = -1 .* ones(Int64, params_optimization.graph_wrapper.dims[1])
        vec[params_optimization.S] .= 1
        
        params_iterator.dict_optional_use[key_vec_sign] = vec
    end

    #fill!(params_iterator.dict_optional_use[key_vec_sign], -1)
    #params_iterator.dict_optional_use[key_vec_sign][
    #    params_iterator.S
    #] .= 1
    
    return nothing
end


"""
Iterand function for greedy optimization of graph fragmentation

# Constructs

```
pgraddesc_iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments
- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function pgraddesc_iterand!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
    
    # 1. implement a gradient vector that stores the grad
    # 2. 

    pgraddesc_set_gradient!(
        params_iterator,
        params_optimization,
    )

    pgraddesc_set_sign!(
        params_iterator,
        params_optimization,
    )

    vec_grad = params_iterator.dict_optional_use[:vec_grad]
    vec_sign = params_iterator.dict_optional_use[:vec_sign]

    # positive and negative sets from PGD description
    s_neg = Vector{Tuple{Float64, Int64, Float64}}()
    s_pos = Vector{Tuple{Float64, Int64, Float64}}()

    # get a scalar based on whether or not you're maximizing or minimizing
    scal = (params_optimization.objective_direction == :maximize) ? 1 : -1
    

    # calcualte gradient for in-set changes
    for v in params_optimization.S

        S_try = sort(setdiff(params_optimization.S, [v]))

        # recalculate distances
        obj_try = get_objective_from_removed_vertices(
            params_optimization, 
            S_try,
        )

        # calculate gradient and push
        grad = scal*(params_iterator.obj - obj_try)/2
        vec_grad[v] = grad
        (grad < 0) && push!(s_neg, (grad, v, obj_try))
    end


    # calcualte gradient for out-of-set changes
    for v in params_optimization.S_comp

        S_try = sort(union(params_optimization.S, [v]))

        # recalculate distances
        obj_try = get_objective_from_removed_vertices(
            params_optimization, 
            S_try,
        )

        grad = scal*(obj_try - params_iterator.obj)/2
        vec_grad[v] = grad
        (grad > 0) && push!(s_pos, (grad, v, obj_try))
    end


    # sort each by absolute value and perform swaps if needed
    sort!(s_neg; by = (x -> abs(x[1])), rev = true, )
    sort!(s_pos; by = (x -> abs(x[1])), rev = true, )
    n_s = min(length(s_neg), length(s_pos))
    println("n_s = $(n_s)")
    #println(s_neg)
    #println("")
    #println(s_pos)
    #println("\n\n")

    if n_s > 0
        
        S_cur = copy(params_optimization.S)

        for k in 1:n_s
            v_neg = s_neg[k][2]
            v_pos = s_pos[k][2]

            vec_sign[v_neg] *= -1
            vec_sign[v_pos] *= -1

            # swap values
            S_cur[S_cur .== v_neg] .= v_pos
        end
        
        sort!(S_cur)
        obj_cur = get_objective_from_removed_vertices(
            params_optimization, 
            S_cur,
        )
        
        #msg = "\n\n** keeping swap at iteration $(params_iterator.i) - F = $(params_iterator.obj), F_0 = $(params_iterator.obj_0), F_try = $(obj_try)\n"

        update_accepted_swap_params!(
            params_iterator,
            params_optimization,
            obj_cur,
            S_cur;
            #msg = msg,
        )

    else
        # stop if there are no actions to be taken
        params_iterator.cont = false
    end
    
end



"""
Iteration logging function (within iteration)

# Constructs

```
pgraddesc_log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function pgraddesc_log_iteration!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("$(params_iterator.i) iterations complete with value $(params_iterator.obj)")
end



"""
Iteration logging function (after iteration)

# Constructs

```
pgraddesc_log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)
```

##  Function Arguments

- `params_iterator`: parameters used in iteration
- `params_optimization`: parameters used to set up the algorithm run
"""
function pgraddesc_log_result!(
    params_iterator::IteratorParameters,
    params_optimization::OptimizationParameters,
)

    @info("Principle gradient descent optimization complete in $(params_iterator.i) iterations.") 
end



"""
Iterand for the greedy optimization approach
"""
pgraddesc_iterand = Iterand(
    pgraddesc_continuation,
    pgraddesc_iterand!;
    log_iteration! = pgraddesc_log_iteration!,
    log_result! = pgraddesc_log_result!,
    opts_prefix = :pgraddesc,
)
