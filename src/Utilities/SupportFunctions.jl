#module SupportFunctions

# load required modules

#=
# export functions and structures
export average_degree
export BoolOrNoth
export bound_value
export build_dict
export build_dict_inverse
export check_kwargs
export check_path
export check_valid_values!
export FloatOrNoth
export get_crude_percentiles
export null_func
export parse_config
export print_valid_values
export printnames
export rand_sample
export random_unif_from_float_or_noth
export read_csv
export rm_edges_by_vertex!
export str_replace
export str_with_linewidth
export try_parse_float
export tuple_mean
export VecOrNoth
export VMS
export write_csv_iterative!
export zip_cols
=#



##########################
###                    ###
###    BUILD MODULE    ###
###                    ###
##########################


###############
#    TYPES    #
###############

# some types
BoolOrNoth = Union{Bool, Nothing}
FloatOrNoth = Union{Float64, Nothing}
VecOrNoth{T} = Union{Vector{T}, Nothing}
VMOrNoth{T} = Union{Vector{T}, Matrix{T}, Nothing}



###################
#    FUNCTIONS    #
###################

"""
Return the average degree, |E|/|V|, of a graph
"""
function average_degree(
    graph::SimpleGraph,
)
    out = ne(graph)/nv(graph)
    return out
end



"""
Bound values in a vector or array between a maximum or minimum.

Bounds can be specified as:

    * A single tuple (applied to the entire object);
    * An array of tuples of the same shape (applied element wise); or
    * An axis-wise vector of bounds applied to an array.


# Constructs

```
bound_value(
    x::Float64,
    bounds::Tuple{Float64, Float64} = (0.0, 1.0)
)
```

```
bound_value(
    x::Int64,
    bounds::Tuple{Float64, Float64} = (0.0, 1.0)
)
```

```
bound_value(
    array_vals::Array,
    bounds::Tuple{Float64, Float64} = (0.0, 1.0)
)
```

```
bound_value(
    array_vals::Array,
    bounds::Vector{Tuple{Float64, Float64}},
    axis::Int = 1
)
```


# Function Arguments

- `x`, `array_vals`: values to bound
- `bounds`: a tuple of form (min, max) to apply. When using an array and a vector of tuples, the vector of bounds must have the same length as the array's axis (in Vectors, this is the length of the vector)

# Keyword Arguments

- `axis`: `1` to apply to column vectors or `0` to apply to each row
"""
function bound_value(
    x::Float64,
    bounds::Tuple{Float64, Float64} = (0.0, 1.0)
)

    b0 = minimum(bounds)
    b1 = maximum(bounds)

    return minimum([maximum([b0, x]), b1])
end

function bound_value(
    x::Int64,
    bounds::Tuple{Float64, Float64} = (0.0, 1.0)
)
    return Int64(bound_value(Float64(x), bounds))
end

function bound_value(
    array_vals::Array,
    bounds::Tuple{Float64, Float64} = (0.0, 1.0)
)
    return bound_value.(array_vals, (bounds, ))
end

function bound_value(
    array_vals::Array,
    bounds::Vector{Tuple{Float64, Float64}},
    axis::Int = 1
)

    transpose_q = (axis == 1) & !isa(array_vals, Vector)
    mat_out = transpose_q ? array_vals : transpose(array_vals)

    m, n = size(mat_out)

    if length(bounds) != n
        @error("Bound vec cannot be completed with axis = $(axis). The length of bounds should be $(ncol(mat_out)). Returning original vector.")
        return array_vals
    end

    for i in 1:n
        mat_out[:, i] = bound_value(mat_out[:, i], bounds[i])
    end

    mat_out = transpose_q ? mat_out : transpose(mat_out)

    return mat_out
end



"""
Build a dictionary using rows of dictionaries (row[i] -> row[j1:jn])
"""
function build_dict(
    df_in::DataFrame;
    type_in::Union{DataType, Nothing} = nothing,
    type_out::Union{DataType, Nothing} = nothing,
    force_array_image_q::Bool = false,
    force_tuple_domain_q::Bool = false
)

    n = size(df_in)[2]
    set_type_in_q = !isa(type_in, Nothing)
    set_type_out_q = !isa(type_out, Nothing)

    # set image and domain
    image = set_type_out_q ? type_out.(df_in[:, n]) : df_in[:, n]
    if (n == 2) & !force_tuple_domain_q
        domain = set_type_in_q ? type_in.(df_in[:, 1]) : df_in[:, 1]
    else
        domain = set_type_in_q ? [tuple((type_in.(x))...) for x in eachrow(Matrix(df_in[:, 1:(n - 1)]))] : [tuple(x...) for x in eachrow(Matrix(df_in[:, 1:(n - 1)]))]
    end

    if (length(Set(domain)) == length(domain)) & !force_array_image_q
        #in two columns, simply map a to b
        dict_out = Dict(zip(domain, image))
    else
        @info("Injection not found in build_dict: mapping multivalued image as lists.")
        dict_out = Dict{eltype(domain), typeof(image)}()
        df_group = DataFrame(:domain => domain, :image => image)
        gb = groupby(df_group, [:domain]);

        for gdf in gb
            key = gdf[1, :domain]
            dict_out[key] = gdf[:, :image]
        end
    end

    return dict_out
end



"""
Invert a dictionary
"""
function build_dict_inverse(
    dict_in::Dict
)

    keys_dict = collect(keys(dict_in))
    vals_dict = collect(values(dict_in))

    if (length(Set(keys_dict)) == length(Set(vals_dict))) & (length(Set(vals_dict)) == length(vals_dict))
        return Dict([(dict_in[x], x) for x in keys_dict])
    else
        @error ("The input dictionary in build_dict_inverse is not injective and does not have an inverse.")
        return nothing
    end
end



"""
check whether or not the keyword arguments are properly specified
"""
function check_kwargs(
    func::Function; 
    kwargs...
)

    all_kwargs = Symbol.(vcat(Base.kwarg_decl.(methods(func))...))
    dict_kwargs_out = Dict{Symbol, Any}()

    for kwarg in keys(kwargs)
        if kwarg in all_kwargs
            dict_kwargs_out[kwarg] = kwargs[kwarg]
        end
    end

    return dict_kwargs_out
end



"""
Return a path if it exists and throw an error otherwise; optional 
    'create_directory_q' can be used to create a directory
"""
function check_path(
    path::String, 
    create_directory_q::Bool; 
    throw_error_q = false,
)
    ispath(path) && (return path);

    out = ""

    if create_directory_q & !occursin(".", basename(path))
        try
            mkdir(path)
            @info("Created directory '$(path)'\n")
            out = path

        catch
            @error("Creation of directory '$(path)' failed. Check for invalid characters.")
        end

        return out

    end

    # final step of failure
    msg_error = "Path '$(path)' not found."
    throw_error_q ? error(msg_error) : @error(msg_error)

    return out
end



"""
Check whether or not a value or set of values are valid

# Constructs

```
check_valid_values!(
    values::Array{String, 1},
    valid_values::Array{String, 1};
    func_str::String = "",
    throw_error_q = true
)
```

# Function Arguments
- `values`: values to check for validity (array of strings or symbols)
- `valid_values`: each element of `values` should be in `valid_values`. If not,
    returns a bool or throws and error (default)

# Keyword Arguments
- `func_str`: optional specification of function name to pass from which
    `check_valid_values!()` was called. Used in error reporting
- `throw_error_q`: throw an error? If `false`, returns a bool
"""
function check_valid_values!(
    values::Array{String, 1},
    valid_values::Array{String, 1};
    func_str::String = "",
    throw_error_q = true
)
    if !issubset(values, valid_values)
        vals_invalid = print_valid_values(setdiff(values, valid_values))
        vals_valid = print_valid_values(valid_values)
        if length(func_str) > 0
            func_str = (func_str[1] != " ") ? " "*func_str : func_str
        end

        str_error ="Error in function$(func_str): invalid values $(vals_invalid) found. Valid values are $(vals_valid)"

        if throw_error_q
            error(str_error)
        else
            @error (str_error)
            return false
        end
    else
        return true
    end
end

# some dispatches
function check_valid_values!(
    values::Array{Symbol, 1}, 
    valid_values::Array{Symbol, 1}; 
    kwargs...
)
    return check_valid_values!(
        string.(values), 
        string.(valid_values); kwargs...)
end

function check_valid_values!(
    value::String, 
    valid_values::Array{String, 1}; 
    kwargs...
)
    return check_valid_values!([value], valid_values; kwargs...)
end

function check_valid_values!(
    value::Symbol, 
    valid_values::Array{Symbol, 1}; 
    kwargs...
)
    return check_valid_values!([value], valid_values; kwargs...)
end



"""
Calculate crude percentiles of a vector `vec`. F

# Constructs

```
get_crude_percentiles(
    vec::Array{Float64, 1},
)
```

```
get_crude_percentiles(
    array::Array{Float64, 2}; 
    axis = 0,
)
```

"""
function get_crude_percentiles(
    vec::Array{Float64, 1},
)

    # add an id, sort values in ascending order
    df_sort = DataFrame(:id => 1:length(vec), :vals => vec)
    sort!(df_sort, [:vals])
    df_sort[!, :percentile] = (1:length(vec))/length(vec)
    sort!(df_sort, [:id])

    return df_sort[:, :percentile]
end

function get_crude_percentiles(
    array::Array{Float64, 2}; 
    axis = 0,
)

    sz = size(array)
    array_out = zeros(sz)

    if axis == 0
        for i in 1:sz[1]
            array_out[i, :] = get_crude_percentiles(array[i, :])
        end
    elseif axis == 1
        for i in 1:sz[2]
            array_out[:, i] = get_crude_percentiles(array[:, i])
        end
    else
        @error("Invalid axis specification '$(axis)' in get_crude_percentiles: valid speecifications are 0 and 1.")
    end

    return array_out
end
   
function get_crude_percentiles(vec::Array{Int64, 1})
    return get_crude_percentiles(Float64.(vec))
end

function get_crude_percentiles(array::Array{Int64, 2}; kwargs...)
    return get_crude_percentiles(Float64.(array); kwargs...)
end



"""
Function that returns `nothing` no matter the arguments
"""
function null_func(
    args...; 
    kwargs...
)
    return nothing
end



"""
read a configuration file and return a dictionary
"""
function parse_config(fp_config::String)

    if ispath(fp_config)
        # read lines from the file
        file_read = open(fp_config, "r")
        lines = readlines(file_read)
        close(file_read)

        # initialize the output dictionary
        dict_out = Dict{String, Any}()
        for i in 1:length(lines)
            line = lines[i]
            if length(line) > 0
                # check conditions
                if string(line[1]) != "#"
                    # drop comments
                    line = split(line, "#")[1]
                    # next, split on colon
                    line = split(line, ":")
                    key = strip(string(line[1]))
                    value = strip(string(line[2]))

                    # split on commas and convert to vector
                    value = split(value, ",")
                    # check first value
                    if tryparse(Float64, value[1]) != nothing
                        # convert to numeric
                        value = tryparse.(Float64, value)
                        value = filter(x -> (x != nothing), value)
                        # check for integer conversion
                        if round.(value) == value
                            value = Int64.(value)
                        end
                    else
                        # convert to character and return
                        value = string.(strip.(value))
                    end

                    # remove arrays and convert to boolean if applicable
                    if length(value) == 1
                        value = value[1]
                        if lowercase(string(value)) in ["false", "true"]
                            value = (lowercase(string(value)) == "true")
                        end
                    end

                    dict_out[key] = value
                end

            end
        end

        return dict_out
    else
        print("\nFile $(fp_config) not found. Please check for the configuration file.")
        return(-1)
    end
end




"""
Print the elements of list (Array) quickly 
"""
function printnames(
    list::Array{String}
)
    for nm in list
        print("$(nm)\n")
    end
end



"""
Print a list of values into a string pretty string
"""
function print_valid_values(array::Array{String, 1})
    str_final = (length(array) > 2) ? ", and " : " and "
    str_valid = join("'".*(string.(array)).*"'", ", ", str_final)
    return str_valid
end
function print_valid_values(array::Array{Int64, 1})
    str_final = (length(array) > 2) ? ", and " : " and "
    str_valid = join("'".*(string.(array)).*"'", ", ", str_final)
    return str_valid
end
function print_valid_values(array::Array{Symbol, 1})
    str_final = (length(array) > 2) ? ", and " : " and "
    str_valid = join("'".*(string.(array)).*"'", ", ", str_final)
    return str_valid
end



"""
Generate a random value x~U(0, 1) from a Float64 or Nothing input

# Constructs 

```
random_unif_from_float_or_noth(
    val::FloatOrNoth,
)
```

##  Initialization Arguments

- `val`: float or nothing input value

##  Keyword Arguments

"""
function random_unif_from_float_or_noth(
    val::FloatOrNoth;
)
    val = isa(val, Nothing) ? Random.rand() : max(min(val, 1.0), 0.0)

    return val
end 



"""
Read in a csv and convert to a dataframe
"""
function read_csv(
    path_in::String,
    header_q::Bool,
    lim::Int64=-1,
    names_vec::Array{Symbol, 1} = Array{Symbol, 1}(),
    delimiter::String = ",";
    dtypes::Union{Dict{Symbol, DataType}, Nothing} = nothing
)

    if lim == -1
        #get some data
        df_in = CSV.File(
            path_in,
            delim = delimiter,
            ignorerepeated = false,
            header = header_q,
            types = dtypes
        ) |> DataFrame
    else
        #get some data
        df_in = CSV.File(
            path_in,
            delim = delimiter,
            ignorerepeated = false,
            header = header_q;
            limit = lim,
            types = dtypes
        ) |> DataFrame
    end

    if (!header_q & (length(names_vec) == size(df_in)[2]))
        dict_rnm = Dict([x for x in zip(Symbol.(names(df_in)), names_vec)])
        rename!(df_in, dict_rnm)
    end

    return df_in
end



"""
Remove all edges associated with vertex while preserving the vertex
"""
function rm_edges_by_vertex!(
    graph::SimpleGraph,
    vertex::Int64,
)
    
    nbrs = collect(neighbors(graph, vertex))
    rem_edge!.((graph, ), (vertex, ), nbrs)
    
    return nbrs
end



"""
using a list of tuples, get the mean tuple
"""
function tuple_mean(
    vec_tups::Array,
)
    return mean.(eachrow(hcat(collect.(vec_tups)...)))
end



"""
Attempt to parse objects into a float
"""
function try_parse_float(x::String)
    tp = tryparse(Float64, x)
    return tp
end
function try_parse_float(x::Symbol)
    out = try_parse_float(String(x))
    return out
end
function try_parse_float(x::Real)
    return Float64(x)
end
function try_parse_float(x::Union{Missing, Nothing})
    return nothing
end



"""
Write a dataframe to a CSV file within an iterator

# Constructs

```
write_csv_iterative!(
    df_write::DataFrame,
    fp_write::String;
    header::Union{Vector{Symbol}, Nothing} = nothing,
    kwargs...
)
```

# Function Arguments

- `df_write`: DataFrame to write to CSV
- `fp_write`: Path of CSV file to write to


# Keywork Arguments

- `header`: vector of names passed to `df_write` to ensure proper ordering of output columns. If `nothing`, write as-is.
- `kwargs...`::


# Example

```

fp_out = "/Users/username/PATHTOFILE/out.csv"
header_iterate = nothing

for i in 1:n_iter
    df_write = func(i)

    header_iterate = write_csv_iterative!(
        df_write,
        fp_out;
        header = header_iterate
    )
end
```

"""
function write_csv_iterative!(
    df_write::DataFrame,
    fp_write::String;
    header::Union{Vector{Symbol}, Nothing} = nothing,
    kwargs...
)
    if !ispath(fp_write)
        header = isa(header, Nothing) ? Symbol.(names(df_write)) : header
        CSV.write(fp_write, df_write[:, header]; append = false, kwargs...)
    else
        header = isa(header, Nothing) ? Symbol.(names(read_csv(fp_wrte, lim = 1))) : header
        CSV.write(fp_write, df_write[:, header]; append = true, kwargs...)
    end

    return header
end



"""
zip columns of a data frame or matrix together for iteration
    """
function zip_cols(mat::DataFrame)
    return zip([mat[:, i] for i in 1:size(mat)[2]]...)
end
function zip_cols(mat::Matrix)
    return zip([mat[:, i] for i in 1:size(mat)[2]]...)
end



#############################################################
#    FUNCTIONS FOR SAMPLING AND OPTIMIZATION EXPERIMENTS    #
#############################################################



"""
Generate a random sample of `vec` of length `n`
"""
function rand_sample(
    vec::AbstractArray,
    n::Int64
)
    i = 1

    m = minimum([n, length(vec)])
    vec_out = Array{Any, 1}(zeros(m))

    while i <= m
        vec_out[i] = Random.rand(vec, 1)[1]
        vec = [x for x in vec if x != vec_out[i]]
        i += 1
    end

    return vec_out
end



"""
Replace substrings using a dictionary that maps old_substr -> new_substr
"""
function str_replace(
    str_in::String,
    dict_repl::Dict{String, String}
)
    str_out = str_in
    for k in keys(dict_repl)
        str_out = replace(str_out, k => dict_repl[k])
    end

    return str_out
end

