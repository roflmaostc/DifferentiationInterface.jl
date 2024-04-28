"""
    Chunked{C}

Wrapper for a chunk of `C` values that will be handled in parallel.

Can be used to wrap several seeds in variants of [`pushforward`](@ref) and [`pullback`](@ref) (not [`hvp`](@ref) at the moment).

# Fields

- ` values::NTuple{C,V}`

# Constructors

    Chunked(values)
    Chunked(val1, val2, ..., valC)
"""
struct Chunked{C,V}
    values::NTuple{C,V}
end

Chunked(values...) = Chunked(values)

#=
This heuristic is taken from ForwardDiff.jl.
Source file: https://github.com/JuliaDiff/ForwardDiff.jl/blob/master/src/prelude.jl
=#

const DEFAULT_CHUNK_SIZE = 8

"""
pick_chunk_size(input_length)

Pick a reasonable chunk size for chunked derivative evaluation with an input of length `input_length`.
    
The result cannot be larger than `DEFAULT_CHUNK_SIZE=$DEFAULT_CHUNK_SIZE`.
"""
function pick_chunk_size(input_length::Integer; threshold::Integer=DEFAULT_CHUNK_SIZE)
    if input_length <= threshold
        return input_length
    else
        nchunks = round(Int, input_length / threshold, RoundUp)
        return round(Int, input_length / nchunks, RoundUp)
    end
end
