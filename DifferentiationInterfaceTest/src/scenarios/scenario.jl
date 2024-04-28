"""
    AbstractScenario

Store a testing scenario composed of a function and its input + output.

This abstract type should never be used directly: construct one of the subtypes corresponding to the operator you want to test.
    
# Subtypes

- [`PushforwardScenario`](@ref)
- [`PullbackScenario`](@ref)
- [`DerivativeScenario`](@ref)
- [`GradientScenario`](@ref)
- [`JacobianScenario`](@ref)
- [`SecondDerivativeScenario`](@ref)
- [`HVPScenario`](@ref)
- [`HessianScenario`](@ref)
    
# Fields

All subtypes have the following fields:

- `f`: function to apply
- `x`: primal input
- `y`: primal output
- `ref`: reference to compare against
- (optional) `dx` or `dy`: tangent for `pushforward` / `pullback` / `hvp`

# Constructor

If `y` is provided, `f` is interpreted as a 2-argument function `f!(y, x) = nothing`.
Otherwise, `f` is interpreted as an 1-argument function `f(x) = y`.

The reference keyword `ref` should be a function that takes `x` (and a potential `tangent`) to return the correct object.

The operator behavior keyword `operator` should be either `:inplace` or `:outofplace` depending on what must be tested.
"""
abstract type AbstractScenario{args,op,F,X,Y,R} end

abstract type AbstractFirstOrderScenario{args,op,F,X,Y,R} <:
              AbstractScenario{args,op,F,X,Y,R} end
abstract type AbstractSecondOrderScenario{args,op,F,X,Y,R} <:
              AbstractScenario{args,op,F,X,Y,R} end

scen_type(scenario::AbstractScenario) = nameof(typeof(scenario))
nb_args(::AbstractScenario{args}) where {args} = args
operator_place(::AbstractScenario{args,op}) where {args,op} = op

function compatible(backend::AbstractADType, scen::AbstractScenario)
    if nb_args(scen) == 2
        return Bool(mutation_support(backend))
    end
    return true
end

function group_by_scen_type(scenarios)
    return Dict(
        st => filter(s -> scen_type(s) == st, scenarios) for
        st in unique(scen_type.(scenarios))
    )
end

function Base.string(scen::S) where {args,op,F,X,Y,S<:AbstractScenario{args,op,F,X,Y}}
    return "$(S.name.name){$args,$op} $(string(scen.f)) : $X -> $Y"
end

## Struct definitions

"""
    PushforwardScenario(f; x, y, dx, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct PushforwardScenario{args,op,F,X,Y,T,R} <: AbstractFirstOrderScenario{args,op,F,X,Y,R}
    f::F
    x::X
    y::Y
    dx::T
    ref::R
end

"""
    PullbackScenario(f; x, y, dy, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct PullbackScenario{args,op,F,X,Y,T,R} <: AbstractFirstOrderScenario{args,op,F,X,Y,R}
    f::F
    x::X
    y::Y
    dy::T
    ref::R
end

"""
    DerivativeScenario(f; x, y, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct DerivativeScenario{args,op,F,X<:Number,Y,R} <:
       AbstractFirstOrderScenario{args,op,F,X,Y,R}
    f::F
    x::X
    y::Y
    ref::R
end

"""
    GradientScenario(f; x, y, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct GradientScenario{args,op,F,X,Y<:Number,R} <:
       AbstractFirstOrderScenario{args,op,F,X,Y,R}
    f::F
    x::X
    y::Y
    ref::R
end

"""
    JacobianScenario(f; x, y, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct JacobianScenario{args,op,F,X<:AbstractArray,Y<:AbstractArray,R} <:
       AbstractFirstOrderScenario{args,op,F,X,Y,R}
    f::F
    x::X
    y::Y
    ref::R
end

"""
    SecondDerivativeScenario(f; x, y, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct SecondDerivativeScenario{args,op,F,X<:Number,Y,R} <:
       AbstractSecondOrderScenario{args,op,F,X,Y,R}
    f::F
    x::X
    y::Y
    ref::R
end

"""
    HVPScenario(f; x, y, dx, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct HVPScenario{args,op,F,X,Y<:Number,T,R} <:
       AbstractSecondOrderScenario{args,op,F,X,Y,R}
    f::F
    x::X
    y::Y
    dx::T
    ref::R
end

"""
    HessianScenario(f; x, y, ref, operator)

See [`AbstractScenario`](@ref) for details.
"""
struct HessianScenario{args,op,F,X<:AbstractArray,Y<:Number,R} <:
       AbstractSecondOrderScenario{args,op,F,X,Y,R}
    f::F
    x::X
    y::Y
    ref::R
end

## Constructors

for S in (
    :DerivativeScenario,
    :GradientScenario,
    :JacobianScenario,
    :SecondDerivativeScenario,
    :HessianScenario,
)
    @eval begin
        function $S(f::F; x::X, y=nothing, ref::R=nothing, operator=:inplace) where {F,X,R}
            args = isnothing(y) ? 1 : 2
            if args == 2
                f(y, x)
            else
                y = f(x)
            end
            return ($S){args,operator,F,X,typeof(y),R}(f, x, y, ref)
        end

        function change_function(s::($S), f)
            return ($S)(
                f;
                x=s.x,
                y=(nb_args(s) == 1 ? nothing : s.y),
                ref=s.ref,
                operator=operator_place(s),
            )
        end
    end
end

for S in (:PushforwardScenario, :HVPScenario)
    @eval begin
        function $S(
            f::F;
            x::X,
            y=nothing,
            ref::R=nothing,
            dx=nothing,
            operator=:inplace,
            chunk_size=nothing,
        ) where {F,X,R}
            args = isnothing(y) ? 1 : 2
            if args == 2
                f(y, x)
            else
                y = f(x)
            end
            if isnothing(dx)
                dx = if isnothing(chunk_size)
                    mysimilar_random(x)
                else
                    Chunked(ntuple(k -> mysimilar_random(x), chunk_size))
                end
            end
            return ($S){args,operator,F,X,typeof(y),typeof(dx),R}(f, x, y, dx, ref)
        end

        function change_function(s::($S), f)
            return ($S)(
                f;
                x=s.x,
                y=(nb_args(s) == 1 ? nothing : s.y),
                ref=s.ref,
                dx=s.dx,
                operator=operator_place(s),
            )
        end
    end
end

for S in (:PullbackScenario,)
    @eval begin
        function $S(
            f::F;
            x::X,
            y=nothing,
            ref::R=nothing,
            dy=nothing,
            operator=:inplace,
            chunk_size=nothing,
        ) where {F,X,R}
            args = isnothing(y) ? 1 : 2
            if args == 2
                f(y, x)
            else
                y = f(x)
            end
            if isnothing(dy)
                dy = if isnothing(chunk_size)
                    mysimilar_random(y)
                else
                    Chunked(ntuple(k -> mysimilar_random(y), chunk_size))
                end
            end
            return ($S){args,operator,F,X,typeof(y),typeof(dy),R}(f, x, y, dy, ref)
        end

        function change_function(s::($S), f)
            return ($S)(
                f;
                x=s.x,
                y=(nb_args(s) == 1 ? nothing : s.y),
                ref=s.ref,
                dy=s.dy,
                operator=operator_place(s),
            )
        end
    end
end
