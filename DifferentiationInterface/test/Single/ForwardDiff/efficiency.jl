using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using DataFrames: DataFrame

@testset verbose = false "Dense efficiency" begin
    # derivative and gradient for `f(x)`

    results1 = benchmark_differentiation(
        [AutoForwardDiff()],
        default_scenarios();
        outofplace=false,
        twoarg=false,
        input_type=Union{Number,AbstractVector},
        output_type=Number,
        second_order=false,
        excluded=[PullbackScenario],
        logging=get(ENV, "CI", "false") == "false",
    )

    # derivative and jacobian for f!(x, y)

    results2 = benchmark_differentiation(
        [AutoForwardDiff()],
        default_scenarios();
        outofplace=false,
        onearg=false,
        input_type=Union{Number,AbstractVector},
        output_type=AbstractVector,
        second_order=false,
        excluded=[PullbackScenario],
        logging=get(ENV, "CI", "false") == "false",
    )

    data = vcat(DataFrame(results1), DataFrame(results2))

    useless_rows =
        startswith.(string.(data[!, :operator]), Ref("prepare")) .|
        startswith.(string.(data[!, :operator]), Ref("value_and"))

    useful_data = data[.!useless_rows, :]

    for row in eachrow(useful_data)
        scen = row[:scenario]
        @testset "$(row[:operator]) - $(string(scen.f)) : $(typeof(scen.x)) -> $(typeof(scen.y))" begin
            @test row[:allocs] == 0
        end
    end
end

@testset verbose = false "Sparse efficiency" begin
    # sparse jacobian for f!(x, y)

    results1 = benchmark_differentiation(
        [MyAutoSparse(AutoForwardDiff(; chunksize=1);)],
        sparse_scenarios();
        input_type=AbstractVector,
        output_type=AbstractVector,
        outofplace=false,
        onearg=false,
        second_order=false,
        logging=get(ENV, "CI", "false") == "false",
    )

    data = vcat(DataFrame(results1))

    useless_rows = startswith.(string.(data[!, :operator]), Ref("prepare"))

    useful_data = data[.!useless_rows, :]

    for row in eachrow(useful_data)
        scen = row[:scenario]
        @testset "$(row[:operator]) - $(string(scen.f)) : $(typeof(scen.x)) -> $(typeof(scen.y))" begin
            @test row[:allocs] == 0
            @test row[:calls] < prod(size(scen.x))
        end
    end
end