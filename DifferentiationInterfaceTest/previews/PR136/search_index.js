var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API reference","title":"API reference","text":"CurrentModule = Main\nCollapsedDocStrings = true","category":"page"},{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"DifferentiationInterfaceTest","category":"page"},{"location":"api/#DifferentiationInterfaceTest","page":"API reference","title":"DifferentiationInterfaceTest","text":"DifferentiationInterfaceTest\n\nTesting and benchmarking utilities for automatic differentiation in Julia.\n\nExports\n\nAbstractScenario\nDerivativeScenario\nGradientScenario\nHVPScenario\nHessianScenario\nJacobianScenario\nPullbackScenario\nPushforwardScenario\nSecondDerivativeScenario\nbenchmark_differentiation\ncomponent_scenarios\ndefault_scenarios\ngpu_scenarios\nstatic_scenarios\ntest_differentiation\n\n\n\n\n\n","category":"module"},{"location":"api/#Entry-points","page":"API reference","title":"Entry points","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"test_differentiation\nbenchmark_differentiation","category":"page"},{"location":"api/#DifferentiationInterfaceTest.test_differentiation","page":"API reference","title":"DifferentiationInterfaceTest.test_differentiation","text":"test_differentiation(\n    backends::Vector{<:ADTypes.AbstractADType};\n    ...\n)\ntest_differentiation(\n    backends::Vector{<:ADTypes.AbstractADType},\n    scenarios::Vector{<:AbstractScenario};\n    correctness,\n    type_stability,\n    call_count,\n    sparsity,\n    detailed,\n    input_type,\n    output_type,\n    allocating,\n    mutating,\n    first_order,\n    second_order,\n    excluded,\n    logging,\n    isapprox,\n    atol,\n    rtol\n)\n\n\nCross-test a list of backends on a list of scenarios, running a variety of different tests.\n\nDefault arguments\n\nscenarios::Vector{<:AbstractScenario}: the output of default_scenarios()\n\nKeyword arguments\n\nTesting:\n\ncorrectness=true: whether to compare the differentiation results with the theoretical values specified in each scenario\nIf a backend object like correctness=AutoForwardDiff() is passed instead of a boolean, the results will be compared using that reference backend as the ground truth.\nOtherwise, the scenario-specific reference operator will be used as the ground truth instead, see AbstractScenario for details.\ncall_count=false: whether to check that the function is called the right number of times\ntype_stability=false: whether to check type stability with JET.jl (thanks to @test_opt)\nsparsity: whether to check sparsity of the jacobian / hessian\ndetailed=false: whether to print a detailed or condensed test log\n\nFiltering:\n\ninput_type=Any: restrict scenario inputs to subtypes of this\noutput_type=Any: restrict scenario outputs to subtypes of this\nallocating=true: consider operators for allocating functions\nmutating=true: consider operators for mutating functions\nfirst_order=true: consider first order operators\nsecond_order=true: consider second order operators\n\nOptions:\n\nlogging=false: whether to log progress\nisapprox=isapprox: function used to compare objects, with the standard signature isapprox(x, y; atol, rtol)\natol=0: absolute precision for correctness testing (when comparing to the reference outputs)\nrtol=1e-3: relative precision for correctness testing (when comparing to the reference outputs)\n\n\n\n\n\ntest_differentiation(\n    backend::ADTypes.AbstractADType,\n    args...;\n    kwargs...\n)\n\n\nShortcut for a single backend.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.benchmark_differentiation","page":"API reference","title":"DifferentiationInterfaceTest.benchmark_differentiation","text":"benchmark_differentiation(\n    backends::Vector{<:ADTypes.AbstractADType};\n    ...\n) -> Vector{DifferentiationInterfaceTest.BenchmarkDataRow}\nbenchmark_differentiation(\n    backends::Vector{<:ADTypes.AbstractADType},\n    scenarios::Vector{<:AbstractScenario};\n    input_type,\n    output_type,\n    allocating,\n    mutating,\n    first_order,\n    second_order,\n    excluded,\n    logging\n) -> Vector{DifferentiationInterfaceTest.BenchmarkDataRow}\n\n\nBenchmark a list of backends for a list of operators on a list of scenarios.\n\nKeyword arguments\n\nfiltering: same as test_differentiation for the filtering part.\nlogging=false: whether to log progress\n\n\n\n\n\n","category":"function"},{"location":"api/#Pre-made-scenario-lists","page":"API reference","title":"Pre-made scenario lists","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"default_scenarios\ncomponent_scenarios\ngpu_scenarios\nstatic_scenarios","category":"page"},{"location":"api/#DifferentiationInterfaceTest.default_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.default_scenarios","text":"default_scenarios()\n\nCreate a vector of AbstractScenarios with standard array types.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.component_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.component_scenarios","text":"component_scenarios()\n\nCreate a vector of AbstractScenarios with component array types from ComponentArrays.jl.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.gpu_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.gpu_scenarios","text":"gpu_scenarios()\n\nCreate a vector of AbstractScenarios with GPU array types from JLArrays.jl.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterfaceTest.static_scenarios","page":"API reference","title":"DifferentiationInterfaceTest.static_scenarios","text":"static_scenarios()\n\nCreate a vector of AbstractScenarios with static array types from StaticArrays.jl.\n\n\n\n\n\n","category":"function"},{"location":"api/#Scenario-types","page":"API reference","title":"Scenario types","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"AbstractScenario\nPushforwardScenario\nPullbackScenario\nDerivativeScenario\nGradientScenario\nJacobianScenario\nSecondDerivativeScenario\nHVPScenario\nHessianScenario","category":"page"},{"location":"api/#DifferentiationInterfaceTest.AbstractScenario","page":"API reference","title":"DifferentiationInterfaceTest.AbstractScenario","text":"AbstractScenario\n\nStore a testing scenario composed of a function and its input + output.\n\nThis abstract type should never be used directly: construct one of the subtypes corresponding to the operator you want to test.\n\nSubtypes\n\nPushforwardScenario\nPullbackScenario\nDerivativeScenario\nGradientScenario\nJacobianScenario\nSecondDerivativeScenario\nHVPScenario\nHessianScenario\n\nFields\n\nAll subtypes have the following fields:\n\nf: function to apply\nx: primal input\ny: primal output\nref: reference to compare against\n\nIn addition, some subtypes contain an additional seed (dx or dy).\n\nConstructor\n\nWhen no seed is needed, the constructor looks like\n\nGradientScenario(f; x, y=nothing, ref=nothing)\n\nWhen a seed is needed, the constructor looks like\n\nPushforwardScenario(f; x, y=nothing, ref=nothing, dx=nothing)\n\nIf y is provided, f is interpreted as a mutating function f!(y, x) = nothing. Otherwise, f is interpreted as an allocating function f(x) = y.\n\nThe reference ref should be a function that takes x (and a potential seed dx or dy) to return the correct object.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterfaceTest.PushforwardScenario","page":"API reference","title":"DifferentiationInterfaceTest.PushforwardScenario","text":"PushforwardScenario(f; x, y, ref, dx)\n\nSee AbstractScenario for details.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterfaceTest.PullbackScenario","page":"API reference","title":"DifferentiationInterfaceTest.PullbackScenario","text":"PullbackScenario(f; x, y, ref, dy)\n\nSee AbstractScenario for details.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterfaceTest.DerivativeScenario","page":"API reference","title":"DifferentiationInterfaceTest.DerivativeScenario","text":"DerivativeScenario(f; x, y, ref)\n\nSee AbstractScenario for details.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterfaceTest.GradientScenario","page":"API reference","title":"DifferentiationInterfaceTest.GradientScenario","text":"GradientScenario(f; x, y, ref)\n\nSee AbstractScenario for details.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterfaceTest.JacobianScenario","page":"API reference","title":"DifferentiationInterfaceTest.JacobianScenario","text":"JacobianScenario(f; x, y, ref)\n\nSee AbstractScenario for details.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterfaceTest.SecondDerivativeScenario","page":"API reference","title":"DifferentiationInterfaceTest.SecondDerivativeScenario","text":"SecondDerivativeScenario(f; x, y, ref)\n\nSee AbstractScenario for details.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterfaceTest.HVPScenario","page":"API reference","title":"DifferentiationInterfaceTest.HVPScenario","text":"HVPScenario(f; x, y, ref, dx)\n\nSee AbstractScenario for details.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterfaceTest.HessianScenario","page":"API reference","title":"DifferentiationInterfaceTest.HessianScenario","text":"HessianScenario(f; x, y, ref)\n\nSee AbstractScenario for details.\n\n\n\n\n\n","category":"type"},{"location":"api/#Internals","page":"API reference","title":"Internals","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"This is not part of the public API.","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [DifferentiationInterfaceTest]\nPublic = false","category":"page"},{"location":"api/#DifferentiationInterfaceTest.AutoZeroForward","page":"API reference","title":"DifferentiationInterfaceTest.AutoZeroForward","text":"AutoZeroForward <: ADTypes.AbstractForwardMode\n\nTrivial backend that sets all derivatives to zero. Used in testing and benchmarking.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterfaceTest.AutoZeroReverse","page":"API reference","title":"DifferentiationInterfaceTest.AutoZeroReverse","text":"AutoZeroReverse <: ADTypes.AbstractReverseMode\n\nTrivial backend that sets all derivatives to zero. Used in testing and benchmarking.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterfaceTest.BenchmarkDataRow","page":"API reference","title":"DifferentiationInterfaceTest.BenchmarkDataRow","text":"BenchmarkDataRow\n\nAd-hoc storage type for differentiation benchmarking results. If you have a vector rows::Vector{BenchmarkDataRow}, you can turn it into a DataFrame as follows:\n\ndf = DataFrames.DataFrame(rows)\n\nFields\n\nThese are not part of the public API.\n\nbackend::String\nmode::Type\nscenario::Symbol\noperator::Symbol\nfunc::Symbol\nmutating::Bool\ninput_type::Type\noutput_type::Type\ninput_size::Tuple\noutput_size::Tuple\nsamples::Int64\ntime::Float64\nbytes::Float64\nallocs::Float64\ncompile_fraction::Float64\ngc_fraction::Float64\nevals::Float64\n\n\n\n\n\n","category":"type"},{"location":"#DifferentiationInterfaceTest","page":"Home","title":"DifferentiationInterfaceTest","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Dev) (Image: Build Status) (Image: Code Style: Blue)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Testing and benchmarking utilities for automatic differentiation (AD) in Julia, based on DifferentiationInterface.","category":"page"}]
}
