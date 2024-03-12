var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API reference","title":"API reference","text":"CurrentModule = DifferentiationInterface\nCollapsedDocStrings = true","category":"page"},{"location":"api/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"DifferentiationInterface","category":"page"},{"location":"api/#DifferentiationInterface.DifferentiationInterface","page":"API reference","title":"DifferentiationInterface.DifferentiationInterface","text":"DifferentiationInterface\n\nAn interface to various automatic differentiation backends in Julia.\n\nExports\n\nderivative\ngradient\ngradient!\njacobian\njacobian!\nmultiderivative\nmultiderivative!\nprepare_derivative\nprepare_gradient\nprepare_jacobian\nprepare_multiderivative\nprepare_pullback\nprepare_pushforward\npullback\npullback!\npushforward\npushforward!\nvalue_and_derivative\nvalue_and_gradient\nvalue_and_gradient!\nvalue_and_jacobian\nvalue_and_jacobian!\nvalue_and_multiderivative\nvalue_and_multiderivative!\nvalue_and_pullback\nvalue_and_pullback!\nvalue_and_pushforward\nvalue_and_pushforward!\n\n\n\n\n\n","category":"module"},{"location":"api/#Utilities","page":"API reference","title":"Utilities","text":"","category":"section"},{"location":"api/#Scalar-to-scalar","page":"API reference","title":"Scalar to scalar","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [DifferentiationInterface]\nPages = [\"scalar_scalar.jl\"]","category":"page"},{"location":"api/#DifferentiationInterface.derivative-Tuple{ADTypes.AbstractADType, Any, Number, Vararg{Any}}","page":"API reference","title":"DifferentiationInterface.derivative","text":"derivative(backend, f, x, [extras]) -> der\n\nCompute the derivative der = f'(x) of a scalar-to-scalar function.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.value_and_derivative","page":"API reference","title":"DifferentiationInterface.value_and_derivative","text":"value_and_derivative(backend, f, x, [extras]) -> (y, der)\n\nCompute the primal value y = f(x) and the derivative der = f'(x) of a scalar-to-scalar function.\n\n\n\n\n\n","category":"function"},{"location":"api/#Scalar-to-array","page":"API reference","title":"Scalar to array","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [DifferentiationInterface]\nPages = [\"scalar_array.jl\"]","category":"page"},{"location":"api/#DifferentiationInterface.multiderivative!-Tuple{AbstractArray, ADTypes.AbstractADType, Any, Number, Vararg{Any}}","page":"API reference","title":"DifferentiationInterface.multiderivative!","text":"multiderivative!(multider, backend, f, x, [extras]) -> multider\n\nCompute the (array-valued) derivative multider = f'(x) of a scalar-to-array function, overwriting multider if possible.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.multiderivative-Tuple{ADTypes.AbstractADType, Any, Number, Vararg{Any}}","page":"API reference","title":"DifferentiationInterface.multiderivative","text":"multiderivative(backend, f, x, [extras]) -> multider\n\nCompute the (array-valued) derivative multider = f'(x) of a scalar-to-array function.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.value_and_multiderivative!","page":"API reference","title":"DifferentiationInterface.value_and_multiderivative!","text":"value_and_multiderivative!(multider, backend, f, x, [extras]) -> (y, multider)\n\nCompute the primal value y = f(x) and the (array-valued) derivative multider = f'(x) of a scalar-to-array function, overwriting multider if possible.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterface.value_and_multiderivative-Tuple{ADTypes.AbstractADType, Any, Number, Vararg{Any}}","page":"API reference","title":"DifferentiationInterface.value_and_multiderivative","text":"value_and_multiderivative(backend, f, x, [extras]) -> (y, multider)\n\nCompute the primal value y = f(x) and the (array-valued) derivative multider = f'(x) of a scalar-to-array function.\n\n\n\n\n\n","category":"method"},{"location":"api/#Array-to-scalar","page":"API reference","title":"Array to scalar","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [DifferentiationInterface]\nPages = [\"array_scalar.jl\"]","category":"page"},{"location":"api/#DifferentiationInterface.gradient!-Tuple{AbstractArray, ADTypes.AbstractADType, Any, AbstractArray, Vararg{Any}}","page":"API reference","title":"DifferentiationInterface.gradient!","text":"gradient!(grad, backend, f, x, [extras]) -> grad\n\nCompute the gradient grad = ∇f(x) of an array-to-scalar function, overwriting grad if possible.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.gradient-Tuple{ADTypes.AbstractADType, Any, AbstractArray, Vararg{Any}}","page":"API reference","title":"DifferentiationInterface.gradient","text":"gradient(backend, f, x, [extras]) -> grad\n\nCompute the gradient grad = ∇f(x) of an array-to-scalar function.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.value_and_gradient!","page":"API reference","title":"DifferentiationInterface.value_and_gradient!","text":"value_and_gradient!(grad, backend, f, x, [extras]) -> (y, grad)\n\nCompute the primal value y = f(x) and the gradient grad = ∇f(x) of an array-to-scalar function, overwriting grad if possible.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterface.value_and_gradient-Tuple{ADTypes.AbstractADType, Any, AbstractArray, Vararg{Any}}","page":"API reference","title":"DifferentiationInterface.value_and_gradient","text":"value_and_gradient(backend, f, x, [extras]) -> (y, grad)\n\nCompute the primal value y = f(x) and the gradient grad = ∇f(x) of an array-to-scalar function.\n\n\n\n\n\n","category":"method"},{"location":"api/#Array-to-array","page":"API reference","title":"Array to array","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [DifferentiationInterface]\nPages = [\"array_array.jl\"]","category":"page"},{"location":"api/#DifferentiationInterface.jacobian!-Tuple{AbstractMatrix, ADTypes.AbstractADType, Any, AbstractArray, Vararg{Any}}","page":"API reference","title":"DifferentiationInterface.jacobian!","text":"jacobian!(jac, backend, f, x, [extras]) -> jac\n\nCompute the Jacobian matrix jac = ∂f(x) of an array-to-array function, overwriting jac if possible.\n\nNotes\n\nRegardless of the shape of x and y, if x has length n and y has length m, then jac is expected to be a m × n matrix. This function acts as if the input and output had been flattened with vec. \n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.jacobian-Tuple{ADTypes.AbstractADType, Any, AbstractArray, Vararg{Any}}","page":"API reference","title":"DifferentiationInterface.jacobian","text":"jacobian(backend, f, x, [extras]) -> jac\n\nCompute the Jacobian matrix jac = ∂f(x) of an array-to-array function.\n\nNotes\n\nRegardless of the shape of x and y, if x has length n and y has length m, then jac is expected to be a m × n matrix. This function acts as if the input and output had been flattened with vec. \n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.value_and_jacobian!","page":"API reference","title":"DifferentiationInterface.value_and_jacobian!","text":"value_and_jacobian!(jac, backend, f, x, [extras]) -> (y, jac)\n\nCompute the primal value y = f(x) and the Jacobian matrix jac = ∂f(x) of an array-to-array function, overwriting jac if possible.\n\nNotes\n\nRegardless of the shape of x and y, if x has length n and y has length m, then jac is expected to be a m × n matrix. This function acts as if the input and output had been flattened with vec. \n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterface.value_and_jacobian-Tuple{ADTypes.AbstractADType, Any, AbstractArray, Vararg{Any}}","page":"API reference","title":"DifferentiationInterface.value_and_jacobian","text":"value_and_jacobian(backend, f, x, [extras]) -> (y, jac)\n\nCompute the primal value y = f(x) and the Jacobian matrix jac = ∂f(x) of an array-to-array function.\n\nNotes\n\nRegardless of the shape of x and y, if x has length n and y has length m, then jac is expected to be a m × n matrix. This function acts as if the input and output had been flattened with vec. \n\n\n\n\n\n","category":"method"},{"location":"api/#Primitives","page":"API reference","title":"Primitives","text":"","category":"section"},{"location":"api/#Pushforward","page":"API reference","title":"Pushforward","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [DifferentiationInterface]\nPages = [\"pushforward.jl\"]","category":"page"},{"location":"api/#DifferentiationInterface.pushforward","page":"API reference","title":"DifferentiationInterface.pushforward","text":"pushforward(backend, f, x, dx, [extras]) -> dy\n\nCompute the Jacobian-vector product dy = ∂f(x) * dx.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterface.pushforward!","page":"API reference","title":"DifferentiationInterface.pushforward!","text":"pushforward!(dy, backend, f, x, dx, [extras]) -> dy\n\nCompute the Jacobian-vector product dy = ∂f(x) * dx, overwriting dy if possible.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterface.value_and_pushforward","page":"API reference","title":"DifferentiationInterface.value_and_pushforward","text":"value_and_pushforward(backend, f, x, dx, [extras]) -> (y, dy)\n\nCompute the primal value y = f(x) and the Jacobian-vector product dy = ∂f(x) * dx.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterface.value_and_pushforward!","page":"API reference","title":"DifferentiationInterface.value_and_pushforward!","text":"value_and_pushforward!(dy, backend, f, x, dx, [extras]) -> (y, dy)\n\nCompute the primal value y = f(x) and the Jacobian-vector product dy = ∂f(x) * dx, overwriting dy if possible.\n\ninfo: Interface requirement\nThis is the only required implementation for a forward mode backend.\n\n\n\n\n\n","category":"function"},{"location":"api/#Pullback","page":"API reference","title":"Pullback","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [DifferentiationInterface]\nPages = [\"pullback.jl\"]","category":"page"},{"location":"api/#DifferentiationInterface.pullback","page":"API reference","title":"DifferentiationInterface.pullback","text":"pullback(backend, f, x, dy, [extras]) -> dx\n\nCompute the vector-Jacobian product dx = ∂f(x)' * dy.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterface.pullback!","page":"API reference","title":"DifferentiationInterface.pullback!","text":"pullback!(dx, backend, f, x, dy, [extras]) -> dx\n\nCompute the vector-Jacobian product dx = ∂f(x)' * dy, overwriting dx if possible.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterface.value_and_pullback","page":"API reference","title":"DifferentiationInterface.value_and_pullback","text":"value_and_pullback(backend, f, x, dy, [extras]) -> (y, dx)\n\nCompute the primal value y = f(x) and the vector-Jacobian product dx = ∂f(x)' * dy.\n\n\n\n\n\n","category":"function"},{"location":"api/#DifferentiationInterface.value_and_pullback!","page":"API reference","title":"DifferentiationInterface.value_and_pullback!","text":"value_and_pullback!(dx, backend, f, x, dy, [extras]) -> (y, dx)\n\nCompute the primal value y = f(x) and the vector-Jacobian product dx = ∂f(x)' * dy, overwriting dx if possible.\n\ninfo: Interface requirement\nThis is the only required implementation for a reverse mode backend.\n\n\n\n\n\n","category":"function"},{"location":"api/#Preparation","page":"API reference","title":"Preparation","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [DifferentiationInterface]\nPages = [\"prepare.jl\"]","category":"page"},{"location":"api/#DifferentiationInterface.prepare_derivative-Tuple{ADTypes.AbstractADType, Any, Number}","page":"API reference","title":"DifferentiationInterface.prepare_derivative","text":"prepare_derivative(backend, f, x) -> extras\n\nCreate an extras object that can be given to derivative operators.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.prepare_gradient-Tuple{ADTypes.AbstractADType, Any, AbstractArray}","page":"API reference","title":"DifferentiationInterface.prepare_gradient","text":"prepare_gradient(backend, f, x) -> extras\n\nCreate an extras object that can be given to gradient operators.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.prepare_jacobian-Tuple{ADTypes.AbstractADType, Any, AbstractArray}","page":"API reference","title":"DifferentiationInterface.prepare_jacobian","text":"prepare_jacobian(backend, f, x) -> extras\n\nCreate an extras object that can be given to jacobian operators.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.prepare_multiderivative-Tuple{ADTypes.AbstractADType, Any, Number}","page":"API reference","title":"DifferentiationInterface.prepare_multiderivative","text":"prepare_multiderivative(backend, f, x) -> extras\n\nCreate an extras object that can be given to multiderivative operators.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.prepare_pullback-Tuple{ADTypes.AbstractADType, Any, Any}","page":"API reference","title":"DifferentiationInterface.prepare_pullback","text":"prepare_pullback(backend, f, x) -> extras\n\nCreate an extras object that can be given to pullback operators.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.prepare_pushforward-Tuple{ADTypes.AbstractADType, Any, Any}","page":"API reference","title":"DifferentiationInterface.prepare_pushforward","text":"prepare_pushforward(backend, f, x) -> extras\n\nCreate an extras object that can be given to pushforward operators.\n\n\n\n\n\n","category":"method"},{"location":"api/#Internals","page":"API reference","title":"Internals","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"These are not part of the public API.","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [DifferentiationInterface]\nPages = [\"backends.jl\", \"mode.jl\", \"utils.jl\"]\nPublic = false","category":"page"},{"location":"api/#DifferentiationInterface.autodiff_mode-Tuple{ADTypes.AbstractForwardMode}","page":"API reference","title":"DifferentiationInterface.autodiff_mode","text":"autodiff_mode(backend)\n\nReturn ForwardMode() or ReverseMode() in a statically predictable way.\n\nThis function must be overloaded for backends that do not inherit from ADTypes.AbstractForwardMode or ADTypes.AbstractReverseMode (e.g. because they support both forward and reverse).\n\nWe classify ADTypes.AbstractFiniteDifferencesMode as forward mode.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.handles_input_type-Tuple{ADTypes.AbstractADType, Type{<:Number}}","page":"API reference","title":"DifferentiationInterface.handles_input_type","text":"handles_input_type(backend, ::Type{X})\n\nCheck if backend can differentiate functions with input type X.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.handles_output_type-Tuple{ADTypes.AbstractADType, Type{<:Number}}","page":"API reference","title":"DifferentiationInterface.handles_output_type","text":"handles_output_type(backend, ::Type{Y})\n\nCheck if backend can differentiate functions with output type Y.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.handles_types-Union{Tuple{Y}, Tuple{X}, Tuple{ADTypes.AbstractADType, Type{X}, Type{Y}}} where {X, Y}","page":"API reference","title":"DifferentiationInterface.handles_types","text":"handles_types(backend, ::Type{X}, ::Type{Y})\n\nCheck if backend can differentiate functions with input type X and output type Y.\n\n\n\n\n\n","category":"method"},{"location":"api/#DifferentiationInterface.ForwardMode","page":"API reference","title":"DifferentiationInterface.ForwardMode","text":"ForwardMode\n\nTrait identifying forward mode AD backends. Used for internal dispatch only.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterface.ReverseMode","page":"API reference","title":"DifferentiationInterface.ReverseMode","text":"ReverseMode\n\nTrait identifying reverse mode AD backends. Used for internal dispatch only.\n\n\n\n\n\n","category":"type"},{"location":"api/#DifferentiationInterface.basisarray-Tuple{ADTypes.AbstractADType, AbstractArray, Any}","page":"API reference","title":"DifferentiationInterface.basisarray","text":"basisarray(backend, a::AbstractArray, i::CartesianIndex)\n\nConstruct the i-th stardard basis array in the vector space of a with element type eltype(a).\n\nNote\n\nIf an AD backend benefits from a more specialized basis array implementation, this function can be extended on the backend type.\n\n\n\n\n\n","category":"method"},{"location":"api/#Package-extensions","page":"API reference","title":"Package extensions","text":"","category":"section"},{"location":"api/","page":"API reference","title":"API reference","text":"These are not part of the public API.","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceChainRulesCoreExt)]","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceDiffractorExt)]","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceEnzymeExt)]","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceFiniteDiffExt)]","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceForwardDiffExt)]","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfacePolyesterForwardDiffExt)]","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceReverseDiffExt)]","category":"page"},{"location":"api/","page":"API reference","title":"API reference","text":"Modules = [Base.get_extension(DifferentiationInterface, :DifferentiationInterfaceZygoteExt)]","category":"page"},{"location":"design/#Design","page":"Design","title":"Design","text":"","category":"section"},{"location":"design/","page":"Design","title":"Design","text":"The operators defined in this package are split into two main parts:","category":"page"},{"location":"design/","page":"Design","title":"Design","text":"the \"utilities\", which are sufficient for most users\nthe \"primitives\", which are mostly relevant for experts or backend developers","category":"page"},{"location":"design/#Utilities","page":"Design","title":"Utilities","text":"","category":"section"},{"location":"design/","page":"Design","title":"Design","text":"Depending on the type of input and output, differentiation operators can have various names. We choose the following terminology for the utilities we provide:","category":"page"},{"location":"design/","page":"Design","title":"Design","text":" scalar output array output\nscalar input derivative multiderivative\narray input gradient jacobian","category":"page"},{"location":"design/","page":"Design","title":"Design","text":"Most backends have custom implementations for all of these, which we reuse whenever possible.","category":"page"},{"location":"design/#Primitives","page":"Design","title":"Primitives","text":"","category":"section"},{"location":"design/","page":"Design","title":"Design","text":"Every utility can also be implemented from either of these two primitives:","category":"page"},{"location":"design/","page":"Design","title":"Design","text":"the pushforward (in forward mode), computing a Jacobian-vector product\nthe pullback (in reverse mode), computing a vector-Jacobian product","category":"page"},{"location":"design/#Variants","page":"Design","title":"Variants","text":"","category":"section"},{"location":"design/","page":"Design","title":"Design","text":"Whenever it makes sense, four variants of the same operator are defined:","category":"page"},{"location":"design/","page":"Design","title":"Design","text":" mutating non-mutating\nprimal too value_and_something!(storage, args...) value_and_something(args...)\ndifferential only something!(storage, args...) something(args...)","category":"page"},{"location":"design/","page":"Design","title":"Design","text":"Replace something with derivative, multiderivative, gradient, jacobian, pushforward or pullback to get the correct name.","category":"page"},{"location":"design/#Preparation","page":"Design","title":"Preparation","text":"","category":"section"},{"location":"design/","page":"Design","title":"Design","text":"In many cases, automatic differentiation can be accelerated if the function has been run at least once (e.g. to record a tape) and if some cache objects are provided. This is a backend-specific procedure, but we expose a common syntax to achieve it.","category":"page"},{"location":"design/","page":"Design","title":"Design","text":"If you run prepare_something(backend, f, x), it will create an object called extras containing the necessary information to speed up the something procedure and its variants. You can them call something(backend, f, x, extras), which should be faster than something(backend, f, x). This is especially worth it if you plan to call something several times in similar settings (same backend, same function). You can think of it as a warm up.","category":"page"},{"location":"design/","page":"Design","title":"Design","text":"By default, all the preparation functions return nothing. We do not make any guarantees on their implementation for each backend, or on the performance gains that can be expected.","category":"page"},{"location":"design/#Backend-requirements","page":"Design","title":"Backend requirements","text":"","category":"section"},{"location":"design/","page":"Design","title":"Design","text":"The only requirement for a backend is to implement either value_and_pushforward! or value_and_pullback!, from which the rest of the operators can be deduced. We provide a standard series of fallbacks, but we leave it to each backend to redefine as many of the utilities as necessary to achieve optimal performance.","category":"page"},{"location":"design/","page":"Design","title":"Design","text":"Every backend we support corresponds to a package extension of DifferentiationInterface.jl (located in the ext subfolder). Advanced users are welcome to code more backends and submit pull requests!","category":"page"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"https://github.com/gdalle/DifferentiationInterface.jl/blob/main/README.md\"","category":"page"},{"location":"#DifferentiationInterface","page":"Home","title":"DifferentiationInterface","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Dev) (Image: Build Status) (Image: Coverage) (Image: Code Style: Blue)","category":"page"},{"location":"","page":"Home","title":"Home","text":"An interface to various automatic differentiation backends in Julia.","category":"page"},{"location":"#Goal","page":"Home","title":"Goal","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides a backend-agnostic syntax to differentiate functions f(x) = y, where x and y are either real numbers or abstract arrays.","category":"page"},{"location":"","page":"Home","title":"Home","text":"It supports in-place versions of every operator, and ensures type stability whenever possible.","category":"page"},{"location":"#Compatibility","page":"Home","title":"Compatibility","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"We support some of the backends defined by ADTypes.jl:","category":"page"},{"location":"","page":"Home","title":"Home","text":"ChainRulesCore.jl with AutoChainRules(ruleconfig)\nDiffractor.jl with AutoDiffractor()\nEnzyme.jl with AutoEnzyme(Val(:forward)) or AutoEnzyme(Val(:reverse))\nFiniteDiff.jl with AutoFiniteDiff()\nForwardDiff.jl with AutoForwardDiff()\nPolyesterForwardDiff.jl with AutoPolyesterForwardDiff(; chunksize=C)\nReverseDiff.jl with AutoReverseDiff()\nZygote.jl with AutoZygote()","category":"page"},{"location":"#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Setup:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> import DifferentiationInterface, ADTypes, ForwardDiff\n\njulia> backend = ADTypes.AutoForwardDiff();\n\njulia> f(x) = sum(abs2, x);","category":"page"},{"location":"","page":"Home","title":"Home","text":"Out-of-place gradient:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> DifferentiationInterface.value_and_gradient(backend, f, [1., 2., 3.])\n(14.0, [2.0, 4.0, 6.0])","category":"page"},{"location":"","page":"Home","title":"Home","text":"In-place gradient:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> grad = zeros(3);\n\njulia> DifferentiationInterface.value_and_gradient!(grad, backend, f, [1., 2., 3.])\n(14.0, [2.0, 4.0, 6.0])\n\njulia> grad\n3-element Vector{Float64}:\n 2.0\n 4.0\n 6.0","category":"page"},{"location":"#Related-packages","page":"Home","title":"Related packages","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"AbstractDifferentiation.jl is the original inspiration for DifferentiationInterface.jl. We aim to be less generic (one input, one output, first order only) but more efficient (type stability, memory reuse).\nAutoDiffOperators.jl is an attempt to bridge ADTypes.jl with AbstractDifferentiation.jl. We provide similar functionality (except for the matrix-like behavior) but cover more backends.","category":"page"},{"location":"#Roadmap","page":"Home","title":"Roadmap","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Goals for future releases:","category":"page"},{"location":"","page":"Home","title":"Home","text":"implement backend-specific cache objects\nsupport in-place functions f!(y, x)\ndefine user-facing functions to test and benchmark backends against each other","category":"page"}]
}
