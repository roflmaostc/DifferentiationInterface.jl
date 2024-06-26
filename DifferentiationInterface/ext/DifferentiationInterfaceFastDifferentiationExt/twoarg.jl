## Pushforward

struct FastDifferentiationTwoArgPushforwardExtras{E1,E2} <: PushforwardExtras
    jvp_exe::E1
    jvp_exe!::E2
end

function DI.prepare_pushforward(f!, y, ::AutoFastDifferentiation, x, dx)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = x_var isa Number ? monovec(x_var) : vec(x_var)
    y_vec_var = vec(y_var)
    jv_vec_var, v_vec_var = jacobian_times_v(y_vec_var, x_vec_var)
    jvp_exe = make_function(jv_vec_var, vcat(x_vec_var, v_vec_var); in_place=false)
    jvp_exe! = make_function(jv_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)
    return FastDifferentiationTwoArgPushforwardExtras(jvp_exe, jvp_exe!)
end

function DI.value_and_pushforward(
    f!,
    y,
    ::AutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationTwoArgPushforwardExtras,
)
    f!(y, x)
    v_vec = vcat(myvec(x), myvec(dx))
    dy = reshape(extras.jvp_exe(v_vec), size(y))
    return y, dy
end

function DI.value_and_pushforward!(
    f!,
    y,
    dy,
    ::AutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationTwoArgPushforwardExtras,
)
    f!(y, x)
    v_vec = vcat(myvec(x), myvec(dx))
    extras.jvp_exe!(vec(dy), v_vec)
    return y, dy
end

function DI.pushforward(
    f!,
    y,
    ::AutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationTwoArgPushforwardExtras,
)
    v_vec = vcat(myvec(x), myvec(dx))
    dy = reshape(extras.jvp_exe(v_vec), size(y))
    return dy
end

function DI.pushforward!(
    f!,
    y,
    dy,
    ::AutoFastDifferentiation,
    x,
    dx,
    extras::FastDifferentiationTwoArgPushforwardExtras,
)
    v_vec = vcat(myvec(x), myvec(dx))
    extras.jvp_exe!(vec(dy), v_vec)
    return dy
end

## Pullback

struct FastDifferentiationTwoArgPullbackExtras{E1,E2} <: PullbackExtras
    vjp_exe::E1
    vjp_exe!::E2
end

function DI.prepare_pullback(f!, y, ::AutoFastDifferentiation, x, dy)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = x_var isa Number ? monovec(x_var) : vec(x_var)
    y_vec_var = y_var isa Number ? monovec(y_var) : vec(y_var)
    vj_vec_var, v_vec_var = jacobian_transpose_v(y_vec_var, x_vec_var)
    vjp_exe = make_function(vj_vec_var, vcat(x_vec_var, v_vec_var); in_place=false)
    vjp_exe! = make_function(vj_vec_var, vcat(x_vec_var, v_vec_var); in_place=true)
    return FastDifferentiationTwoArgPullbackExtras(vjp_exe, vjp_exe!)
end

function DI.pullback(
    f!, y, ::AutoFastDifferentiation, x, dy, extras::FastDifferentiationTwoArgPullbackExtras
)
    v_vec = vcat(myvec(x), myvec(dy))
    if x isa Number
        return only(extras.vjp_exe(v_vec))
    else
        return reshape(extras.vjp_exe(v_vec), size(x))
    end
end

function DI.pullback!(
    f!,
    y,
    dx,
    ::AutoFastDifferentiation,
    x,
    dy,
    extras::FastDifferentiationTwoArgPullbackExtras,
)
    v_vec = vcat(myvec(x), myvec(dy))
    extras.vjp_exe!(vec(dx), v_vec)
    return dx
end

function DI.value_and_pullback(
    f!,
    y,
    backend::AutoFastDifferentiation,
    x,
    dy,
    extras::FastDifferentiationTwoArgPullbackExtras,
)
    dx = DI.pullback(f!, y, backend, x, dy, extras)
    f!(y, x)
    return y, dx
end

function DI.value_and_pullback!(
    f!,
    y,
    dx,
    backend::AutoFastDifferentiation,
    x,
    dy,
    extras::FastDifferentiationTwoArgPullbackExtras,
)
    DI.pullback!(f!, y, dx, backend, x, dy, extras)
    f!(y, x)
    return y, dx
end

## Derivative

struct FastDifferentiationTwoArgDerivativeExtras{E1,E2} <: DerivativeExtras
    der_exe::E1
    der_exe!::E2
end

function DI.prepare_derivative(f!, y, ::AutoFastDifferentiation, x)
    x_var = only(make_variables(:x))
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = monovec(x_var)
    y_vec_var = vec(y_var)
    der_vec_var = derivative(y_vec_var, x_var)
    der_exe = make_function(der_vec_var, x_vec_var; in_place=false)
    der_exe! = make_function(der_vec_var, x_vec_var; in_place=true)
    return FastDifferentiationTwoArgDerivativeExtras(der_exe, der_exe!)
end

function DI.value_and_derivative(
    f!, y, ::AutoFastDifferentiation, x, extras::FastDifferentiationTwoArgDerivativeExtras
)
    f!(y, x)
    der = reshape(extras.der_exe(monovec(x)), size(y))
    return y, der
end

function DI.value_and_derivative!(
    f!,
    y,
    der,
    ::AutoFastDifferentiation,
    x,
    extras::FastDifferentiationTwoArgDerivativeExtras,
)
    f!(y, x)
    extras.der_exe!(der, monovec(x))
    return y, der
end

function DI.derivative(
    f!, y, ::AutoFastDifferentiation, x, extras::FastDifferentiationTwoArgDerivativeExtras
)
    der = reshape(extras.der_exe(monovec(x)), size(y))
    return der
end

function DI.derivative!(
    f!,
    y,
    der,
    ::AutoFastDifferentiation,
    x,
    extras::FastDifferentiationTwoArgDerivativeExtras,
)
    extras.der_exe!(der, monovec(x))
    return der
end

## Jacobian

struct FastDifferentiationTwoArgJacobianExtras{E1,E2} <: JacobianExtras
    jac_exe::E1
    jac_exe!::E2
end

function DI.prepare_jacobian(
    f!, y, backend::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}}, x
)
    x_var = make_variables(:x, size(x)...)
    y_var = make_variables(:y, size(y)...)
    f!(y_var, x_var)

    x_vec_var = vec(x_var)
    y_vec_var = vec(y_var)
    jac_var = if backend isa AutoSparse
        sparse_jacobian(y_vec_var, x_vec_var)
    else
        jacobian(y_vec_var, x_vec_var)
    end
    jac_exe = make_function(jac_var, x_vec_var; in_place=false)
    jac_exe! = make_function(jac_var, x_vec_var; in_place=true)
    return FastDifferentiationTwoArgJacobianExtras(jac_exe, jac_exe!)
end

function DI.value_and_jacobian(
    f!,
    y,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationTwoArgJacobianExtras,
)
    f!(y, x)
    jac = extras.jac_exe(vec(x))
    return y, jac
end

function DI.value_and_jacobian!(
    f!,
    y,
    jac,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationTwoArgJacobianExtras,
)
    f!(y, x)
    extras.jac_exe!(jac, vec(x))
    return y, jac
end

function DI.jacobian(
    f!,
    y,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationTwoArgJacobianExtras,
)
    jac = extras.jac_exe(vec(x))
    return jac
end

function DI.jacobian!(
    f!,
    y,
    jac,
    ::Union{AutoFastDifferentiation,AutoSparse{<:AutoFastDifferentiation}},
    x,
    extras::FastDifferentiationTwoArgJacobianExtras,
)
    extras.jac_exe!(jac, vec(x))
    return jac
end
