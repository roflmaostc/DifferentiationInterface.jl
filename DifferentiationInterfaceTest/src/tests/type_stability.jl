function filt(@nospecialize f)
    return f !== Base.print
end

## Pushforward

function test_jet(ba::AbstractADType, scen::PushforwardScenario{1,:outofplace}; ref_backend)
    @compat (; f, x, y, dx) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x, dx)

    if Bool(pushforward_performance(ba))
        JET.@test_opt function_filter = filt value_and_pushforward(f, ba, x, dx, extras)
        JET.@test_opt function_filter = filt pushforward(f, ba, x, dx, extras)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PushforwardScenario{1,:inplace}; ref_backend)
    @compat (; f, x, y, dx) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x, dx)
    dy_in = mysimilar(y)

    if Bool(pushforward_performance(ba))
        JET.@test_opt function_filter = filt value_and_pushforward!(
            f, dy_in, ba, x, dx, extras
        )
        JET.@test_opt function_filter = filt pushforward!(f, dy_in, ba, x, dx, extras)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PushforwardScenario{2,:outofplace}; ref_backend)
    @compat (; f, x, y, dx) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, mysimilar(y), ba, x, dx)
    y_in = mysimilar(y)

    if Bool(pushforward_performance(ba))
        JET.@test_opt function_filter = filt value_and_pushforward(
            f!, y_in, ba, x, dx, extras
        )
        JET.@test_opt function_filter = filt pushforward(f!, y_in, ba, x, dx, extras)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PushforwardScenario{2,:inplace}; ref_backend)
    @compat (; f, x, y, dx) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, mysimilar(y), ba, x, dx)
    y_in, dy_in = mysimilar(y), mysimilar(y)

    if Bool(pushforward_performance(ba))
        JET.@test_opt function_filter = filt value_and_pushforward!(
            f!, y_in, dy_in, ba, x, dx, extras
        )
        JET.@test_opt function_filter = filt pushforward!(
            f!, y_in, dy_in, ba, x, dx, extras
        )
    end
    return nothing
end

## Pullback

function test_jet(ba::AbstractADType, scen::PullbackScenario{1,:outofplace}; ref_backend)
    @compat (; f, x, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x, dy)

    if Bool(pullback_performance(ba))
        JET.@test_opt function_filter = filt value_and_pullback(f, ba, x, dy, extras)
        JET.@test_opt function_filter = filt pullback(f, ba, x, dy, extras)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PullbackScenario{1,:inplace}; ref_backend)
    @compat (; f, x, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x, dy)
    dx_in = mysimilar(x)

    if Bool(pullback_performance(ba))
        JET.@test_opt function_filter = filt value_and_pullback!(
            f, dx_in, ba, x, dy, extras
        )
        JET.@test_opt function_filter = filt pullback!(f, dx_in, ba, x, dy, extras)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PullbackScenario{2,:outofplace}; ref_backend)
    @compat (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, mysimilar(y), ba, x, dy)
    y_in = mysimilar(y)

    if Bool(pullback_performance(ba))
        JET.@test_opt function_filter = filt value_and_pullback(f!, y_in, ba, x, dy, extras)
        JET.@test_opt function_filter = filt pullback(f!, y_in, ba, x, dy, extras)
    end
    return nothing
end

function test_jet(ba::AbstractADType, scen::PullbackScenario{2,:inplace}; ref_backend)
    @compat (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, mysimilar(y), ba, x, dy)
    y_in, dx_in = mysimilar(y), mysimilar(x)

    if Bool(pullback_performance(ba))
        JET.@test_opt function_filter = filt value_and_pullback!(
            f!, y_in, dx_in, ba, x, dy, extras
        )
        JET.@test_opt function_filter = filt pullback!(f!, y_in, dx_in, ba, x, dy, extras)
    end
    return nothing
end

## Derivative

function test_jet(ba::AbstractADType, scen::DerivativeScenario{1,:outofplace}; ref_backend)
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)

    JET.@test_opt function_filter = filt value_and_derivative(f, ba, x, extras)
    JET.@test_opt function_filter = filt derivative(f, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::DerivativeScenario{1,:inplace}; ref_backend)
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    der_in = mysimilar(y)

    JET.@test_opt function_filter = filt value_and_derivative!(f, der_in, ba, x, extras)
    JET.@test_opt function_filter = filt derivative!(f, der_in, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::DerivativeScenario{2,:outofplace}; ref_backend)
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, mysimilar(y), ba, x)
    y_in = mysimilar(y)

    JET.@test_opt function_filter = filt value_and_derivative(f!, y_in, ba, x, extras)
    JET.@test_opt function_filter = filt derivative(f!, y_in, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::DerivativeScenario{2,:inplace}; ref_backend)
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, mysimilar(y), ba, x)
    y_in, der_in = mysimilar(y), mysimilar(y)

    JET.@test_opt function_filter = filt value_and_derivative!(
        f!, y_in, der_in, ba, x, extras
    )
    JET.@test_opt function_filter = filt derivative!(f!, y_in, der_in, ba, x, extras)
    return nothing
end

## Gradient

function test_jet(ba::AbstractADType, scen::GradientScenario{1,:outofplace}; ref_backend)
    @compat (; f, x) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)

    JET.@test_opt function_filter = filt value_and_gradient(f, ba, x, extras)
    JET.@test_opt function_filter = filt gradient(f, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::GradientScenario{1,:inplace}; ref_backend)
    @compat (; f, x) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    grad_in = mysimilar(x)

    JET.@test_opt function_filter = filt value_and_gradient!(f, grad_in, ba, x, extras)
    JET.@test_opt function_filter = filt gradient!(f, grad_in, ba, x, extras)
    return nothing
end

## Jacobian

function test_jet(ba::AbstractADType, scen::JacobianScenario{1,:outofplace}; ref_backend)
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)

    JET.@test_opt function_filter = filt value_and_jacobian(f, ba, x, extras)
    JET.@test_opt function_filter = filt jacobian(f, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::JacobianScenario{1,:inplace}; ref_backend)
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_in = Matrix{eltype(y)}(undef, length(y), length(x))

    JET.@test_opt function_filter = filt value_and_jacobian!(f, jac_in, ba, x, extras)
    JET.@test_opt function_filter = filt jacobian!(f, jac_in, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::JacobianScenario{2,:outofplace}; ref_backend)
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, mysimilar(y), ba, x)
    y_in = mysimilar(y)

    JET.@test_opt function_filter = filt value_and_jacobian(f!, y_in, ba, x, extras)
    JET.@test_opt function_filter = filt jacobian(f!, y_in, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::JacobianScenario{2,:inplace}; ref_backend)
    @compat (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, mysimilar(y), ba, x)
    y_in, jac_in = mysimilar(y), Matrix{eltype(y)}(undef, length(y), length(x))

    JET.@test_opt function_filter = filt value_and_jacobian!(
        f!, y_in, jac_in, ba, x, extras
    )
    JET.@test_opt function_filter = filt jacobian!(f!, y_in, jac_in, ba, x, extras)
    return nothing
end

## Second derivative

function test_jet(
    ba::AbstractADType, scen::SecondDerivativeScenario{1,:outofplace}; ref_backend
)
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)

    JET.@test_opt function_filter = filt second_derivative(f, ba, x, extras)
    return nothing
end

function test_jet(
    ba::AbstractADType, scen::SecondDerivativeScenario{1,:inplace}; ref_backend
)
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)
    der2_in = mysimilar(y)

    JET.@test_opt function_filter = filt second_derivative!(f, der2_in, ba, x, extras)
    return nothing
end

## HVP

function test_jet(ba::AbstractADType, scen::HVPScenario{1,:outofplace}; ref_backend)
    @compat (; f, x, dx) = deepcopy(scen)
    extras = prepare_hvp(f, ba, x, dx)

    JET.@test_opt function_filter = filt hvp(f, ba, x, dx, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::HVPScenario{1,:inplace}; ref_backend)
    @compat (; f, x, dx) = deepcopy(scen)
    extras = prepare_hvp(f, ba, x, dx)
    p_in = mysimilar(x)

    JET.@test_opt function_filter = filt hvp!(f, p_in, ba, x, dx, extras)
    return nothing
end

## Hessian

function test_jet(ba::AbstractADType, scen::HessianScenario{1,:outofplace}; ref_backend)
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)

    JET.@test_opt function_filter = filt hessian(f, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, scen::HessianScenario{1,:inplace}; ref_backend)
    @compat (; f, x, y) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)
    hess_in = Matrix{typeof(y)}(undef, length(x), length(x))

    JET.@test_opt function_filter = filt hessian!(f, hess_in, ba, x, extras)
    return nothing
end
