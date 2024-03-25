## Pushforward

function DI.prepare_pushforward(f, ::AutoFastDifferentiation, x)
    x_var = if x isa Number
        only(make_variables(:x))
    else
        make_variables(:x, size(x)...)
    end
    y_var = f(x_var)

    x_vec_var = x_var isa Number ? [x_var] : x_var
    y_vec_var = y_var isa Number ? [y_var] : y_var
    jv_vec_var, v_vec_var = jacobian_times_v(y_vec_var, x_vec_var)
    jvp_exe = make_function(jv_vec_var, [x_vec_var; v_vec_var]; in_place=false)
    return jvp_exe
end

function DI.value_and_pushforward(
    f, ::AutoFastDifferentiation, x, dx, jvp_exe::RuntimeGeneratedFunction
)
    y = f(x)
    v_vec = vcat(myvec(x), myvec(dx))
    jv_vec = jvp_exe(v_vec)
    if y isa Number
        return y, only(jv_vec)
    else
        return y, jv_vec
    end
end

function DI.value_and_pushforward(
    f, backend::AutoFastDifferentiation, x, dx, extras::Nothing
)
    jvp_exe = DI.prepare_pushforward(f, backend, x)
    return DI.value_and_pushforward(f, backend, x, dx, jvp_exe)
end

function DI.value_and_pushforward!!(f, dy, backend::AutoFastDifferentiation, x, dx, extras)
    y, new_dy = DI.value_and_pushforward(f, backend, x, dx, extras)
    return y, myupdate!!(dy, new_dy)
end