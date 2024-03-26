## Pullback

function DI.value_and_pullback!!(
    f!, y, _dx, ::AutoReverseEnzyme, x::Number, dy, extras::Nothing
)
    dy_sametype = convert(typeof(y), copy(dy))
    _, new_dx = only(autodiff(Reverse, f!, Const, Duplicated(y, dy_sametype), Active(x)))
    return y, new_dx
end

function DI.value_and_pullback!!(
    f!, y, dx, ::AutoReverseEnzyme, x::AbstractArray, dy, extras::Nothing
)
    dx_sametype = convert(typeof(x), dx)
    dx_sametype .= zero(eltype(dx_sametype))
    dy_sametype = convert(typeof(y), copy(dy))  # TODO: how to get rid of copy?
    autodiff(Reverse, f!, Const, Duplicated(y, dy_sametype), Duplicated(x, dx_sametype))
    return y, myupdate!!(dx, dx_sametype)
end