check_available(backend::AutoSparse) = check_available(dense_ad(backend))
twoarg_support(backend::AutoSparse) = twoarg_support(dense_ad(backend))
pushforward_performance(backend::AutoSparse) = pushforward_performance(dense_ad(backend))
pullback_performance(backend::AutoSparse) = pullback_performance(dense_ad(backend))
hvp_mode(backend::AutoSparse{<:SecondOrder}) = hvp_mode(dense_ad(backend))
