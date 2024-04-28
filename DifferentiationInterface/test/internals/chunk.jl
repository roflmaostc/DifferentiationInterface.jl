@test DI.pick_chunk_size.(1:(DI.DEFAULT_CHUNK_SIZE)) == 1:(DI.DEFAULT_CHUNK_SIZE)
@test all(
    DI.pick_chunk_size.((DI.DEFAULT_CHUNK_SIZE + 1):(5DI.DEFAULT_CHUNK_SIZE)) .<=
    DI.DEFAULT_CHUNK_SIZE,
)
@test all(
    DI.pick_chunk_size.((DI.DEFAULT_CHUNK_SIZE + 1):(5DI.DEFAULT_CHUNK_SIZE)) .>=
    DI.DEFAULT_CHUNK_SIZE / 2,
)
