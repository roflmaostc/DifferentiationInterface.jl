using DifferentiationInterface, DifferentiationInterfaceTest
using Tracker: Tracker
using Test

for backend in [AutoTracker()]
    @test check_available(backend)
    @test !check_twoarg(backend)
    @test !check_hessian(backend; verbose=false)
end

test_differentiation(AutoTracker(); second_order=false, logging=LOGGING);
