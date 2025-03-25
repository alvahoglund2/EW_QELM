includet("../src/basis.jl")
includet("../src/hamiltonian.jl")
includet("../src/time_evolution.jl")
includet("../src/measurements.jl")
includet("../src/main_system_state.jl")
includet("../src/reservoir_state.jl")
includet("../src/effective_measurements.jl")
includet("generate_spin_measurements.jl")

ent_state_types = [singlet_state, triplet0_state, tripletn1_state, tripletp1_state]
noise_level_min = 0.5
nbr_ent_states = 10000
nbr_sep_states = 10000

sep_measurements, ent_measurements = generate_spin_measurements(ent_state_types, noise_level_min, nbr_ent_states, nbr_sep_states)

## -------------- Plot ----------------
scatter(sep_measurements, label = "Separable States", xlabel = "X ⊗ X", ylabel = "Y ⊗ Y", zlabel = "Z ⊗ Z",  legend=:topright)
scatter!(ent_measurements, label = "Werner states, p> $noise_level_min", xlabel = "X ⊗ X", ylabel = "Y ⊗ Y", zlabel = "Z ⊗ Z")
