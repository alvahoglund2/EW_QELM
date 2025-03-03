includet("../src/basis.jl")
includet("../src/hamiltonian.jl")
includet("../src/time_evolution.jl")
includet("../src/measurments.jl")
includet("../src/main_system_state.jl")
includet("../src/reservoir_state.jl")
includet("../src/effective_measurments.jl")
includet("generate_effective_measurments.jl")

## -------------- Define System ----------------
t_eval = 1.0
nbr_sep_states = 10000
nbr_ent_states = 10000

sys_qd = 2
res_qd = 3

d, d_main, dA_main, dB_main, d_res = total_basis(sys_qd, res_qd)
hamiltonian_type = Hdot_so_b
h_seed = 1
hamiltonian = random_hamiltonian_no_seed(d, hamiltonian_type, seed = h_seed)

qn = 3
ρ_R = res_ground_state(hamiltonian, d, d_res, qn)

ent_state_types = [singlet_state]
p_min = 0.8
## ------------ Measure states ----------------

sep_states = [random_separable_state(d_main, dA_main, dB_main) for i in 1:nbr_sep_states]
ent_states = vcat([werner_state_list(d_main, nbr_ent_states, type, p_min) for type in ent_state_types]...)

eff_measurments = get_effective_measurments(hamiltonian, ρ_R, t_eval, d, d_main, d_res)
sep_measuments = measure_states(sep_states, eff_measurments)
ent_measurments = measure_states(ent_states, eff_measurments)

all_measurments = vcat(sep_measuments, ent_measurments)

## ------------ pca ----------------

using MultivariateStats

pca = fit(PCA, real(all_measurments)', maxoutdim=3, pratio=1.0)
pca_sep_measurments = predict(pca, real(sep_measuments)')
pca_ent_measurments = predict(pca, real(ent_measurments)')

## ------------ Save data ----------------
labels = vcat([-1 for i in 1:size(ent_measurments,1)], [1 for i in 1:size(sep_measuments,1)])

np.save("Charge_Plots/data/measurments.npy", all_measurments)
np.save("Charge_Plots/data/labels.npy", labels)

## ------------ plot ----------------

using Plots
plotly()
# Plot a 3D scatter plot

scatter(pca_sep_measurments[1,:],pca_sep_measurments[2,:], pca_sep_measurments[3,:], label = "Separable States", xlabel = "PC1", ylabel = "PC2", zlabel = "PC3",  legend=:topright)
scatter!(pca_ent_measurments[1,:],pca_ent_measurments[2,:], pca_ent_measurments[3,:], label = "Entangled States", xlabel = "PC1", ylabel = "PC2", zlabel = "PC3")

