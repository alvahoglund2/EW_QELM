includet("../src/basis.jl")
includet("../src/hamiltonian.jl")
includet("../src/time_evolution.jl")
includet("../src/measurments.jl")
includet("../src/main_system_state.jl")
includet("../src/reservoir_state.jl")
includet("../src/effective_measurments.jl")
includet("generate_effective_measurments.jl")

## -------------- Define System ----------------
ent_state_types = [triplet0_state]
sep_state_types = [random_separable_state]
mix_sep_state_types = [random_separable_mixed_state]
state_types = [ent_state_types, sep_state_types, mix_sep_state_types]

noise_level_min = 1/3

t_eval = 1.0

nbr_ent_states =1000
nbr_sep_states = 10000
nbr_mix_sep_states = 10000

nbr_states = [nbr_ent_states, nbr_sep_states, nbr_mix_sep_states]

sys_qd = 2
res_qd = 3

conserved_qn = QuantumDots.fermionnumber
d, d_main, dA_main, dB_main, d_res = total_basis(sys_qd, res_qd, conserved_qn = conserved_qn)
hamiltonian_type = Hdot_so_b

hamiltonian = random_hamiltonian(d, hamiltonian_type, seed=4)

qn = 2
focknbrs = 2+qn

ρ_R = res_ground_state(hamiltonian, d, d_res, qn)


## ------------ Measure states ----------------
sep_measurments, ent_measurments, mix_sep_measurements= charge_measurments(hamiltonian, ρ_R, t_eval, d, d_main, dA_main, dB_main, d_res, nbr_states, state_types, noise_level_min, focknbrs)

all_measurments = vcat(sep_measurments, ent_measurments, mix_sep_measurements)

using Polyhedra, CDDLib

function get_extreme_points(v)
    p = polyhedron(v, CDDLib.Library())
    removevredundancy!(p)
    return points(p)
end


v = vrep(sep_measurments[1:1000,:])
@time p =get_extreme_points(v)
p

## ------------ pca ----------------

using MultivariateStats

pca = fit(PCA, real(all_measurments)', maxoutdim=3, pratio=1.0)
pca_sep_measurments = predict(pca, real(sep_measurments)')
pca_ent_measurments = predict(pca, real(ent_measurments)')
pca_mix_sep_measurements = predict(pca, real(mix_sep_measurements)')

## ------------ Save data ----------------
labels = vcat([-1 for i in 1:size(ent_measurments,1)], [1 for i in 1:size(sep_measurments,1)], [0 for i in 1:size(mix_sep_measurements,1)])

np.save("Charge_Plots/data/measurments.npy", all_measurments)
np.save("Charge_Plots/data/labels.npy", labels)

## ------------ plot ----------------

ρ=random_separable_mixed_state(d_main, dA_main, dB_main)

tr(ρ^2)

# Plot a 3D scatter plot
using Plots

plotlyjs()

scatter(pca_sep_measurments[1,:],pca_sep_measurments[2,:], pca_sep_measurments[3,:], label = "Pure Separable States", xlabel = "PC1", ylabel = "PC2", zlabel = "PC3",  legend=:topright)
scatter!(pca_ent_measurments[1,:],pca_ent_measurments[2,:], pca_ent_measurments[3,:], label = "Entangled States", xlabel = "PC1", ylabel = "PC2", zlabel = "PC3")
scatter!(pca_mix_sep_measurements[1,:],pca_mix_sep_measurements[2,:], pca_mix_sep_measurements[3,:], label = "Mixed Separable States", xlabel = "PC1", ylabel = "PC2", zlabel = "PC3")

