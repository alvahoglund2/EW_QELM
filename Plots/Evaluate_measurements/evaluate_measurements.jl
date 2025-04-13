using JLD2
using MultivariateStats

function all_measurements()
    seeds = [1,2,3,4]
    res_qds = [1,2,3,4,5]
    qns = nothing 
    if res_qd <= 4
        qns = [i for i in 0:res_qd*2]
    else
        qns = [i for i in 0:3]
    end

    for seed in seeds
        for res_qd in res_qds
            for qn in qns
                load("Plots/Evaluate_measurements/data/measmat_resqd_$(res_qd)_qn_$(qn)_seed_$(seed).jld2", "A")
            end
        end
    end
end

function compare_seeds_and_qn(res_qd, seeds, qns)
    sv = []
    for seed in seeds
        for qn in qns 
            m = load("Plots/Evaluate_measurements/data/measmat_resqd_$(res_qd)_qn_$(qn)_seed_$(seed).jld2", "A")
            U, S, V = svd(Matrix(m))
            count = sum(S .> 0.1)
            push!(sv, count)
        end
    end
    return sv
end

function compare_seeds_and_qn_ratio(res_qd, seeds, qns)
    sv = []
    for seed in seeds
        for qn in qns 
            m = load("Plots/Evaluate_measurements/data/measmat_resqd_$(res_qd)_qn_$(qn)_seed_$(seed).jld2", "A")
            U, S, V = svd(Matrix(m))
            #Largest sv
            maxsv = maximum(S)
            minsv = minimum(S)
            ratiosv = maxsv/minsv
            push!(sv, ratiosv)
        end
    end
    return sv
end

function compare_PC(res_qd, seeds, qns)
    pc_nbr = []
    for seed in seeds
        for qn in qns 
            
            m = load("Plots/Evaluate_measurements/data/measmat_resqd_$(res_qd)_qn_$(qn)_seed_$(seed).jld2", "A")
            #Standardize each column
            m = real(Matrix(m))
            m = (m .- mean(m, dims=1)) ./ std(m, dims=1)
            pca = fit(PCA, real(Matrix(m)), pratio = 0.8)
            n_components = size(pca)[2]
            push!(pc_nbr, n_components)
        end
    end
    return pc_nbr
end


res_qd = 3
seeds = [1, 2, 4]
qn = [i for i in 0:res_qd*2]

pc_nbr = compare_PC(res_qd, seeds, qn)

plot(pc_nbr[qn.+1])
plot!(pc_nbr[(qn.+1).*2])
plot!(pc_nbr[(qn.+1).*3])



#qn = [0, 1, 2,3, 4, 5]
#sv_counts = compare_seeds_and_qn_ratio(res_qd, seeds, qn)

# Define plot
using Plots


#plt = plot()
#i = 1
#s = 1
#n = length(qn)
#i = 1

#plot!(plt,qn, sv_counts[1+(i-1)*n:n+(i-1)*n], label="sv count, seed $(i)", xlabel="qn", ylabel="sv count", marker=:circle)
#i = 2
#plot!(plt,qn, sv_counts[1+(i-1)*n:n+(i-1)*n], label="sv count, seed $(i)", xlabel="qn", ylabel="sv count",marker=:circle)
#i = 3
#plot!(plt,qn, sv_counts[1+(i-1)*n:n+(i-1)*n], label="sv count, seed $(i+1)", xlabel="qn", ylabel="sv count", marker=:circle)
