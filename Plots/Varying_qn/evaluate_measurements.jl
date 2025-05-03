using JLD2
using MultivariateStats

function get_measurement_matrix(res_qd, qn, seed)
    m = load("Plots/Varying_qn/Measurement_Operators/measurementop_res_$(res_qd)_qn_$(qn)_seed_$(seed).jld2", "A")
    m_vec = [vec(e) for e in m]
    m = hcat(m_vec...)
    return m
end

function compare_seeds_and_qn(res_qd, seeds, qns)
    sv = []
    for seed in seeds
        for qn in qns 
            m = get_measurement_matrix(res_qd, qn, seed)
            U, S, V = svd(Matrix(m))
            count = sum(S./maximum(S) .> 0.1)
            push!(sv, count)
        end
    end
    return sv
end

function ordered_sv(res_qd, seeds, qns)
    sv = []
    for seed in seeds
        for qn in qns 
            m = get_measurement_matrix(res_qd, qn, seed)
            print(size(Matrix(m)))
            U, S, V = svd(Matrix(m))
            # Sort s
            S = sort(S, rev=true)
            push!(sv, S)
        end
    end
    return sv
end

function condition_numbers(res_qd, seeds, qns)
    sv = []
    for seed in seeds
        for qn in qns 
            m = get_measurement_matrix(res_qd, qn, seed)
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

function condition_numbers_avg(res_qd, seeds, qns)
    sv_seeds = []
    for seed in seeds
        sv_seed = []
        for qn in qns 
            m = get_measurement_matrix(res_qd, qn, seed)
            U, S, V = svd(Matrix(m))
            #Largest sv
            maxsv = maximum(S)
            minsv = minimum(S)
            ratiosv = maxsv/minsv
            push!(sv_seed, ratiosv)
        end
        push!(sv_seeds, sv_seed)
    end
    sv_mean = mean(sv_seeds, dims=1)
    return sv_mean
end

function compare_PC(res_qd, seeds, qns)
    pc_nbr = []
    for seed in seeds
        for qn in qns 
            m = get_measurement_matrix(res_qd, qn, seed)
            m = real(Matrix(m))
            m = (m .- mean(m, dims=1)) ./ std(m, dims=1)
            pca = fit(PCA, real(Matrix(m)), pratio = 0.8)
            n_components = size(pca)[2]
            push!(pc_nbr, n_components)
        end
    end
    return pc_nbr
end

function print_measop(res_qd, qn, seed, measurement)
    m = load("Plots/Varying_qn/Measurement_Operators/measurementop_res_$(res_qd)_qn_$(qn)_seed_$(seed).jld2", "A")
    println("Measurement matrix for res_qd: $(res_qd), qn: $(qn), seed: $(seed)")
    m_print =Matrix(m[measurement])
    
    print(m_print[1:4])
    print(m_print[5:8])
    print(m_print[9:12])
    print(m_print[13:16])
end
##========================
res_qd = 4
qn = 1
seed = 4
measurement = 1
print_measop(res_qd, qn, seed, measurement)

res_qd = 3
seeds = [1, 2, 4]
qn = [0:res_qd*2...]
sv_counts = compare_seeds_and_qn(res_qd, seeds, qn)

##===========================

res_qd = 2
seeds = [1, 2, 4, 5, 6, 7]
qn = [0:res_qd*2...]
cn = condition_numbers(res_qd, seeds, qn)
p = plot(yscale=:log10, xlabel = "log()", ylabel = "sv", legend = :topleft,)
for (i, seed) in enumerate(seeds)
    start_idx = 1 + (i - 1) * length(qn)
    end_idx = i * length(qn)
    plot!(p, qn, cn[start_idx:end_idx],  marker=:circle)
end
display(p)

##========================
function plot_avg_cn()
    res_qds = [1,2, 3, 4]
    p = plot(
        yscale = :log10,
        legend = :topleft,
        palette = palette[2:5],
        xlabel = "Particle number",
        ylabel = "Condition number",
        guidefontsize = 12,         
        legendfontsize = 10,        
        tickfontsize = 10,          
    )
    for res_qd in res_qds
        seeds = [1, 2, 4, 5, 6, 7]
        qn = [0:res_qd*2...]              # x-axis values
        cn_avg = condition_numbers_avg(res_qd, seeds, qn)  # your custom function
        plot!(p, qn, cn_avg[1],
            label = "$(res_qd) QD reservoir",
            marker = :circle,
            alpha = 0.8,
            xticks = qn)
    end
    return p
end

p = plot_avg_cn()
display(p)
gr()
#Save figure
savefig(p, "Plots/Varying_qn/condition_numbers.png")

##========================
using Plots
qn = [1,2,3,4,5]
y1 = sin.(qn)
cn_avg = condition_numbers_avg(4, seeds, qn)[1]
plot(x, y1, title="Trigonometric functions", label=["sin(x)" "cos(x)"], linewidth=3)
plot(x, cn_avg, title="Trigonometric functions", label=["sin(x)" "cos(x)"], linewidth=3)


## ========================

res_qd = 6
qn = 0
seed = 1
m = get_measurement_matrix(res_qd, qn, seed)
svd(Matrix(m))



res_qd = 4
qns = [0:res_qd*2...]
