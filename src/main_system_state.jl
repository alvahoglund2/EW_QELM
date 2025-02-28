function vac_state(d :: FermionBasis)
    v0 = zeros(2^length(keys(d)))
    v0[1] = 1
    v0 = sparse(v0)
    return v0
end

function singlet_state(d :: FermionBasis)
    v0 = vac_state(d)
    v = (d[1, :↑]'d[2, :↓]' - d[1, :↓]'d[2, :↑]')*v0
    v = v/norm(v)
    ρ = v*v'
    return ρ
end

function triplet0_state(d :: FermionBasis)
    v0 = vac_state(d)
    v = (d[1, :↑]'d[2, :↓]' + d[1, :↓]'d[2, :↑]')*v0
    v = v/norm(v)
    ρ = v*v'
    return ρ
end

function tripletn1_state(d :: FermionBasis)
    v0 = vac_state(d)
    v = (d[1, :↑]'d[2, :↑]' - d[1, :↓]'d[2, :↓]')*v0
    v = v/norm(v)
    ρ = v*v'
    return ρ
end 

function tripletp1_state(d :: FermionBasis)
    v0 = vac_state(d)
    v = (d[1, :↑]'d[2, :↑]' + d[1, :↓]'d[2, :↓]')*v0
    v = v/norm(v)
    ρ = v*v'
    return ρ
end 

function werner_state(d :: FermionBasis, p :: AbstractFloat, target_state :: Function)
    ρ_target = target_state(d)
    ρ_w = p*ρ_target + (1-p)*max_mixed_state(d)
    return ρ_w
end

werner_state_list(d :: FermionBasis, size :: Integer, target_state :: Function, p_min :: AbstractFloat) = 
    [werner_state(d, p_min + (1 - p_min) * rand(), target_state) for i in 1:size]


function max_mixed_state(d :: FermionBasis)
    v0 = vac_state(d)
    nbr_dots = length(keys(d))/2
    ρ_mixed = sparse(zeros(length(v0), length(v0)))
    for i in 1:nbr_dots
        for j in i+1:nbr_dots
            uu = d[i, :↑]'d[j, :↑]'*v0
            ud = d[i, :↑]'d[j, :↓]'*v0
            du = d[i, :↓]'d[j, :↑]'*v0
            dd = d[i, :↓]'d[j, :↓]'*v0
            ρ_mixed += uu*uu' + ud*ud' + du*du' + dd*dd'
        end
    end
    ρ_mixed = ρ_mixed/tr(ρ_mixed)
    return ρ_mixed
end

function random_state(d :: FermionBasis)
    v0 = vac_state(d)
    spatial_labels = get_spatial_labels(d)
    v = zeros(length(v0))
    for i in spatial_labels
        vup = d[i, :↑]'*v0
        vdown = d[i, :↓]'*v0
        θ = acos(2 * rand() - 1) 
        ϕ = rand()*pi*2
        v += (cos(θ/2)*vup + exp(im*ϕ)*sin(θ/2)*vdown)
    end
    v = v/norm(v)
    return v
end

function random_separable_mixed_state(d :: FermionBasis, dA :: FermionBasis, dB :: FermionBasis)
    nbr_states = rand(1:10)
    states = [random_separable_state(d, dA, dB) for i in 1:nbr_states]
    probs = rand(nbr_states)
    probs = probs / sum(probs)
    total_state = sum([probs[i]*states[i] for i in 1:nbr_states])
    return total_state
end

function random_separable_state(d :: FermionBasis, dA :: FermionBasis, dB :: FermionBasis)
    v1 = random_state(dA)
    v2 = random_state(dB)
    ρ1 = v1*v1'
    ρ2 = v2*v2'
    ρ = wedge([ρ1, ρ2], [dA, dB], d)
    return ρ
end