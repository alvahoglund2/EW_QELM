function def_basis(qd_nbrs::AbstractVector{<:Integer}; conserved_qn = QuantumDots.fermionnumber)
    """
        def_basis(qd_nbrs::AbstractVector{<:Integer})
    
    Return a fermion basis sorted on spin for a given set of labeled quantum dots. 
    """
    spins = (:↑, :↓)
    qn = conserved_qn
    labels_all = [(i, s) for i in qd_nbrs for s in spins]
    return FermionBasis(labels_all; qn)
    println(qd_nbrs)
end

function main_system_basis(qd_nbrs :: AbstractVector{<:Integer}, split_point :: Integer; conserved_qn = QuantumDots.fermionnumber)
    """
        main_system_basis(qd_nbrs :: AbstractVector{<:Integer}, split_point :: Integer)
    
    Return the fermion basis d for the main system.
    Also returns the basis split into two systems, dA and dB.
    """
    d = def_basis(qd_nbrs, conserved_qn = conserved_qn)
    dA = def_basis(qd_nbrs[1:split_point], conserved_qn = conserved_qn)
    dB = def_basis(qd_nbrs[split_point+1:end], conserved_qn = conserved_qn)
    return d, dA, dB
end

function total_basis(nbr_dots_main ::Integer, nbr_dots_res :: Integer; main_system_split_point :: Integer = nbr_dots_main-1, conserved_qn = QuantumDots.fermionnumber)
    """
        total_basis(nbr_dots_main ::Integer, nbr_dots_res :: Integer, main_system_split_point :: Integer = nbr_dots_main-1)
    
    # Returns
    - d_tot : Basis for total system
    - d_main: Basis for the main system
    - dA_main, dB_main : Basis for the main system split into two subsystems
    - d_res : Basis for the reservoir
    """
    main_labels = 1:nbr_dots_main
    res_labels = nbr_dots_main+1:nbr_dots_main+nbr_dots_res
    all_labels = vcat(main_labels, res_labels)

    d_main, dA_main, dB_main = main_system_basis(main_labels, main_system_split_point, conserved_qn = conserved_qn)
    d_res = def_basis(res_labels, conserved_qn = conserved_qn)
    d_tot = def_basis(all_labels, conserved_qn = conserved_qn)
    return d_tot, d_main, dA_main, dB_main, d_res
end


function get_spatial_labels(d :: FermionBasis)
    return unique([t[1] for t in keys(d)])
end

function get_basis_dim(d :: FermionBasis)
    return 2^length(keys(d))
end


function get_qubit_idx(d_main :: FermionBasis)
    nbr_dots = length(get_spatial_labels(d_main))

    if nbr_dots == 1
        get_single_qubit_idx(d_main)
    elseif nbr_dots == 2
        return get_two_qubit_idx(d_main)
    else
        throw(ArgumentError("Only implemented for 1 or 2 dots"))
    end
end

function get_single_qubit_idx(d_main :: FermionBasis)
    v0 = vac_state(d_main)
    u_idx = findall(!iszero, d_main[1, :↑]'*v0)[1]
    d_idx = findall(!iszero, d_main[1, :↓]'*v0)[1]
    return sort!([u_idx, d_idx])
end

function get_two_qubit_idx(d_main :: FermionBasis)
    v0 = vac_state(d_main)
    uu_idx =findall(!iszero, d_main[1, :↑]'d_main[2, :↑]'*v0)[1]
    ud_idx =findall(!iszero, d_main[1, :↑]'d_main[2, :↓]'*v0)[1]
    du_idx =findall(!iszero, d_main[1, :↓]'d_main[2, :↑]'*v0)[1]
    dd_idx =findall(!iszero, d_main[1, :↓]'d_main[2, :↓]'*v0)[1]
    return sort!([uu_idx, ud_idx, du_idx, dd_idx])
end

@time get_qubit_idx(d_main_test)