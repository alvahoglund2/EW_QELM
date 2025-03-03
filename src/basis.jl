function def_basis(qd_nbrs::AbstractVector{<:Integer})
    """
        def_basis(qd_nbrs::AbstractVector{<:Integer})
    
    Return a fermion basis sorted on spin for a given set of labeled quantum dots. 
    """
    spins = (:↑, :↓)
    qn = QuantumDots.fermionnumber
    labels_all = [(i, s) for i in qd_nbrs for s in spins]
    return FermionBasis(labels_all; qn)
    println(qd_nbrs)
end

function main_system_basis(qd_nbrs :: AbstractVector{<:Integer}, split_point :: Integer)
    """
        main_system_basis(qd_nbrs :: AbstractVector{<:Integer}, split_point :: Integer)
    
    Return the fermion basis d for the main system.
    Also returns the basis split into two systems, dA and dB.
    """
    d = def_basis(qd_nbrs)
    dA = def_basis(qd_nbrs[1:split_point])
    dB = def_basis(qd_nbrs[split_point+1:end])
    return d, dA, dB
end


function total_basis(nbr_dots_main ::Integer, nbr_dots_res :: Integer, main_system_split_point :: Integer = nbr_dots_main-1)
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

    d_main, dA_main, dB_main = main_system_basis(main_labels, main_system_split_point)
    d_res = def_basis(res_labels)
    d_tot = def_basis(all_labels)
    return d_tot, d_main, dA_main, dB_main, d_res
end


function get_spatial_labels(d :: FermionBasis)
    return unique([t[1] for t in keys(d)])
end

function get_basis_dim(d :: FermionBasis)
    return 2^length(keys(d))
end

function get_qubit_idx()
    return [7, 8, 9, 10]
end