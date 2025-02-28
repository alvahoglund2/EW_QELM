
function empty_res_state(d :: FermionBasis)
    v0 = zeros(get_basis_dim(d))
    v0[1] = 1
    v0 = sparse(v0)
    return v0
end

function res_eigenstate(H :: AbstractMatrix, d :: FermionBasis, dres :: FermionBasis, qn, excitation_level :: Integer)
    """
        res_eigenstate(H, d, dres, qn, excitation_level)
    
    Calculates the eigenstates of the Hamiltonian for a given quantum number.
    Returns the eigenstates according to the excitation level. (1=ground state)
    """

    Hres = partial_trace(H, dres, d)

    block_index = dres.symmetry.qntoinds[qn]
    Hres_block = Hres[block_index, block_index]

    vals, vecs = eigen(Hres_block)

    ground_state = zeros(ComplexF64, get_basis_dim(dres))
    ground_state[block_index] = vecs[:,excitation_level]
    
    ρ_ground = ground_state*ground_state'
    ρ_ground = ρ_ground/tr(ρ_ground)
    return sparse(ρ_ground)
end

res_ground_state(H, d, dres, qn,) = res_eigenstate(H, d, dres, qn, 1)
