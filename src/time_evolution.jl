function state_evolution(ρ0:: AbstractMatrix, t_end :: AbstractFloat, hamiltonian:: AbstractMatrix)
    """
        state_evolution(ρ0:: AbstractMatrix, t_end :: AbstractFloat, hamiltonian:: AbstractMatrix)
    
    Time evolves a state ρ0 with a given Hamiltonian for a time t_end.
    Returns the time evolved state.
    """
    Hd = Matrix(hamiltonian)
    u = sparse(exp(-im * t_end * Hd))
    ρt = u * ρ0 * u'
    return ρt
end

function operator_evolution(M:: AbstractMatrix, t_end :: AbstractFloat, hamiltonian:: AbstractMatrix)
    """
        operator_evolution(M:: AbstractMatrix, t_end :: AbstractFloat, hamiltonian:: AbstractMatrix)
    
    Time evolves an operator M with a given Hamiltonian for a time t_end.
    Returns the time evolved operator.
    """
    Hd = Matrix(hamiltonian)
    u = sparse(cis(-t_end * Hd))
    Mt = u' * M * u
    return Mt
end

function ρ_ode(dρ :: AbstractMatrix, ρ :: AbstractMatrix, p, t)
    Hd = p
    dρ .= -im * (Hd * ρ - ρ * Hd)
end

function state_time_ev(ρ0 :: AbstractMatrix, t_span ::Tuple, hamiltonian :: AbstractMatrix)
    """
        state_time_ev(ρ0 :: AbstractMatrix, t_span ::Tuple, hamiltonian :: AbstractMatrix)
    
    Time evolves a state ρ0 with a given Hamiltonian for a time t_span. 
    Returns the state as a function of time.
    """

    params = hamiltonian
    prob = ODEProblem(ρ_ode, ρ0, t_span, params, reltol=1e-12, abstol=1e-12)
    ρt = solve(prob, Tsit5())
    return ρt
end

 
function get_propagator_block(t_eval :: AbstractFloat, hamiltonian:: AbstractMatrix, d:: FermionBasis, focknbrs)
    """
        get_propagator_block(t_eval :: AbstractFloat, hamiltonian:: AbstractMatrix, d:: FermionBasis, focknbrs)

    Return a vector of propagator blocks for the specified forcknumbers 
    """
    u_blocks = []
    for fn in focknbrs
        ind = d.symmetry.qntoinds[fn]
        h_block = hamiltonian[ind,ind] |> Matrix
        u_block = cis(-t_eval * h_block)
        push!(u_blocks, sparse(u_block))
    end
    return u_blocks
end

function operator_evolution_blocks(M:: AbstractMatrix, u_blocks, d:: FermionBasis, focknbrs)
    """
        operator_evolution_blocks(M:: AbstractMatrix, u_blocks, d:: FermionBasis, focknbrs)
    
    Return the time evolved operator. Only evolves the blocks corresponding to the focknbrs and u_blocks.
    """
    Mt = spzeros(ComplexF64, get_basis_dim(d), get_basis_dim(d))
    for (i, fn) in enumerate(focknbrs)
        ind = d.symmetry.qntoinds[fn]
        M_block = M[ind,ind] 
        Mt_block = u_blocks[i]' * M_block * u_blocks[i]
        Mt[ind,ind] = Mt_block
    end
    return Mt
end