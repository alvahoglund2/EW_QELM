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
    u = sparse(exp(-im * t_end * Hd))
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