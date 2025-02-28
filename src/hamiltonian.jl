# QD Enery levels
Hsingle_e(ϵ, d :: FermionBasis) = sum(
    ϵ[i] * d[i, σ]' * d[i, σ] 
    for σ ∈ [:↑, :↓], i ∈ get_spatial_labels(d))

# Tunneling Hamiltonian
Hsingle_t(t, d :: FermionBasis) = sum(
    t[i,j]*d[i, σ]'d[j, σ] 
    for σ ∈ [:↑, :↓], i ∈ get_spatial_labels(d), j ∈ get_spatial_labels(d) if i != j)

# Magnetic field Hamiltonian
Hsingle_e_b(ϵ, d :: FermionBasis) = sum(
    ϵ[i] * d[i, σ]' * d[i, σ] +(-1)^(n+1) * d[i, σ]' * d[i, σ] 
    for (n, σ) ∈ enumerate([:↑, :↓]), i ∈ get_spatial_labels(d))

# Spin-orbit Hamiltonian
Hsingle_t_so(t, d :: FermionBasis) = sum(
    t[i,j]*d[i, σ1]'d[j, σ2] 
    for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓], i ∈ get_spatial_labels(d), j ∈ get_spatial_labels(d) if i != j)

# Coulomb interaction Hamiltonians
Hcoulomb_inter(U_inter, d :: FermionBasis) = sum(
    U_inter[i,j]*d[i, σ1]'d[j, σ2]'d[j, σ2]d[i, σ1] 
    for σ1 ∈ [:↑, :↓], σ2 ∈ [:↑, :↓], i ∈ get_spatial_labels(d), j ∈ get_spatial_labels(d) if i < j)

Hcoulomb_intra(U_intra, d :: FermionBasis) = sum(
    U_intra[i]*d[i, :↑]'d[i, :↓]'d[i, :↓]d[i, :↑] 
    for i ∈ get_spatial_labels(d))

Hcoulomb(U_intra, Uinter, d :: FermionBasis) = 
    Hcoulomb_inter(Uinter, d) + Hcoulomb_intra(U_intra, d)

#Total Hamiltonians with options for spin-orbit and magnetic field
Hdot(ϵ, t, Uintra, Uinter, d :: FermionBasis) =
     Hsingle_e(ϵ, d) + Hsingle_t(t, d) + Hcoulomb(Uintra, Uinter, d)

Hdot_so(ϵ, t, Uintra, Uinter, d :: FermionBasis) = 
    Hsingle_e(ϵ, d) + Hsingle_t_so(t, d) + Hcoulomb(Uintra, Uinter, d)

Hdot_b(ϵ, t, Uintra, Uinter, d :: FermionBasis) = 
    Hsingle_e_b(ϵ, d) + Hsingle_t(t, d) + Hcoulomb(Uintra, Uinter, d)

Hdot_so_b(ϵ, t, Uintra, Uinter, d :: FermionBasis) = 
    Hsingle_e_b(ϵ, d) + Hsingle_t_so(t, d) + Hcoulomb(Uintra, Uinter, d)


function random_hamiltonian_rng(d :: FermionBasis, H_type; ϵ_factor = 1.0, t_factor = 1.0, Uintra_factor=10.0, Uinter_factor = 1.0, seed=nothing)
    """
        random_hamiltonian(d :: FermionBasis, H_type, ϵ_factor = 1.0, t_factor = 1.0, Uintra_factor=10.0, Uinter_factor = 1.0, seed=nothing)
    
    Return a random Hamiltonian of type H_type for a given basis d.
    Paramaters randomized in the uniform distributions: 
        t : [0,1]*t_factor + im*[0,1]*t_factor
        Uintra : [1,2]*Uintra_factor
        Uinter : [0,1]*Uinter_factor
        ϵ : [0,1]*ϵ_factor
    """
    
    rng = MersenneTwister()
    if seed !== nothing
        rng = MersenneTwister(seed)
    end

    nbr_dots = Int(length(keys(d))/2)
    ϵ = rand(rng, nbr_dots)*ϵ_factor
    Uintra = (rand(rng, nbr_dots) .+ 1)*Uintra_factor
    Uinter = rand(rng, nbr_dots,nbr_dots)*Uinter_factor
    t = rand(rng, nbr_dots,nbr_dots)*t_factor  + im*rand(rng, nbr_dots,nbr_dots)*t_factor
    t = 0.5*(t + t')
    return H_type(ϵ, t, Uintra, Uinter, d)
end


function random_hamiltonian_Random(d :: FermionBasis, H_type; ϵ_factor = 1.0, t_factor = 1.0, Uintra_factor=10.0, Uinter_factor = 1.0, seed=nothing)
    """
        random_hamiltonian(d :: FermionBasis, H_type, ϵ_factor = 1.0, t_factor = 1.0, Uintra_factor=10.0, Uinter_factor = 1.0, seed=nothing)
    
    Return a random Hamiltonian of type H_type for a given basis d.
    Paramaters randomized in the uniform distributions: 
        t : [0,1]*t_factor + im*[0,1]*t_factor
        Uintra : [1,2]*Uintra_factor
        Uinter : [0,1]*Uinter_factor
        ϵ : [0,1]*ϵ_factor
    """
    
    Random.seed!(seed)
    nbr_dots = Int(length(keys(d))/2)
    ϵ = rand(nbr_dots)*ϵ_factor
    Uintra = (rand(nbr_dots) .+ 1)*Uintra_factor
    Uinter = rand(nbr_dots,nbr_dots)*Uinter_factor
    t = rand(nbr_dots,nbr_dots)*t_factor  + im*rand(nbr_dots,nbr_dots)*t_factor
    t = 0.5*(t + t')
    Random.seed!(nothing)
    return H_type(ϵ, t, Uintra, Uinter, d)
end

function random_hamiltonian_no_seed(d :: FermionBasis, H_type; ϵ_factor = 1.0, t_factor = 1.0, Uintra_factor=10.0, Uinter_factor = 1.0, seed=10)
    """
        random_hamiltonian(d :: FermionBasis, H_type, ϵ_factor = 1.0, t_factor = 1.0, Uintra_factor=10.0, Uinter_factor = 1.0, seed=nothing)
    
    Return a random Hamiltonian of type H_type for a given basis d.
    Paramaters randomized in the uniform distributions: 
        t : [0,1]*t_factor + im*[0,1]*t_factor
        Uintra : [1,2]*Uintra_factor
        Uinter : [0,1]*Uinter_factor
        ϵ : [0,1]*ϵ_factor
    """
    nbr_dots = Int(length(keys(d))/2)
    ϵ = rand(nbr_dots)*ϵ_factor
    Uintra = (rand(nbr_dots) .+ 1)*Uintra_factor
    Uinter = rand(nbr_dots,nbr_dots)*Uinter_factor
    t = rand(nbr_dots,nbr_dots)*t_factor  + im*rand(nbr_dots,nbr_dots)*t_factor
    t = 0.5*(t + t')
    return H_type(ϵ, t, Uintra, Uinter, d)
end