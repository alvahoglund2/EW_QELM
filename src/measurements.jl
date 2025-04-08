# Pauli matrices
sx(i, d) = (d[i, :↓]'*d[i, :↑] + d[i, :↑]'*d[i, :↓])
sy(i, d) = im*(d[i, :↓]'*d[i, :↑] - d[i, :↑]'*d[i, :↓])
sz(i, d) = (d[i, :↑]'*d[i, :↑] - d[i, :↓]'*d[i, :↓])
si(i, d) = [0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0] # ? 


function pauli_matrix(spatial_labels, d :: FermionBasis, pauli_matrix_type :: Function)
    N = length(keys(d))
    s = sparse(zeros(2^N, 2^N))
    for i in spatial_labels
        s+= pauli_matrix_type(i, d)
    end
    return s
end

function pauli_string(d :: FermionBasis, dA :: FermionBasis, dB :: FermionBasis, pauli_matrix_typeA :: Function, pauli_matrix_typeB :: Function)
    spatial_labelsA = get_spatial_labels(dA)
    spatial_labelsB = get_spatial_labels(dB)
    opA = pauli_matrix(spatial_labelsA, dA, pauli_matrix_typeA)
    opB = pauli_matrix(spatial_labelsB, dB, pauli_matrix_typeB)
    return(wedge([opA, opB], [dA, dB], d))
end

# charge operator
nbr_op(n :: Integer, d :: FermionBasis) = d[n, :↑]'*d[n, :↑] + d[n, :↓]'*d[n, :↓]

# 2 charges operator
nbr2_op(n :: Integer, d :: FermionBasis) = d[n, :↑]'*d[n, :↑]*d[n, :↓]'*d[n, :↓]

function expectation_value(ρ :: AbstractMatrix, op :: AbstractMatrix)
    if !isapprox(op, op')
        throw(ArgumentError("Operator is not Hermitian: $(op)"))
    end
    return process_complex(tr(ρ*op))
end

function process_complex(value, tolerance=1e-3)
    if abs(imag(value)) > tolerance
        throw(ArgumentError("Expectation value has an imaginary part: $(imag(value))"))
    end
    return real(value)
end

function measure_states(state_list, eff_measurements, d_main :: FermionBasis)
    n_states = length(state_list)
    n_measurements = length(eff_measurements)
    result = zeros(n_states, n_measurements)
    idx = get_qubit_idx(d_main)
    for (i, state) in enumerate(state_list)
        for (j, eff_measurement) in enumerate(eff_measurements)
            trunc_state = state[idx,idx]
            result[i, j] = expectation_value(trunc_state, eff_measurement)
        end
    end    
    return result
end
