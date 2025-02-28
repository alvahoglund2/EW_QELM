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

expectation_value(ρ :: AbstractMatrix, op :: AbstractMatrix) = process_complex(tr(ρ*op))

function process_complex(value, tolerance=1e-3)
    if abs(imag(value)) > tolerance
        throw(ArgumentError("Expectation value has an imaginary part: $(imag(value))"))
    end
    return real(value)
end
