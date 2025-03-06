using Pkg
Pkg.activate(pwd())
Pkg.instantiate()

using Plots
using QuantumDots, QuantumDots.BlockDiagonals
using Random
using SparseArrays
using LinearAlgebra
using LinearMaps

using PyCall
PyCall.python
np = pyimport("numpy")