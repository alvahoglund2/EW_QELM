using Pkg
Pkg.activate(pwd())
Pkg.instantiate()

using Plots
using QuantumDots
using Random
using SparseArrays
using LinearAlgebra
using LinearMaps

using PyCall
PyCall.python
np = pyimport("numpy")