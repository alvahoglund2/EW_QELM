using Pkg
Pkg.activate(pwd())
Pkg.instantiate()

using QuantumDots, QuantumDots.BlockDiagonals
using Random
using SparseArrays
using LinearAlgebra
using LinearMaps
using RandomMatrices
using Polyhedra, CDDLib
using MultivariateStats

using PyCall
PyCall.python
np = pyimport("numpy")

using Plots
plotly()
