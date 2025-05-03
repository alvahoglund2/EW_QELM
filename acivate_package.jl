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
using JLD2

plotly()

palette = [
    RGB(120/255, 94/255, 240/255),  # Purple
    RGB(100/255, 143/255, 255/255),  # Blue
    RGB(90/255, 200/255, 250/255),
    RGB(220/255, 38/255, 127/255),  # Pink
    RGB(254/255, 97/255, 0/255),    # Red-Orange
    RGB(255/255, 176/255, 0/255),   # Orange
]