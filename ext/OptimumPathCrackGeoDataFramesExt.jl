module OptimumPathCrackGeoDataFramesExt

using OptimumPathCrack
using OptimumPathCrack: swap
using DataFrames, Graphs, SparseArrays
using GeoDataFrames
const GI = GeoDataFrames.GeoInterface

include("writeGPKGFile.jl")

end
