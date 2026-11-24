module OptimumPathCrack

using CSV
using DataFrames
using Dates
using Distributions
using Geodesy
using Graphs
using Random
using SparseArrays
using TOML

swap(t::Tuple) = (t[2], t[1])

# Functions implemented in package extensions (see ext/). They gain methods once the
# package(s) they need are loaded.
"""
	writeGPKGFile(g, coords, distance_matrix, weight_matrix;
				  removed_matrix, path_matrix, od, filename)

Writes the network (and optionally a crack result) as GeoPackage files
`filename_edges.gpkg` and `filename_vertices.gpkg`. Requires `using GeoDataFrames`.
"""
function writeGPKGFile end

"""
	getGoogleDirection(origin, destination, deptime, key)

Queries the Google Directions API. Requires `using HTTP, JSON`.
"""
function getGoogleDirection end

"""
	getMapOSM(; place, location, radius, nfile, efile, city)

Downloads a street network from OpenStreetMap with Python's osmnx.
Requires `using PyCall` (and osmnx, networkx in that Python).
"""
function getMapOSM end

const EXTENSION_HINTS = Dict{Function, String}(
	writeGPKGFile => "using GeoDataFrames",
	getGoogleDirection => "using HTTP, JSON",
	getMapOSM => "using PyCall",
)

function __init__()
	Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
		hint = get(EXTENSION_HINTS, exc.f, nothing)
		if hint !== nothing && isempty(methods(exc.f))
			print(io, "\n`$(nameof(exc.f))` is provided by a package extension; run `$hint` first.")
		end
	end
end

include("getTravelTimes.jl")
include("buildNetworkFromTables.jl")
include("buildSquareNetwork.jl")
include("buildCityLattice.jl")
include("networkSummary.jl")
include("cellList.jl")
include("create_ODs.jl")
include("crackOptimalPaths.jl")
include("crackStatistics.jl")
include("writeGML.jl")
include("DisorderDist.jl")

export DisorderDist, buildNetworkFromTables, buildSquareNetwork, buildCityLattice,
	networkSummary, integratedRemovals,
	cellList_lat_lon, cellList_euclidean, crackOptimalPaths, crackStatistics,
	writeCrackStatistics, readCrackStatistics, getGoogleDirection, getMapOSM,
	getTravelTimes, create_ODs, writeGML, writeGPKGFile

end
