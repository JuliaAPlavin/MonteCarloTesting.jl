### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ e3da8cba-8951-48f8-825d-935d127fc5c8
begin
	using RectiGrids
	using Statistics: mean
	using Distributions: fit, cdf, ccdf, Poisson
	using Parameters
	using SplitApplyCombine
	using StatsBase: competerank
	import AxisKeys
end

# ╔═╡ 9453efa7-2881-4200-8e2d-3383f263f137
using LazyStack

# ╔═╡ 4270034c-17c1-11ec-031b-9b819327fadf
Base.@kwdef struct MCSamples{T, AT <: AbstractArray{T}}
	real::T
	random::AT
end

# ╔═╡ 8cbbf270-edbf-4e07-9260-0af3aed00e1d
begin
	realval(mc::MCSamples) = mc.real
	randomvals(mc::MCSamples) = mc.random
	realrandomvals(mc::MCSamples) = [[mc.real]; mc.random]
end

# ╔═╡ 530bbae8-e832-44c7-8b5a-bdc062cb5bef
function montecarlo(; real, random)
	MCSamples(; real, random)
end

# ╔═╡ acbb7ce8-c1ec-4c15-a2ac-d6d48e97b212
struct Fraction end

# ╔═╡ 7b167e1d-0fe0-44a1-af35-6e76748bb928
mct = montecarlo(real=0, random=[-1, 0, 0, 0, 0, 0, 0, 1, 1])

# ╔═╡ c081871f-2359-4cd4-a5cb-49c0a9688ad3
function mapsamples(f, mcs::MCSamples; mapfunc=map)
	return MCSamples(;
		real=f(realval(mcs)),
		random=mapfunc(f, randomvals(mcs)),
	)
end

# ╔═╡ f0318549-27ee-4f8e-9f21-cd667e69745f
mc1 = montecarlo(
	real=range(0.5, 0.6, length=100) |> collect,
	random=[rand(100) for _ in 1:1000]
)

# ╔═╡ cd73f805-e7aa-4d56-92d9-f65011a2f1cd
function swap_realval(mc::MCSamples, randix::Int)
	MCSamples(;
		real=mc.random[randix],
		random=[mc.random[begin:randix-1]; [mc.real]; mc.random[randix+1:end]],
	)
end

# ╔═╡ 32632c1f-a353-4e8b-9a53-8370008a74f9


# ╔═╡ f5474665-42e5-4b9f-989e-a5699e1b3107
Base.first(ka::KeyedArray) = first(parent(ka))

# ╔═╡ 20dfa82e-0f3d-4972-b69d-7a4293deee8a
begin
	@with_kw struct MCSamplesMulti{A <: AbstractArray{<:MCSamples}}
		arr::A
		
		@assert length(unique(nrandom.(arr))) == 1
	end
	
	nrandom(mc::MCSamples) = length(randomvals(mc))
	nrandom(mcm::MCSamplesMulti) = nrandom(first(mcm.arr))

	Base.size(mcm::MCSamplesMulti) = size(mcm.arr)
	Base.getindex(mcm::MCSamplesMulti, I::Int...) = mcm.arr[I...]
	Base.getindex(mcm::MCSamplesMulti, I...) = MCSamplesMulti(mcm.arr[I...])
	(mcm::MCSamplesMulti)(args...; kwargs...) = MCSamplesMulti(mcm.arr(args...; kwargs...))
end

# ╔═╡ fce8717a-6e89-4270-8cee-038a99142993
begin
	sampletype(::Type{<:MCSamples{T}}) where {T} = T
	sampletype(::Type{<:MCSamplesMulti{A}}) where {A} = sampletype(eltype(A))
	sampletype(mc::Union{<:MCSamples, <:MCSamplesMulti}) = sampletype(typeof(mc))
	Base.eltype(mc::MCSamples) = sampletype(mc)
end

# ╔═╡ cdc0bc47-e0c6-4d5f-8fd6-397b949c97ed
function pvalue(mc::MCSamples, mode::Type{Fraction}=Fraction; alt)
	@assert eltype(mc) <: Real
	@assert alt(realval(mc), realval(mc))
	np = sum(alt.(randomvals(mc), realval(mc)))
	n = length(randomvals(mc))
	return (np + 1) / (n + 1)
end

# ╔═╡ 312f6f64-91bd-481f-b21b-5a189254ef2f
function pvalue(mc::MCSamples, mode::Type{Poisson}; alt)
	@assert eltype(mc) <: Real
	dist = fit(Poisson, randomvals(mc))
	return alt == (>=) ? ccdf(dist, realval(mc) - 1) :
		   alt == (<=) ?  cdf(dist, realval(mc)) :
		   @assert false
end

# ╔═╡ ec957120-17b6-42bf-bc26-4671ec784455
Base.broadcastable(mcm::MCSamplesMulti) = mcm.arr

# ╔═╡ 8c8763dc-1f6c-447e-a80d-09cef3efefc7
function pvalues_all(mc::MCSamples, mode::Type{Fraction}=Fraction; alt)
	@assert eltype(mc) <: Real
	@assert alt(realval(mc), realval(mc))
	ranks = competerank(realrandomvals(mc); lt=!alt)
	nps = length(ranks) + 1 .- ranks
	return KeyedArray(nps ./ length(ranks), 0:nrandom(mc))
end

# ╔═╡ 54829dbe-652e-439e-bf32-3d34930854d8
pvalue(mct; alt= >=), pvalues_all(mct; alt= >=)

# ╔═╡ 992a0f69-cc62-45f7-976c-1329f9dd3dc6
pvalue(mct; alt= <=), pvalues_all(mct; alt= <=)

# ╔═╡ 448e29aa-f1cb-4353-bfca-a372706c30f3
function mapsamples(f, mcm::MCSamplesMulti; mapfunc=map)
	return MCSamplesMulti(
		map(mcm.arr) do mc
			mapsamples(f, mc; mapfunc)
		end
	)
end

# ╔═╡ 9619145e-4ce5-4230-9249-42b5010cc4f8
mc2 = mapsamples(mc1) do xs
	mean(xs)
end

# ╔═╡ 38a544b8-9f24-4eb5-aecd-7f4fc842d6c2
pvalue(mc2, alt= >=), pvalue(mc2, alt= <=)

# ╔═╡ 058ebf88-d5cc-4cd5-8c8d-3b446630be58
@assert pvalue(mc2, alt= >=) + pvalue(mc2, alt= <=) > 1

# ╔═╡ b9db5957-0e03-422f-9d22-cffef74e1195
pvalue(mc2, Fraction, alt= >=)

# ╔═╡ d8b939bc-e550-4f25-a2d7-f32dd74922cf
pvalue(swap_realval(mc2, 12), alt= >=)

# ╔═╡ 6ee783ab-6fdd-472d-9d32-0a2a0cab449f
pvalues_all(mc2; alt= >=)

# ╔═╡ 7e730830-74c2-424b-95cc-c69f026592fb
function map_w_params(f, mcs::MCSamples, params; mapfunc=map)
	return MCSamplesMulti(
		map(KeyedArray(params)) do pars
			mapsamples(mcs; mapfunc) do sample
				f(sample, pars)
			end
		end
	)
end

# ╔═╡ 8faae6a7-7d45-49b8-a4f5-dd37183d5aab
function map_w_params(f, mcm::MCSamplesMulti, params; mapfunc=map)
	return MCSamplesMulti(
		map(mcm.arr) do mc
			map(KeyedArray(params)) do pars
				mapsamples(mc; mapfunc) do sample
					f(sample, pars)
				end
			end
		end |> stack
	)
end

# ╔═╡ 90af90c8-cb89-4365-b575-5f942f78cbb1
mc3 = map_w_params(mc1, grid(n=10:100)) do xs, ps
	mean(xs[1:ps.n])
end

# ╔═╡ 9152fa0e-399a-4a4a-b632-2919bb15e279
mc4 = mapsamples(x -> x^2, mc3)

# ╔═╡ 68117c26-1697-4180-9c86-36c0a12896be
pvalue.(mc4, alt= >=)

# ╔═╡ 368d1664-71f3-4070-bae0-8351243b9b14
pvalue.(mc3, alt= >=)

# ╔═╡ f140cf8a-d089-4291-ae46-155e32c9748a
pvalue.(mc3, Fraction, alt= >=)

# ╔═╡ a8dd41ef-c3d4-4735-af20-54c5772022cd
mc5 = map_w_params(mc3, grid(p=2:4, mul=0.1:0.1:2)) do x, ps
	ps.mul * x^ps.p
end

# ╔═╡ 67090b29-e406-4d62-8c89-37a969df8101
pvalue.(mc5, alt= >=)

# ╔═╡ e4e250a0-be64-4c84-bf4f-47798deb1925
function pvalue_wtrials(mcm::MCSamplesMulti; alt)
	# pvalue for each realization and parameter value:
	ps_all = map(mcm.arr) do mc
		pvalues_all(mc; alt)
	end |> stack

	# pvalue for each realization:
	pretrials = minimum.(splitdims(ps_all, 1))
	@assert length(pretrials) == nrandom(mcm) + 1
	# actual pretrial pvalue:
	pretrial = pretrials[1]
	posttrial = mean(pretrials .<= pretrial)
	(; pretrial, posttrial)
end

# ╔═╡ 3b610570-e0d8-49f5-be69-5f8ae1029a94
pvalue_wtrials(mc5; alt= >=)

# ╔═╡ 498d8ed9-964b-4485-8f10-0623c1d9790d


# ╔═╡ 29d38c61-bd53-4599-ac43-f83b1451ee51
begin
	AxisKeys.axiskeys(mcm::MCSamplesMulti) = axiskeys(mcm.arr)
	AxisKeys.named_axiskeys(mcm::MCSamplesMulti) = named_axiskeys(mcm.arr)
end

# ╔═╡ c18788a4-5989-4b2f-a01b-e7e736ee5fc6
axiskeys(mc5), named_axiskeys(mc5)

# ╔═╡ 28e204c0-156b-4e9d-b43d-a223e4eb02f8
named_axiskeys(mc5(n=100)), mc5(n=100)

# ╔═╡ ae02c6a9-4b96-4cbe-a4ee-cce5504c3459
pvalue_wtrials(mc5(n=90:100); alt= >=)

# ╔═╡ d3337bfb-1b4d-45ac-9cb5-946a1a7bd6cd


# ╔═╡ 384c7d8b-a7db-4622-ab48-df44204d8e81
mcp = montecarlo(real=10, random=rand(Poisson(4), 1000))

# ╔═╡ fb0476c5-6fd1-4145-ad94-bb909b5405cb
pvalue(mcp, alt= >=), pvalue(mcp, alt= <=)

# ╔═╡ 6d674bda-5a19-44e5-9064-363cde17a814
@assert pvalue(mcp, alt= >=) + pvalue(mcp, alt= <=) > 1

# ╔═╡ 24b7cc8e-4db1-4e28-9347-ea75f1612cf9
pvalue(mcp, Poisson, alt= >=), pvalue(mcp, Poisson, alt= <=)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LazyStack = "1fad7336-0346-5a1a-a56f-a06ba010965b"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
RectiGrids = "8ac6971d-971d-971d-971d-971d5ab1a71a"
SplitApplyCombine = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
AxisKeys = "~0.1.19"
Distributions = "~0.25.16"
LazyStack = "~0.0.7"
Parameters = "~0.12.2"
RectiGrids = "~0.1.4"
SplitApplyCombine = "~1.1.4"
StatsBase = "~0.33.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0-rc1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "b8d49c34c3da35f220e7295659cd0bab8e739fed"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.33"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisKeys]]
deps = ["AbstractFFTs", "CovarianceEstimation", "IntervalSets", "InvertedIndices", "LazyStack", "LinearAlgebra", "NamedDims", "OffsetArrays", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "8b382307c6195762a5473ba3522a2830c3014620"
uuid = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
version = "0.1.19"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4ce9393e871aca86cc457d9f66976c3da6902ea7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.4.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "4866e381721b30fac8dda4c8cb1d9db45c8d2994"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.37.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "bc3930158d2be029e90b7c40d1371c4f54fa04db"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.6"

[[deps.DataAPI]]
git-tree-sha1 = "bec2532f8adb82005476c141ec23e921fc20971b"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.8.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Dictionaries]]
deps = ["Indexing", "Random"]
git-tree-sha1 = "6eafb1f48014b50f9b25f8d37cf6684cea01382c"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.11"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "f4efaa4b5157e0cdb8283ae0b5428bc9208436ed"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.16"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "caf289224e622f518c9dbfe832cdafa17d7c80a6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.4"

[[deps.IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[deps.LazyStack]]
deps = ["LinearAlgebra", "NamedDims", "OffsetArrays", "Test", "ZygoteRules"]
git-tree-sha1 = "a8bf67afad3f1ee59d367267adb7c44ccac7fdee"
uuid = "1fad7336-0346-5a1a-a56f-a06ba010965b"
version = "0.0.7"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "fb4530603a1e62aa5ed7569f283d4b47c2a92f61"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.38"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RectiGrids]]
deps = ["AxisKeys", "Random"]
git-tree-sha1 = "6df151d9c6a9964cf315341f9a6ce7a0556b6814"
uuid = "8ac6971d-971d-971d-971d-971d5ab1a71a"
version = "0.1.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[deps.SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "3cdd86a92737fa0c8af19aecb1141e71057dc2db"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.1.4"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "a8f30abc7c64a39d389680b74e749cf33f872a70"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "46d7ccc7104860c38b11966dd1f72ff042f382e4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.10"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═e3da8cba-8951-48f8-825d-935d127fc5c8
# ╠═4270034c-17c1-11ec-031b-9b819327fadf
# ╠═8cbbf270-edbf-4e07-9260-0af3aed00e1d
# ╠═530bbae8-e832-44c7-8b5a-bdc062cb5bef
# ╠═20dfa82e-0f3d-4972-b69d-7a4293deee8a
# ╠═fce8717a-6e89-4270-8cee-038a99142993
# ╠═ec957120-17b6-42bf-bc26-4671ec784455
# ╠═acbb7ce8-c1ec-4c15-a2ac-d6d48e97b212
# ╠═cdc0bc47-e0c6-4d5f-8fd6-397b949c97ed
# ╠═312f6f64-91bd-481f-b21b-5a189254ef2f
# ╠═8c8763dc-1f6c-447e-a80d-09cef3efefc7
# ╠═7b167e1d-0fe0-44a1-af35-6e76748bb928
# ╠═54829dbe-652e-439e-bf32-3d34930854d8
# ╠═992a0f69-cc62-45f7-976c-1329f9dd3dc6
# ╠═c081871f-2359-4cd4-a5cb-49c0a9688ad3
# ╠═448e29aa-f1cb-4353-bfca-a372706c30f3
# ╠═7e730830-74c2-424b-95cc-c69f026592fb
# ╠═f0318549-27ee-4f8e-9f21-cd667e69745f
# ╠═9619145e-4ce5-4230-9249-42b5010cc4f8
# ╠═38a544b8-9f24-4eb5-aecd-7f4fc842d6c2
# ╠═058ebf88-d5cc-4cd5-8c8d-3b446630be58
# ╠═b9db5957-0e03-422f-9d22-cffef74e1195
# ╠═cd73f805-e7aa-4d56-92d9-f65011a2f1cd
# ╠═d8b939bc-e550-4f25-a2d7-f32dd74922cf
# ╠═6ee783ab-6fdd-472d-9d32-0a2a0cab449f
# ╠═90af90c8-cb89-4365-b575-5f942f78cbb1
# ╠═9152fa0e-399a-4a4a-b632-2919bb15e279
# ╠═368d1664-71f3-4070-bae0-8351243b9b14
# ╠═68117c26-1697-4180-9c86-36c0a12896be
# ╠═f140cf8a-d089-4291-ae46-155e32c9748a
# ╠═9453efa7-2881-4200-8e2d-3383f263f137
# ╠═8faae6a7-7d45-49b8-a4f5-dd37183d5aab
# ╠═a8dd41ef-c3d4-4735-af20-54c5772022cd
# ╠═67090b29-e406-4d62-8c89-37a969df8101
# ╠═32632c1f-a353-4e8b-9a53-8370008a74f9
# ╠═f5474665-42e5-4b9f-989e-a5699e1b3107
# ╠═e4e250a0-be64-4c84-bf4f-47798deb1925
# ╠═3b610570-e0d8-49f5-be69-5f8ae1029a94
# ╠═498d8ed9-964b-4485-8f10-0623c1d9790d
# ╠═29d38c61-bd53-4599-ac43-f83b1451ee51
# ╠═c18788a4-5989-4b2f-a01b-e7e736ee5fc6
# ╠═28e204c0-156b-4e9d-b43d-a223e4eb02f8
# ╠═ae02c6a9-4b96-4cbe-a4ee-cce5504c3459
# ╠═d3337bfb-1b4d-45ac-9cb5-946a1a7bd6cd
# ╠═384c7d8b-a7db-4622-ab48-df44204d8e81
# ╠═fb0476c5-6fd1-4145-ad94-bb909b5405cb
# ╠═6d674bda-5a19-44e5-9064-363cde17a814
# ╠═24b7cc8e-4db1-4e28-9347-ea75f1612cf9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
