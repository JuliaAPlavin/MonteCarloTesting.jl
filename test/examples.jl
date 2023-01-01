### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ e1c26815-6a58-4d61-8e1e-567e1f8c94b5
begin
	using Pkg
	eval(:(Pkg.develop(path="..")))
	using MonteCarloTesting
end

# ╔═╡ e3da8cba-8951-48f8-825d-935d127fc5c8
begin
	using RectiGrids
	using Distributions: Poisson
	using Statistics: mean
end

# ╔═╡ ef34f91c-c8cb-4d28-9257-e4d95c828275
using DisplayAs: Text as AsText

# ╔═╡ ab1ba14c-604d-4c42-96fc-eb57fcf94a5d
md"""
Load required packages:
"""

# ╔═╡ 06ac9ba4-c9b1-43e3-987c-0f536a081e03
md"""
Generate some data...
"""

# ╔═╡ f0bbf388-6a3f-4faa-9aaf-85595f57b2dd
md"""
The real/true/actual/measured value:
"""

# ╔═╡ 4129a478-a169-4aed-a843-2c381fd067de
real = range(0.5, 0.6, length=100) |> collect

# ╔═╡ b8193a80-3c6f-4fdc-ade5-c596caad2dcb
md"""
A function to generate random realizations of data under the null hypothesis:
"""

# ╔═╡ b555bdde-303e-40fb-bca8-e5cf61b77933
randomfunc = rng -> rand(rng, 100)

# ╔═╡ 00d7a2db-496a-47b0-b3f1-92544533d915
md"""
Gather everything into a single struct:
"""

# ╔═╡ f0318549-27ee-4f8e-9f21-cd667e69745f
mc1 = montecarlo(; real, randomfunc, nrandom=1000)

# ╔═╡ ce4ebd6c-5325-43d1-bcc7-85c34818979a
md"""
It has some sasic accessors:
"""

# ╔═╡ 8c5ec184-221d-451c-813d-7f0b427efe89
nrandom(mc1)

# ╔═╡ d6fe18ae-d823-437a-aad8-d0d8eb773f2c
realval(mc1)

# ╔═╡ 9eb78b5f-fd49-4841-901f-be7c6de51755
randomvals(mc1)

# ╔═╡ bfd0a34a-47d3-4b67-bbef-875453fe503b
md"""
Compute the average of each (real & random) sample:
"""

# ╔═╡ 9619145e-4ce5-4230-9249-42b5010cc4f8
mc2 = mapsamples(mc1) do xs
	mean(xs)
end

# ╔═╡ b57a7b67-f2f2-477c-826f-8af32fc06f40
md"""
Estimate p-value: probability for the `xs` average to be as high as observed, assuming the null hypothesis.
"""

# ╔═╡ 0ac08d03-ba8a-4816-a690-7b266a4b2cf0
pvalue(mc2, alt= >=)

# ╔═╡ 38a544b8-9f24-4eb5-aecd-7f4fc842d6c2
md"""
Two p-values for alternatives in the opposite directions always sum to `> 1`:
"""

# ╔═╡ 058ebf88-d5cc-4cd5-8c8d-3b446630be58
pvalue(mc2, alt= >=) + pvalue(mc2, alt= <=)

# ╔═╡ 8faae6a7-7d45-49b8-a4f5-dd37183d5aab
md"""
Processing of the dataset can have some free parameters tried in the process. We provide a way to properly account for multiple trials in p-value estimation.

In this simplistic example, the single free parameter is `n`: number of first elements in `xs` to consider. Perform data processing (compute the average), varying `n`:
"""

# ╔═╡ 90af90c8-cb89-4365-b575-5f942f78cbb1
mc3 = map_w_params(mc1, grid(n=10:100)) do xs, ps
	mean(xs[1:ps.n])
end

# ╔═╡ 9152fa0e-399a-4a4a-b632-2919bb15e279
md"""
Compute p-values for each `n`:
"""

# ╔═╡ 368d1664-71f3-4070-bae0-8351243b9b14
pvalue.(mc3, alt= >=) |> AsText

# ╔═╡ 67bcb80a-e1eb-441c-99f7-596fb2ffafdf
md"""
As expected, they are close to 0.5 for small `n`, reaching the original p-value compute on whole vectors for `n = 100`.
"""

# ╔═╡ b475aedc-834b-46bf-8934-99305bbbaffb
md"""
Of course, we can add more parameters:
"""

# ╔═╡ a8dd41ef-c3d4-4735-af20-54c5772022cd
mc5 = map_w_params(mc3, grid(p=2:4, mul=0.1:0.1:2)) do x, ps
	@assert 10 <= ps.n <= 100
	ps.mul * x^ps.p
end

# ╔═╡ 725d9e47-bb28-457c-9384-548558186b19
md"""
... and compute p-values for each combination of them:
"""

# ╔═╡ 67090b29-e406-4d62-8c89-37a969df8101
pvalue.(mc5, alt= >=) |> AsText

# ╔═╡ 32632c1f-a353-4e8b-9a53-8370008a74f9
md"""
Compute the so-called pre- and post-trial p-values (see `pvalue_wtrials()` docs):
"""

# ╔═╡ 3b610570-e0d8-49f5-be69-5f8ae1029a94
pvalue_wtrials(mc5; alt= >=)

# ╔═╡ 4041e581-6b69-419e-a429-08762c3584f0
md"""
Parametrized Monte-Carlo containers can be filtered to a subset of parameter values using `AxisKeys` interface:
"""

# ╔═╡ ae02c6a9-4b96-4cbe-a4ee-cce5504c3459
pvalue_wtrials(mc5(n=90:100); alt= >=)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DisplayAs = "0b91fe84-8a4c-11e9-3e1d-67c38462b6d6"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
MonteCarloTesting = "b75b2f39-c526-438a-aeb3-f18deacfdc57"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
RectiGrids = "8ac6971d-971d-971d-971d-971d5ab1a71a"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DisplayAs = "~0.1.2"
Distributions = "~0.25.37"
RectiGrids = "~0.1.6"
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
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "1ee88c4c76caa995a885dc2f22a5d548dfbbc0ba"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisKeys]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "IntervalSets", "InvertedIndices", "LazyStack", "LinearAlgebra", "NamedDims", "OffsetArrays", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "8fd46bee70b52bfc11d90f5db51b33bda9c148df"
uuid = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
version = "0.1.24"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "926870acb6cbcf029396f2f2de030282b6bc1941"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.4"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "a3e070133acab996660d31dcf479ea42849e368f"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

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

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Dictionaries]]
deps = ["Indexing", "Random"]
git-tree-sha1 = "66bde31636301f4d217a161cabe42536fa754ec8"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.17"

[[deps.DisplayAs]]
git-tree-sha1 = "44e8d47bc0b56ec09115056a692e5fa0976bfbff"
uuid = "0b91fe84-8a4c-11e9-3e1d-67c38462b6d6"
version = "0.1.2"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "6a8dc9f82e5ce28279b6e3e2cea9421154f5bd0d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.37"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "3fe985505b4b667e1ae303c9ca64d181f09d5c05"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.3"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

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

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

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

[[deps.MonteCarloTesting]]
deps = ["AxisKeys", "Distributions", "EllipsisNotation", "InteractiveUtils", "LazyStack", "Markdown", "Parameters", "RectiGrids", "SplitApplyCombine", "Statistics", "StatsBase"]
path = "../../home/aplavin/work/work/neutrino_events/dev/MonteCarloTesting"
uuid = "b75b2f39-c526-438a-aeb3-f18deacfdc57"
version = "0.1.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "88dce79529a358f6efd13225d131bec958a18f1d"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.43"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

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
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

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
git-tree-sha1 = "f7e83e8dcc2b6e78d09331720d75cc37355d122f"
uuid = "8ac6971d-971d-971d-971d-971d5ab1a71a"
version = "0.1.6"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "8f82019e525f4d5c669692772a6f4b0a58b06a6a"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.2.0"

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
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e08890d19787ec25029113e88c34ec20cac1c91e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.0.0"

[[deps.SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "dec0812af1547a54105b4a6615f341377da92de6"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.2.0"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "7f5a513baec6f122401abfc8e9c074fdac54f6c1"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

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
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

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
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

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
# ╟─ab1ba14c-604d-4c42-96fc-eb57fcf94a5d
# ╠═e1c26815-6a58-4d61-8e1e-567e1f8c94b5
# ╠═e3da8cba-8951-48f8-825d-935d127fc5c8
# ╠═ef34f91c-c8cb-4d28-9257-e4d95c828275
# ╟─06ac9ba4-c9b1-43e3-987c-0f536a081e03
# ╟─f0bbf388-6a3f-4faa-9aaf-85595f57b2dd
# ╠═4129a478-a169-4aed-a843-2c381fd067de
# ╟─b8193a80-3c6f-4fdc-ade5-c596caad2dcb
# ╠═b555bdde-303e-40fb-bca8-e5cf61b77933
# ╟─00d7a2db-496a-47b0-b3f1-92544533d915
# ╠═f0318549-27ee-4f8e-9f21-cd667e69745f
# ╟─ce4ebd6c-5325-43d1-bcc7-85c34818979a
# ╠═8c5ec184-221d-451c-813d-7f0b427efe89
# ╠═d6fe18ae-d823-437a-aad8-d0d8eb773f2c
# ╠═9eb78b5f-fd49-4841-901f-be7c6de51755
# ╟─bfd0a34a-47d3-4b67-bbef-875453fe503b
# ╠═9619145e-4ce5-4230-9249-42b5010cc4f8
# ╟─b57a7b67-f2f2-477c-826f-8af32fc06f40
# ╠═0ac08d03-ba8a-4816-a690-7b266a4b2cf0
# ╟─38a544b8-9f24-4eb5-aecd-7f4fc842d6c2
# ╠═058ebf88-d5cc-4cd5-8c8d-3b446630be58
# ╟─8faae6a7-7d45-49b8-a4f5-dd37183d5aab
# ╠═90af90c8-cb89-4365-b575-5f942f78cbb1
# ╟─9152fa0e-399a-4a4a-b632-2919bb15e279
# ╠═368d1664-71f3-4070-bae0-8351243b9b14
# ╟─67bcb80a-e1eb-441c-99f7-596fb2ffafdf
# ╟─b475aedc-834b-46bf-8934-99305bbbaffb
# ╠═a8dd41ef-c3d4-4735-af20-54c5772022cd
# ╟─725d9e47-bb28-457c-9384-548558186b19
# ╠═67090b29-e406-4d62-8c89-37a969df8101
# ╟─32632c1f-a353-4e8b-9a53-8370008a74f9
# ╠═3b610570-e0d8-49f5-be69-5f8ae1029a94
# ╟─4041e581-6b69-419e-a429-08762c3584f0
# ╠═ae02c6a9-4b96-4cbe-a4ee-cce5504c3459
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
