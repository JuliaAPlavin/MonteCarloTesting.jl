### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ fb5e9a8f-2244-466d-b42f-0bbeb5fdd59f
using Distributions

# ╔═╡ b985e8b8-0230-48ce-a354-7270b04932fe
using AxisKeysExtra

# ╔═╡ ce600804-9c41-4bce-a206-2acfe6818a4a
using DataPipes

# ╔═╡ 3273ef17-79aa-48af-8278-b894bcba0c5e
using Tables

# ╔═╡ 44fe2fbc-4f34-45dd-b56b-830f7235a4a2
using PyPlotUtils; pyplot_style!()

# ╔═╡ 0ea5cf1e-da96-4da5-aefa-da0a97bf3645
using PlutoUI

# ╔═╡ e1c26815-6a58-4d61-8e1e-567e1f8c94b5
begin
	# using Revise
	# using Pkg
	# eval(:(Pkg.develop(path="..")))
	using MonteCarloTesting
end

# ╔═╡ 298a0dea-f330-4240-930e-650a946d27ee
using RectiGrids

# ╔═╡ f51f284d-1669-4f6c-a128-0e0ec911ac82
using Statistics: mean

# ╔═╡ ef34f91c-c8cb-4d28-9257-e4d95c828275
using DisplayAs: Text as AsText

# ╔═╡ 1b9f3c54-12f9-4668-b851-e375020947ef
using Statistics

# ╔═╡ a39e055d-7c16-4ada-9ac1-8981be6b2c5c
using StatsBase

# ╔═╡ 0916ff0a-6486-4062-8e4b-0758838ac2da
md"""
!!! info "MonteCarloTesting.jl"
	Statistical hypothesis testing via Monte Carlo simulations.

`MonteCarloTesting.jl` performs hypothesis testing in cases without analytic solutions relying on simulations instead. The only requirement is to be able to generate synthetic data under the null hypothesis.

In this notebook, we walk through examples that do have analytic solutions as well, for ease of comparison.
"""

# ╔═╡ 6e703f42-acea-44ab-91ca-b668b097282a
md"""
# Examples
## Simple p-value computation
"""

# ╔═╡ f0bbf388-6a3f-4faa-9aaf-85595f57b2dd
md"""
Let's start with a very simple example. This is our real/true/actual/measured value:
"""

# ╔═╡ 468b8932-1ff6-4810-9ad2-94b841d7afd9
real = 1.0

# ╔═╡ c3ad595d-87cc-42d1-8853-09c3356c5752
md"""
The null hypothesis is the standard normal distribution. We only need to tell `MonteCarloTesting` how to generate values from it:
"""

# ╔═╡ b3a24d98-ed1d-40ce-8a90-2a42bca866f2
randomfunc = rng -> randn(rng)

# ╔═╡ 69330de7-b203-48a3-83ac-e277ca21ac8d
md"""
Gather the real value and the null generator into the MC samples struct:
"""

# ╔═╡ c80a5559-7ee7-4321-b7d5-6038223f262f
mc = montecarlo(; real, randomfunc, nrandom=1000)

# ╔═╡ 74ff85c3-7eed-4b7c-90e9-ab6bff4104bd
md"""
It is helpful to expand and examine the struct in Pluto. It basically contains the real value, and the array of random values. This random array is a lazy `MappedArray` above, but can be a plain array as well.
"""

# ╔═╡ 07323a0b-4156-4a8f-9817-52a1ab51a994
md"""
This samples struct has some basic accessors:
"""

# ╔═╡ f65b12f1-5329-45b8-9807-5e8a1b106543
nrandom(mc)

# ╔═╡ 9da2d3c2-a390-4eba-b65e-f2fb8a2e09ec
realval(mc)

# ╔═╡ 314adf4f-41dd-4e36-be4d-c8a1fadbb97c
randomvals(mc)

# ╔═╡ 55eb54a5-b483-49da-ad31-51afd248970f
md"""
`MonteCarloTesting` helps calculating the p-value under the null hypothesis from the samples. You only specify the alternative hypothesis direction --- typically, `>=` or `<=`:
"""

# ╔═╡ 39f4d6fa-a1be-42db-b5ab-34fbe607fcde
pvalue(mc; alt = >=)

# ╔═╡ fa582689-c152-4bc1-98e4-bd5491aaaefc
md"""
Indeed, the probability for a standard normal variable to exceed 1 is about 16%.
"""

# ╔═╡ cff93ffd-cbe7-4fa3-8fe4-286fcbe3c16d
md"""
The p-value is never anticonservative on average. For example, when the true p-value is small and cannot be resolved with the specified number of samples, the result is about `1/nrandom`:
"""

# ╔═╡ 04a653b2-5171-4bf9-bfe8-66ed385e3813
pvalue(montecarlo(; real=100., randomfunc, nrandom=1000); alt = >=)

# ╔═╡ 1d927a6d-7963-4496-b5b1-d92bf90ce7f8
md"""
Reported p-values for two complementary alternative hypotheses are always greater than 1, by about the same `1/nrandom` fraction:
"""

# ╔═╡ e594d327-8e20-4474-a402-0738182a4a0b
pvalue(mc, alt = >=) + pvalue(mc, alt = <=)

# ╔═╡ 135f22bc-a137-48db-90c8-c012d452f6db
md"""
## Typical usecase
### Compute test statistic
"""

# ╔═╡ 2c5b97b9-d85d-42bd-b50c-c85f3a96119f
md"""
In this example, we assume that the measurement is a vector of 100 values.

The null hypothesis is that each element comes from the standard uniform distribution:
"""

# ╔═╡ f0318549-27ee-4f8e-9f21-cd667e69745f
mc_vec = montecarlo(;
	real=range(0.5, 0.7, length=100) .+ rand(Normal(0, 0.001), 100),
	randomfunc=rng -> rand(rng, 100),
	nrandom=10^4
)

# ╔═╡ bfd0a34a-47d3-4b67-bbef-875453fe503b
md"""
Now we need to choose a test statistic that distills our measurement (a vector) into a single number. `MonteCarloTesting` then maps this function over all --- real and random --- samples.

Let's say we want to use the average as the test statistic:
"""

# ╔═╡ 9619145e-4ce5-4230-9249-42b5010cc4f8
mc_avg = mapsamples(mc_vec) do xs
	mean(xs)
end

# ╔═╡ b57a7b67-f2f2-477c-826f-8af32fc06f40
md"""
Now each sample is a single number, and we can estimate the p-value --- probability for the measurement average to be as high as observed, assuming the null hypothesis:
"""

# ╔═╡ 0ac08d03-ba8a-4816-a690-7b266a4b2cf0
pvalue(mc_avg, alt= >=)

# ╔═╡ fce1d1b3-1fe8-43ae-8a35-136d9b8fca5a
md"""
### Perform multiple trials
"""

# ╔═╡ 8faae6a7-7d45-49b8-a4f5-dd37183d5aab
md"""
Processing of the dataset can have multiple parameters tried in the process. For example, our data collection process may suggest that we are most likely to find differences from the null, if any, closer to the end of the measurement vector.

It may seem intuitive to just scan over all tails of the vector, compute p-values, and choose the lowest of them. However, this approach suffers from the [multiple comparisons problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem). `MonteCarloTesting` provides a way to properly account for multiple trials in p-value estimation.

In this example, the single free parameter is `n`: number of last elements in `xs` to consider when taking the average. Perform data processing of our vectors --- compute the average for different values of the `n` parameter:
"""

# ╔═╡ 90af90c8-cb89-4365-b575-5f942f78cbb1
mc_avg_n = map_w_params(mc_vec, grid(n=1:100)) do xs, ps
	mean(@view xs[end-ps.n+1:end])
end

# ╔═╡ 3a134034-8e10-416c-ac99-1667f707fa63
md"""
The returned struct is a thin wrapper around a keyed array, and implements corresponding methods where it makes sense:
"""

# ╔═╡ 38bb492f-aaad-4758-bac5-f1dbea55ba48
named_axiskeys(mc_avg_n)

# ╔═╡ 8b12a11c-5c41-461b-a286-79ad46313188
mc_avg_n(n=10)

# ╔═╡ 9152fa0e-399a-4a4a-b632-2919bb15e279
md"""
Compute p-values for each `n` by broadcasting:
"""

# ╔═╡ 368d1664-71f3-4070-bae0-8351243b9b14
pvalue.(mc_avg_n, alt= >=) |> AsText

# ╔═╡ 67bcb80a-e1eb-441c-99f7-596fb2ffafdf
md"""
As expected, p-values are large for small `n`, and reach exactly the original p-value computed on whole vectors at `n = 100`.
"""

# ╔═╡ b475aedc-834b-46bf-8934-99305bbbaffb
md"""
Of course, we can add more parameters on top:
"""

# ╔═╡ a8dd41ef-c3d4-4735-af20-54c5772022cd
mc_fancy = map_w_params(mc_avg_n, grid(p=2:4, mul=0.1:0.1:2)) do x, ps
	ps.mul * x^ps.p
end

# ╔═╡ 725d9e47-bb28-457c-9384-548558186b19
md"""
... and compute p-values for each combination of them:
"""

# ╔═╡ 67090b29-e406-4d62-8c89-37a969df8101
pvalue.(mc_fancy, alt= >=) |> AsText

# ╔═╡ 32632c1f-a353-4e8b-9a53-8370008a74f9
md"""
The minimum of p-values for different parameter values is often called the pre-trial p-value:
"""

# ╔═╡ dd3e706b-24b1-445b-910c-4795286e7f49
pvalue.(mc_avg_n; alt= >=) |> with_axiskeys(findmin)

# ╔═╡ 3555cc0c-5b1d-4639-aa63-1820c4d97160
md"""
This is not the actual chance probability because of the multiple comparisons issue.

The post-trial p-value is the actual probability to obtain the pre-trial p-value as low as it is observed:
"""

# ╔═╡ 3b610570-e0d8-49f5-be69-5f8ae1029a94
pvalue_post(mc_avg_n; alt= >=)

# ╔═╡ 50ea2a39-de44-44ef-94ea-34fdc22ad885
md"""
Note that it is somewhat higher than the original p-value where we didn't perform any trials and averaged the whole vector:
"""

# ╔═╡ d7eddf63-8d7d-4de8-b646-d2f6d23f9bac
pvalue(mc_avg, alt= >=)

# ╔═╡ 7fbd73a1-95a2-42f5-bf9c-6d4eec5a9119
md"""
## Evaluating p-value correctness and more
"""

# ╔═╡ 6f2f83f1-d40f-4e90-9c78-110bffa4d17c
pvals = map(1:10^5) do _
	mc = montecarlo(real=rand(), randomfunc=rng -> rand(rng), nrandom=100)
	pvalue(mc; alt= >=)
end

# ╔═╡ 3ae83929-255f-488e-9100-c2cd3bbacde9
pvals_trials = map(grid(dep=[:complete, :partial, :none], nrandom=[100], eff=[0, 0.1])) do p
	alt = >=
	ThreadsX.map(1:10^4) do _
		mc = montecarlo(;real=rand(10) .+ p.eff, randomfunc=rng -> rand(rng, 10), p.nrandom)
		mcp = map_w_params(mc, grid(p=1:10)) do s, pp
			if p.dep == :complete
				s[1]
			elseif p.dep == :partial
				mean(view(s, 1:pp.p))
			elseif p.dep == :none
				s[pp.p]
			end
		end
		(
			pretrial=minimum(pvalue.(mcp; alt)),
			harmmean=harmmean(pvalue.(mcp; alt)),
			posttrial_minimum=pvalue_post(mcp; alt, combine=minimum),
			posttrial_harmmean=pvalue_post(mcp; alt, combine=harmmean),
			posttrial_prod=pvalue_post(mcp; alt, combine=prod),
		)
	end
end

# ╔═╡ 2bfa1528-67a9-4ee5-a928-1e8cd365ef36
function plot_pvals(pvals)
	common(ax) = let
		ax.fill_between([0, 1], [0, 1], [1, 1], color=:r, alpha=0.17)
		ax.fill_between([0, 1], [0, 0], [0, 1], color=:g, alpha=0.2)
		ax.plot(sort(pvals), range(0, 1, length=length(pvals)), color=:k)
		ax.axvline(median(pvals); ls="--")
		ax.set_aspect(:equal)
	end
	
	plt.figure()
	common(plt.gca())
	set_xylims((0..1)^2)

	xylabels("P-value", "Probability of lower p-value")

	full_ax = plt.gca()
	ins_ax = full_ax.inset_axes([0.55, 0.05, 0.4, 0.4])
	common(ins_ax)
	set_xylims((0..0.05)^2; ax=ins_ax)
	ins_ax.set_xticklabels([])
	ins_ax.set_yticklabels([])

	ins_ax.patch.set_edgecolor("0.7")
	ins_ax.patch.set_linewidth(3)

	full_ax.indicate_inset_zoom(ins_ax, edgecolor="black")
	
	plt.gcf()
end

# ╔═╡ 3a8e8672-ceee-401d-8953-fed27919e2b2
plot_pvals(pvals)

# ╔═╡ b0ccd5f9-d4cd-4dd7-953d-328abca78ad0
map([:pretrial, :harmmean, :posttrial_minimum, :posttrial_harmmean, :posttrial_prod]) do k
	k => map(pvals_trials |> rowtable) do r
		plot_pvals(k.(r.value))
		plt.title("dep: $(r.dep)")
		plt.gcf()
	end
end

# ╔═╡ 8ee73b57-db1f-4c94-89aa-544f1830c531


# ╔═╡ 38f499c8-bac1-4583-a414-03ab0f2f71c9
# import ThreadsX

# ╔═╡ f18b9502-e6ef-4530-9570-7fea1f1a1572
TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AxisKeysExtra = "b7a0d2b7-1990-46dc-b5dd-87820ecd1b09"
DataPipes = "02685ad9-2d12-40c3-9f73-c6aeda6a7ff5"
DisplayAs = "0b91fe84-8a4c-11e9-3e1d-67c38462b6d6"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
MonteCarloTesting = "b75b2f39-c526-438a-aeb3-f18deacfdc57"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyPlotUtils = "5384e752-6c47-47b3-86ac-9d091b110b31"
RectiGrids = "8ac6971d-971d-971d-971d-971d5ab1a71a"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[compat]
AxisKeysExtra = "~0.1.0"
DataPipes = "~0.2.5"
DisplayAs = "~0.1.2"
Distributions = "~0.25.68"
MonteCarloTesting = "~0.1.0"
PlutoUI = "~0.7.39"
PyPlotUtils = "~0.1.4"
RectiGrids = "~0.1.13"
StatsBase = "~0.33.14"
Tables = "~1.7.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "485c0473a21ed60863da0c71dbff2e6b6b6f2e94"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "8557017cfc7b58baea05a43ed35538857e6d35b4"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.19"

[[deps.AccessorsExtra]]
deps = ["Accessors", "ConstructionBase", "InverseFunctions", "Reexport", "Requires"]
git-tree-sha1 = "ef45a3c71f3a7e98a107ec66222e04250185c7bb"
uuid = "33016aad-b69d-45be-9359-82a41f556fd4"
version = "0.1.9"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisKeys]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "IntervalSets", "InvertedIndices", "LazyStack", "LinearAlgebra", "NamedDims", "OffsetArrays", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "88cc6419032d0e3ea69bc65d012aa82302774ab8"
uuid = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
version = "0.2.7"

[[deps.AxisKeysExtra]]
deps = ["AxisKeys", "Compat", "RectiGrids", "Reexport", "SplitApplyCombine", "StructArrays"]
git-tree-sha1 = "2d6d9b3fff06c813935122c9af951fbf53b00cd0"
uuid = "b7a0d2b7-1990-46dc-b5dd-87820ecd1b09"
version = "0.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "5856d3031cdb1f3b2b6340dfdc66b6d9a149a374"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.2.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "d5b014b216dc891e81fea299638e4c10c657b582"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.2"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "3c8de95b4e932d76ec8960e12d681eba580e9674"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.8"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataPipes]]
deps = ["Accessors", "SplitApplyCombine"]
git-tree-sha1 = "ab6b5bf476e9111b0166cc3f8373638204d7fafd"
uuid = "02685ad9-2d12-40c3-9f73-c6aeda6a7ff5"
version = "0.2.17"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "96dc5c5c8994be519ee3420953c931c55657a3f2"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.24"

[[deps.DirectionalStatistics]]
deps = ["AccessorsExtra", "IntervalSets", "InverseFunctions", "LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "156365de4369a6cf587d0d59ce52fe688f2b5f92"
uuid = "e814f24e-44b0-11e9-2fd5-aba2b6113d95"
version = "0.1.19"

[[deps.DisplayAs]]
git-tree-sha1 = "43c017d5dd3a48d56486055973f443f8a39bb6d9"
uuid = "0b91fe84-8a4c-11e9-3e1d-67c38462b6d6"
version = "0.1.6"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "334a5896c1534bb1aa7aa2a642d30ba7707357ef"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.68"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "dc45fbbe91d6d17a8e187abad39fb45963d97388"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.5.13"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "076bb0da51a8c8d1229936a1af7bdfacd65037e1"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.2"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

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
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LazyStack]]
deps = ["ChainRulesCore", "LinearAlgebra", "NamedDims", "OffsetArrays"]
git-tree-sha1 = "2eb4a5bf2eb0519ebf40c797ba5637d327863637"
uuid = "1fad7336-0346-5a1a-a56f-a06ba010965b"
version = "0.0.8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

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
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MonteCarloTesting]]
deps = ["AxisKeys", "AxisKeysExtra", "InteractiveUtils", "Markdown", "Parameters", "Random", "RectiGrids", "Requires", "SplitApplyCombine", "Statistics", "StatsBase"]
git-tree-sha1 = "438b9576df188b069acefd4b43bc9b3b3af3e52c"
uuid = "b75b2f39-c526-438a-aeb3-f18deacfdc57"
version = "0.1.8"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "f39537cbe1cf4f407e65bdf7aca6b04f5877fbb1"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "1.1.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.NonNegLeastSquares]]
deps = ["Distributed", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "1271344271ffae97e2855b0287356e6ea5c221cc"
uuid = "b7351bd1-99d9-5c5d-8786-f205a815c4d7"
version = "0.4.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

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
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "3d5bf43e3e8b412656404ed9466f1dcbf7c50269"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "53b8b07b721b77144a0fbbbc2675222ebf40a02d"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.94.1"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "f9d953684d4d21e947cb6d642db18853d43cb027"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.0"

[[deps.PyPlotUtils]]
deps = ["Accessors", "AxisKeys", "Colors", "DataPipes", "DirectionalStatistics", "DomainSets", "IntervalSets", "LinearAlgebra", "NonNegLeastSquares", "PyCall", "PyPlot", "StatsBase", "Unitful"]
git-tree-sha1 = "37b811018ec1ebde8e4698c118b72f90ed2ec275"
uuid = "5384e752-6c47-47b3-86ac-9d091b110b31"
version = "0.1.17"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RectiGrids]]
deps = ["AxisKeys", "ConstructionBase", "Random", "StaticArraysCore"]
git-tree-sha1 = "940a23a7472b7352ff0ebbb661da8bbb5d7f932f"
uuid = "8ac6971d-971d-971d-971d-971d5ab1a71a"
version = "0.1.13"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

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
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

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
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "48f393b0231516850e39f6c756970e7ca8b77045"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.2.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "dfec37b90740e3b9aa5dc2613892a3fc155c3b42"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.6"

[[deps.StaticArraysCore]]
git-tree-sha1 = "ec2bd695e905a3c755b33026954b119ea17f2d22"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.3.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "b649200e887a487468b71821e2644382699f1b0f"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.11.0"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─0916ff0a-6486-4062-8e4b-0758838ac2da
# ╟─6e703f42-acea-44ab-91ca-b668b097282a
# ╟─f0bbf388-6a3f-4faa-9aaf-85595f57b2dd
# ╠═468b8932-1ff6-4810-9ad2-94b841d7afd9
# ╟─c3ad595d-87cc-42d1-8853-09c3356c5752
# ╠═b3a24d98-ed1d-40ce-8a90-2a42bca866f2
# ╟─69330de7-b203-48a3-83ac-e277ca21ac8d
# ╠═c80a5559-7ee7-4321-b7d5-6038223f262f
# ╟─74ff85c3-7eed-4b7c-90e9-ab6bff4104bd
# ╟─07323a0b-4156-4a8f-9817-52a1ab51a994
# ╠═f65b12f1-5329-45b8-9807-5e8a1b106543
# ╠═9da2d3c2-a390-4eba-b65e-f2fb8a2e09ec
# ╠═314adf4f-41dd-4e36-be4d-c8a1fadbb97c
# ╟─55eb54a5-b483-49da-ad31-51afd248970f
# ╠═39f4d6fa-a1be-42db-b5ab-34fbe607fcde
# ╟─fa582689-c152-4bc1-98e4-bd5491aaaefc
# ╟─cff93ffd-cbe7-4fa3-8fe4-286fcbe3c16d
# ╠═04a653b2-5171-4bf9-bfe8-66ed385e3813
# ╟─1d927a6d-7963-4496-b5b1-d92bf90ce7f8
# ╠═e594d327-8e20-4474-a402-0738182a4a0b
# ╟─135f22bc-a137-48db-90c8-c012d452f6db
# ╟─2c5b97b9-d85d-42bd-b50c-c85f3a96119f
# ╠═f0318549-27ee-4f8e-9f21-cd667e69745f
# ╟─bfd0a34a-47d3-4b67-bbef-875453fe503b
# ╠═9619145e-4ce5-4230-9249-42b5010cc4f8
# ╟─b57a7b67-f2f2-477c-826f-8af32fc06f40
# ╠═0ac08d03-ba8a-4816-a690-7b266a4b2cf0
# ╟─fce1d1b3-1fe8-43ae-8a35-136d9b8fca5a
# ╟─8faae6a7-7d45-49b8-a4f5-dd37183d5aab
# ╠═90af90c8-cb89-4365-b575-5f942f78cbb1
# ╟─3a134034-8e10-416c-ac99-1667f707fa63
# ╠═38bb492f-aaad-4758-bac5-f1dbea55ba48
# ╠═8b12a11c-5c41-461b-a286-79ad46313188
# ╟─9152fa0e-399a-4a4a-b632-2919bb15e279
# ╠═368d1664-71f3-4070-bae0-8351243b9b14
# ╟─67bcb80a-e1eb-441c-99f7-596fb2ffafdf
# ╟─b475aedc-834b-46bf-8934-99305bbbaffb
# ╠═a8dd41ef-c3d4-4735-af20-54c5772022cd
# ╟─725d9e47-bb28-457c-9384-548558186b19
# ╠═67090b29-e406-4d62-8c89-37a969df8101
# ╟─32632c1f-a353-4e8b-9a53-8370008a74f9
# ╠═dd3e706b-24b1-445b-910c-4795286e7f49
# ╟─3555cc0c-5b1d-4639-aa63-1820c4d97160
# ╠═3b610570-e0d8-49f5-be69-5f8ae1029a94
# ╟─50ea2a39-de44-44ef-94ea-34fdc22ad885
# ╠═d7eddf63-8d7d-4de8-b646-d2f6d23f9bac
# ╟─7fbd73a1-95a2-42f5-bf9c-6d4eec5a9119
# ╠═6f2f83f1-d40f-4e90-9c78-110bffa4d17c
# ╠═3ae83929-255f-488e-9100-c2cd3bbacde9
# ╠═3a8e8672-ceee-401d-8953-fed27919e2b2
# ╠═b0ccd5f9-d4cd-4dd7-953d-328abca78ad0
# ╠═2bfa1528-67a9-4ee5-a928-1e8cd365ef36
# ╠═8ee73b57-db1f-4c94-89aa-544f1830c531
# ╠═fb5e9a8f-2244-466d-b42f-0bbeb5fdd59f
# ╠═38f499c8-bac1-4583-a414-03ab0f2f71c9
# ╠═b985e8b8-0230-48ce-a354-7270b04932fe
# ╠═ce600804-9c41-4bce-a206-2acfe6818a4a
# ╠═3273ef17-79aa-48af-8278-b894bcba0c5e
# ╠═44fe2fbc-4f34-45dd-b56b-830f7235a4a2
# ╠═0ea5cf1e-da96-4da5-aefa-da0a97bf3645
# ╠═f18b9502-e6ef-4530-9570-7fea1f1a1572
# ╠═e1c26815-6a58-4d61-8e1e-567e1f8c94b5
# ╠═298a0dea-f330-4240-930e-650a946d27ee
# ╠═f51f284d-1669-4f6c-a128-0e0ec911ac82
# ╠═ef34f91c-c8cb-4d28-9257-e4d95c828275
# ╠═1b9f3c54-12f9-4668-b851-e375020947ef
# ╠═a39e055d-7c16-4ada-9ac1-8981be6b2c5c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
