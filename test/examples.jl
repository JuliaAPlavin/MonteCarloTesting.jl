### A Pluto.jl notebook ###
# v0.19.22

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

# ╔═╡ a39e055d-7c16-4ada-9ac1-8981be6b2c5c
using StatsBase

# ╔═╡ 0916ff0a-6486-4062-8e4b-0758838ac2da
md"""
!!! info "MonteCarloTesting.jl"
	Statistical hypothesis testing via Monte Carlo simulations.

`MonteCarloTesting.jl` performs hypothesis testing based on simulations, even when no analytic solutions are available. The only requirement is to be able to generate synthetic data under the null hypothesis.

In this notebook, we walk through examples that _do_ have analytic solutions for ease of comparison.
"""

# ╔═╡ 6e703f42-acea-44ab-91ca-b668b097282a
md"""
# Examples
## Simple p-value computation
"""

# ╔═╡ f0bbf388-6a3f-4faa-9aaf-85595f57b2dd
md"""
Let's start with a very simple example. Suppose there is an experiment with measurements that have the standard normal distribution under the null hypothesis. We need to tell `MonteCarloTesting` how to generate random values under the null:
"""

# ╔═╡ b05dbced-8909-4e44-aee9-3665c38bad11
randomfunc = rng -> randn(rng)

# ╔═╡ ead1ece0-91a0-4f3a-983c-2e0763918655
md"""
Imagine that actually performing the experiment resulted in a value of `1`:
"""

# ╔═╡ 468b8932-1ff6-4810-9ad2-94b841d7afd9
real = 1.0

# ╔═╡ 69330de7-b203-48a3-83ac-e277ca21ac8d
md"""
This value ($1$) is greater than the expectation under the null ($0$). But how significant is this difference? \
We aim to estimate how likely or unlikely this outcome is to arise by chance: $\mathbb{P}(x ≥ 1)$ under the null.

First, gather the real value and the null generator into the MC samples struct:
"""

# ╔═╡ c80a5559-7ee7-4321-b7d5-6038223f262f
mc = montecarlo(; real, randomfunc, nrandom=999)

# ╔═╡ 74ff85c3-7eed-4b7c-90e9-ab6bff4104bd
md"""
It is helpful to expand and examine the struct in Pluto. It basically contains the real value, and an array of random values.
"""

# ╔═╡ 07323a0b-4156-4a8f-9817-52a1ab51a994
md"""
This samples struct provides some basic accessors:
"""

# ╔═╡ f65b12f1-5329-45b8-9807-5e8a1b106543
nrandom(mc)

# ╔═╡ 9da2d3c2-a390-4eba-b65e-f2fb8a2e09ec
realval(mc)

# ╔═╡ 314adf4f-41dd-4e36-be4d-c8a1fadbb97c
randomvals(mc)

# ╔═╡ 55eb54a5-b483-49da-ad31-51afd248970f
md"""
The `pvalue` function calculates the p-value under the null hypothesis based on the real value and random samples. The alternative hypothesis direction also needs to be specified --- typically, `>=` or `<=`:
"""

# ╔═╡ 39f4d6fa-a1be-42db-b5ab-34fbe607fcde
pvalue(mc; alt = >=)

# ╔═╡ fa582689-c152-4bc1-98e4-bd5491aaaefc
md"""
Indeed, the probability for a standard normal variable to exceed 1 is about 16%.
"""

# ╔═╡ cff93ffd-cbe7-4fa3-8fe4-286fcbe3c16d
md"""
The p-value from `MonteCarloTesting` is never anticonservative on average. For example, when the true p-value is small and cannot be resolved with the specified number of samples, the result is `1/(nrandom+1)`:
"""

# ╔═╡ 04a653b2-5171-4bf9-bfe8-66ed385e3813
pvalue(montecarlo(; real=100., randomfunc, nrandom=999); alt = >=)

# ╔═╡ 1d927a6d-7963-4496-b5b1-d92bf90ce7f8
md"""
Reported p-values for two complementary alternative hypotheses are always greater than 1, by about the same `1/(nrandom+1)` fraction:
"""

# ╔═╡ e594d327-8e20-4474-a402-0738182a4a0b
pvalue(mc, alt = >=) + pvalue(mc, alt = <=)

# ╔═╡ 135f22bc-a137-48db-90c8-c012d452f6db
md"""
## Typical usecase
### Basic
"""

# ╔═╡ 2c5b97b9-d85d-42bd-b50c-c85f3a96119f
md"""
Now consider a more relevant example where analytic solution is harder to derive. Positions of 100 points in the $[0, 1]$ interval are measured, and we want to check whether they cluster together compared to the uniform distribution. \
A benefit of simulation testing here is that it directly generalizes with little effort to any kinds of space (2d region, sphere, ...) and other arbitrary conditions.

For this example, we craft the measured dataset so that there is some clustering.
"""

# ╔═╡ 0bd6accd-415d-4097-862c-89ebcfaf3ccd
md"""
Visualize real measurements and a few random realization:
"""

# ╔═╡ 4d5ad9d1-2b12-4cd3-a4c1-c304017b089e
md"""
Some clustering is apparent, but the question is how significant --- what's the probability this could arise by chance?
"""

# ╔═╡ da95f93f-fac2-4afa-8337-bdfddcfa150b
md"""
First, suppose we are interested in clustering at the fixed scale of $0.1$. Let's take the fraction of pairs closer than $0.1$ as the test statistic:
"""

# ╔═╡ 71e2becb-e482-485a-992c-975c8744ea97
md"""
The p-value is small, so we can say that clustering is significant indeed!
"""

# ╔═╡ fce1d1b3-1fe8-43ae-8a35-136d9b8fca5a
md"""
### Multiple trials
"""

# ╔═╡ 452512e6-f7ec-423f-924e-f924d9fa0778
md"""
But what if we don't want to focus on any clustering scale apriori, and are interested in any kind of clustering? \
We can perform the same computation as above for a range of different scales instead of $0.1$. However, this approach suffers from the [multiple comparisons problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem).
"""

# ╔═╡ d4443968-63d5-479f-8c38-6dfa7cc6f2f5
md"""
`MonteCarloTesting` provides a way to properly account for multiple trials in p-value estimation. \
In this example, there is one free parameter: the clustering scale. Let's take a range of its values, and compute the number of pairs for each:
"""

# ╔═╡ 3a134034-8e10-416c-ac99-1667f707fa63
md"""
The result of such an operation is a regular keyed array from `AxisKeys`. All array and keyed array operations work on it:
"""

# ╔═╡ c4275684-b330-462c-b68e-580ca9f06e32
md"""
First, ensure that we get the same p-value as above for `scale=0.1`. \
The result should be exactly the same, as all randomness happens in the original `mc_vec` construction, not in any further computations:
"""

# ╔═╡ 9152fa0e-399a-4a4a-b632-2919bb15e279
md"""
Compute p-values for each `n` by broadcasting:
"""

# ╔═╡ f36b2b62-79a4-43fd-8446-badccdcd5dd6
md"""
Find the minimum of these p-values, and the scale it is attained at:
"""

# ╔═╡ a0a0d4a6-c77e-4a13-ad3c-75c8c775ad8d
md"""
This minimum is smaller than $p$ we obtained above for `scale = 0.1`, indicating that a different scale may be better in detecting clustering in this dataset.

However, because of the multiple comparisons issue, the minimum is not the probability! \
We need to compute the "post-trial" p-value: the actual probability to obtain the minimum "pre-trial" p-value as low as it is observed:
"""

# ╔═╡ 2d788639-d5b6-4306-9b8c-34a4c09d13b8
md"""
All p-values we obtained so far are demonstrated in this plot:
"""

# ╔═╡ d1835c86-350d-491f-96ff-126216fae9f8
md"""
Here, the minimum reaches $10^{-4} = 1/N_\mathrm{random}$, indicating that more random realization could better resolve those p-values. See the `nrandom` argument to `montecarlo(...)`.
"""

# ╔═╡ 1ac62ca5-bb62-4da9-98d3-14689d96ce04
md"""
Further, instead of using the minimum p-value, another combination can be useful in post-trial computations. A common choice when many pre-trial p-values are expected to be similarly significant (not the case in our example!) is the harmonic mean:
"""

# ╔═╡ f699be96-8192-49ac-a425-301607c4934a
md"""
Note that the combination function needs to be decided beforehand, not after looking at computation results.
"""

# ╔═╡ 164222d0-2676-49df-ab70-8e86ea912c6d
md"""
### P-value intervals
"""

# ╔═╡ b7e4b2ff-45ec-413f-b3b5-e00084dd40a8
md"""
#### Uncertainty due to randomness
"""

# ╔═╡ 6663f2d0-d026-4437-96e8-48e71f02d620
md"""
`MonteCarloTesting` also provides basic tools to evaluate the accuracy of obtained p-values. Uncertainties arise from the inherent randomness of simulations.

Use `pvalue_mcinterval()` instead of `pvalue()` to compute the binomial uncertainty interval, at the 95% level by default:
"""

# ╔═╡ c56e4af7-d44d-42fe-922f-6d07d1fa9296
md"""
With p-value this small, comparable with $1/N_\mathrm{random}$, the relative uncertainty is quite large. It can be improved with more random realizations (`nrandom` argument).
"""

# ╔═╡ a77b95d3-3d7e-44d5-bed4-1e95b0bb0e85
md"""
This plot illustrates confidence intervals for the whole scan:
"""

# ╔═╡ 1350a0b4-c683-4987-8239-1691a464e899
md"""
#### Ties in the test statistic values
"""

# ╔═╡ bb9f7d85-6922-45b7-abc3-d8efc49e60f9
md"""
When the test statistic is discrete, it can only assume a finite number of values.\
This is the case in our example: we count the number of pairs. It can be informative to understand, how much this discreteness affects p-value estimates.

The `pvalue_tiesinterval` function calculates the interval where discreteness doesn't allow resolving finer details:
"""

# ╔═╡ 7863463b-7045-4731-b570-6db000ee3149
md"""
Visualization for the whole scan:
"""

# ╔═╡ 9c1f440c-e8eb-44a4-9b60-15504cd7ec31
md"""
Here, the ties interval is narrow in our region of interest. If that wasn't the case, thinking of a different test statistic could be useful.
"""

# ╔═╡ 7fbd73a1-95a2-42f5-bf9c-6d4eec5a9119
md"""
## Evaluating p-value correctness and more
"""

# ╔═╡ 48c90289-6c64-4906-bc3d-c7d0c9411c34
md"""
This section isn't documented. Here, we evaluate how pre/post-trial p-values behave under different dependence structures between trials for different parameter values.
"""

# ╔═╡ 6f2f83f1-d40f-4e90-9c78-110bffa4d17c
pvals = map(1:10^5) do _
	mc = montecarlo(real=rand(), randomfunc=rng -> rand(rng), nrandom=100)
	pvalue(mc; alt= >=)
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

# ╔═╡ 8ee73b57-db1f-4c94-89aa-544f1830c531


# ╔═╡ 38f499c8-bac1-4583-a414-03ab0f2f71c9
import ThreadsX

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
end;

# ╔═╡ 58be6aa4-43b0-4dd0-bc9d-fd6ff927e11d
pvals_trials |> AsText

# ╔═╡ b0ccd5f9-d4cd-4dd7-953d-328abca78ad0
map([:pretrial, :harmmean, :posttrial_minimum, :posttrial_harmmean, :posttrial_prod]) do k
	k => map(pvals_trials |> rowtable) do r
		plot_pvals(getproperty.(r.value, k))
		plt.title("dep: $(r.dep)")
		plt.gcf()
	end
end

# ╔═╡ f18b9502-e6ef-4530-9570-7fea1f1a1572
TableOfContents()

# ╔═╡ e31adc26-70af-4b78-a2af-134915e55f71
import Random

# ╔═╡ f0318549-27ee-4f8e-9f21-cd667e69745f
mc_vec = let
	Random.seed!(123)
	montecarlo(;
		real=clamp.([rand(Normal(0.1, 0.05), 20); rand(Normal(0.5, 0.05), 20); rand(Uniform(0, 1), 60)], Ref(0..1)),
		randomfunc=rng -> rand(rng, 100),
		nrandom=10^4-1,
	)
end

# ╔═╡ ee5e67c8-8394-4b85-b655-092cb94779aa
let
	plt.figure(figsize=(6, 2))
	let v = realval(mc_vec)
		plt.scatter(v, fill(0, length(v)), marker="|", color=:k)
	end
	for i in 1:5
		v = randomvals(mc_vec)[i]
		plt.scatter(v, fill(i, length(v)), marker="|")
	end
	plt.yticks(
		0:5,
		[["Real"]; ["Rand #$i" for i in 1:5]]
	)
	plt.gca().invert_yaxis()
	plt.xlim(0, 1)
	plt.gcf()
end

# ╔═╡ 59fdf6d8-7eea-45bf-b787-665ef17147ef
mc_close01 = mapsamples(mc_vec) do xs
	mean(
		((x, y),) -> abs(x - y) < 0.1,
		Iterators.product(xs, xs)
	)
end

# ╔═╡ 61d04ccc-36a6-431e-9c09-06be4e07fa33
pvalue(mc_close01, alt= >=)

# ╔═╡ d9da774b-e3c9-4688-b9c4-15d80745f4d1
mcint = pvalue_mcinterval(mc_close01, alt= >=)

# ╔═╡ 742944d6-6fd1-405b-a385-72f301c285bd
maximum(mcint) / minimum(mcint)

# ╔═╡ 73d237f6-fbc0-4387-b83c-50b6dd81b111
mean(mcint), 1/nrandom(mc_close01)

# ╔═╡ 9435477e-08cd-47fe-88f2-4b74fb75305f
pvalue_tiesinterval(mc_close01, alt= >=)

# ╔═╡ d00aab47-550c-4a1a-ae81-a5980a53368b
mc_close = map_w_params(mc_vec, grid(scale=0:0.01:1); mapfunc=ThreadsX.map) do xs, p
	mean(
		((x, y),) -> abs(x - y) < p.scale,
		Iterators.product(xs, xs)
	)
end;

# ╔═╡ d9772d78-2a93-4818-9af9-018573f99a62
mc_close |> AsText

# ╔═╡ 38bb492f-aaad-4758-bac5-f1dbea55ba48
named_axiskeys(mc_close)

# ╔═╡ fdbad101-3f31-47f5-adb2-4e55bf192c54
length(mc_close)

# ╔═╡ 8b12a11c-5c41-461b-a286-79ad46313188
mc_close(scale=0.1)

# ╔═╡ 2e5466fe-17cf-47d9-a30b-021d7b2c9894
pvalue(mc_close(scale=0.1), alt = >=)

# ╔═╡ 368d1664-71f3-4070-bae0-8351243b9b14
pvals_close = pvalue.(mc_close, alt= >=);

# ╔═╡ 5b73562f-9576-4fe3-bafb-ca98771feac7
pvals_close |> AsText

# ╔═╡ fdbcda0f-a281-4bb0-a1a7-63295ac3af2c
with_axiskeys(findmin)(pvals_close)

# ╔═╡ 3b610570-e0d8-49f5-be69-5f8ae1029a94
pvalue_post(mc_close; alt= >=)

# ╔═╡ 3dd0aa2b-d1d1-454a-a5f7-f5d0d11c96ae
let
	plt.figure()
	plot_ax(pvals_close; label="Parameter scan\n(pre-trial)")
	
	plt.scatter(0.1, pvalue(mc_close01, alt= >=); color=:k)
	plt.text(0.1, pvalue(mc_close01, alt= >=), "  scale = 0.1, no scan", va=:center)

	let (p, k) = with_axiskeys(findmin)(pvals_close)
		plt.scatter(k.scale, p; color=:C0)
		plt.text(k.scale, p, "   minimum in scan", va=:center, color=:C0)
	end

	# plt.scatter(1, pvalue_post(mc_close; alt= >=), marker="_", color=:C1, s=0.3e3, clip_on=false, label="Parameter scan\n(post-trial)")
	plt.plot([0.95, 1.01], fill(pvalue_post(mc_close; alt= >=), 2), color=:C1, clip_on=false, label=" Parameter scan\n (post-trial)")
	
	plt.yscale(:log)
	plt.xlim(0, 1)
	xylabels("Clustering scale", "p-value")

	legend_inline_right()
	
	plt.gcf()
end

# ╔═╡ 2bd0b39b-8ce4-4661-adac-059e31153503
pvalue_post(mc_close; alt= >=, combine=harmmean)

# ╔═╡ 6d12367e-b320-4b23-9608-4b1b6c41c968
let
	plt.figure()
	plot_ax(pvals_close; label="Parameter scan\n(pre-trial)")
	fill_between_ax(pvalue_mcinterval.(mc_close, alt= >=); alpha=0.3, edgecolor=:C0, lw=1)

	let scale = 0.1
		p = pvalue_mcinterval(mc_close01, alt = >=)
		errorbar(0.1, p; color=:k, fmt=".")
		plt.text(0.1, mean(p), "  scale = 0.1, no scan", va=:center)
	end
	
	plt.yscale(:log)
	plt.xlim(0, 1)
	xylabels("Clustering scale", "p-value")

	legend_inline_right()
	
	plt.gcf()
end

# ╔═╡ a50b5689-d186-4adb-b7c4-390270fb40ee
let
	plt.figure()
	plot_ax(pvals_close; label="Parameter scan\n(pre-trial)")
	fill_between_ax(pvalue_tiesinterval.(mc_close, alt= >=); alpha=0.3, edgecolor=:C0, lw=1)

	let scale = 0.1
		p = pvalue_tiesinterval(mc_close01, alt = >=)
		errorbar(0.1, p; color=:k, fmt=".")
		plt.text(0.1, mean(p), "  scale = 0.1, no scan", va=:center)
	end
	
	plt.yscale(:log)
	plt.xlim(0, 1)
	xylabels("Clustering scale", "p-value")

	legend_inline_right()
	
	plt.gcf()
end

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
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RectiGrids = "8ac6971d-971d-971d-971d-971d5ab1a71a"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
ThreadsX = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"

[compat]
AxisKeysExtra = "~0.1.4"
DataPipes = "~0.3.5"
DisplayAs = "~0.1.6"
Distributions = "~0.25.80"
MonteCarloTesting = "~0.1.16"
PlutoUI = "~0.7.50"
PyPlotUtils = "~0.1.26"
RectiGrids = "~0.1.17"
StatsBase = "~0.33.21"
Tables = "~1.10.0"
ThreadsX = "~0.1.11"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0-beta4"
manifest_format = "2.0"
project_hash = "6b37c5fef002b7ed81b69c0546e14db4ce529279"

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
git-tree-sha1 = "b9661b900b50ba475145b311a9a0ef9d2a9c85ea"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.26"
weakdeps = ["StaticArrays"]

    [deps.Accessors.extensions]
    StaticArraysExt = "StaticArrays"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0310e08cb19f5da31d08341c6120c047598f5b9c"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.5.0"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisKeys]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "IntervalSets", "InvertedIndices", "LazyStack", "LinearAlgebra", "NamedDims", "OffsetArrays", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "f1f6c24c1be95d4baa0880903641fa4a15e06d9c"
uuid = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
version = "0.2.12"

[[deps.AxisKeysExtra]]
deps = ["AxisKeys", "Compat", "RectiGrids", "Reexport", "SplitApplyCombine", "StructArrays"]
git-tree-sha1 = "45ee23fc00ac8b5984783d4d5ef680b2a8caf85d"
uuid = "b7a0d2b7-1990-46dc-b5dd-87820ecd1b09"
version = "0.1.4"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "7fe6d92c4f281cf4ca6f2fba0ce7b299742da7ca"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.37"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "02d2316b7ffceff992f3096ae48c7829a8aa0638"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.3"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "e32a90da027ca45d84678b826fffd3110bb3fc90"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.8.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.ConstructionBaseExtras]]
deps = ["ConstructionBase", "IntervalSets", "StaticArraysCore"]
git-tree-sha1 = "bb56454b5793791c8d313f5220b1da9db8399274"
uuid = "914cd950-b775-4282-9f32-54fc4544c321"
version = "0.1.1"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "3c8de95b4e932d76ec8960e12d681eba580e9674"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.8"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataPipes]]
git-tree-sha1 = "44a632423521bd0d189ed49a7ecb521bf10e1758"
uuid = "02685ad9-2d12-40c3-9f73-c6aeda6a7ff5"
version = "0.3.5"

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

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "e82c3c97b5b4ec111f3c1b55228cebc7510525a2"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.25"

[[deps.DirectionalStatistics]]
deps = ["Accessors", "IntervalSets", "InverseFunctions", "LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "776bcd884b034903e6090485fc918019838b9ef0"
uuid = "e814f24e-44b0-11e9-2fd5-aba2b6113d95"
version = "0.1.22"

[[deps.DisplayAs]]
git-tree-sha1 = "43c017d5dd3a48d56486055973f443f8a39bb6d9"
uuid = "0b91fe84-8a4c-11e9-3e1d-67c38462b6d6"
version = "0.1.6"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "74911ad88921455c6afcad1eefa12bd7b1724631"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.80"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "aa0f95312367be88ec8459994ae31a6e308eee2d"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.6.4"

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
git-tree-sha1 = "d3ba08ab64bdfd27234d3f61956c966266757fe6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.7"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.FlexiMaps]]
deps = ["Accessors", "InverseFunctions"]
git-tree-sha1 = "a72e3a3c87f62724e701ffb46a3724df8326fb47"
uuid = "6394faf6-06db-4fa8-b750-35ccc60383f7"
version = "0.1.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "1cd7f0af1aa58abc02ea1d872953a97359cb87fa"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.4"

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

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "16c0cc91853084cb5f58a78bd209513900206ce6"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.4"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

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
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "680e733c3a0a9cea9e935c8c2184aea6a63fa0b5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.21"

    [deps.LogExpFunctions.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    ChangesOfVariablesExt = "ChangesOfVariables"
    InverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "4d5917a26ca33c66c8e5ca3247bd163624d35493"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MonteCarloTesting]]
deps = ["Accessors", "ConstructionBaseExtras", "FlexiMaps", "InteractiveUtils", "IntervalSets", "Markdown", "Printf", "Random", "StatsBase"]
git-tree-sha1 = "41004feffd735e6f0a0f13614d398858aced70b9"
uuid = "b75b2f39-c526-438a-aeb3-f18deacfdc57"
version = "0.1.16"
weakdeps = ["AxisKeys", "Distributions", "RectiGrids"]

    [deps.MonteCarloTesting.extensions]
    DistributionsExt = "Distributions"
    RectiGridsExt = ["AxisKeys", "RectiGrids"]

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "dc9144f80a79b302b48c282ad29b1dc2f10a9792"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "1.2.1"

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
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+0"

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

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6f4fbcd1ad45905a5dee3f4256fabb49aa2110c6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.7"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

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
git-tree-sha1 = "62f417f6ad727987c755549e9cd88c46578da562"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.95.1"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "f9d953684d4d21e947cb6d642db18853d43cb027"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.0"

[[deps.PyPlotUtils]]
deps = ["Accessors", "ColorTypes", "DataPipes", "DirectionalStatistics", "DomainSets", "FlexiMaps", "IntervalSets", "LinearAlgebra", "NonNegLeastSquares", "PyCall", "PyPlot", "Statistics", "StatsBase"]
git-tree-sha1 = "0db936b203ca7f348b54c444b2503363b94e4e2f"
uuid = "5384e752-6c47-47b3-86ac-9d091b110b31"
version = "0.1.26"

    [deps.PyPlotUtils.extensions]
    AxisKeysExt = "AxisKeys"
    AxisKeysUnitfulExt = ["AxisKeys", "Unitful"]
    UnitfulExt = "Unitful"

    [deps.PyPlotUtils.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "786efa36b7eff813723c4849c90456609cf06661"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RectiGrids]]
deps = ["AxisKeys", "ConstructionBase", "ConstructionBaseExtras", "Random"]
git-tree-sha1 = "ee0f3d50454f3fede3114aa16f6af6c1edd39ad1"
uuid = "8ac6971d-971d-971d-971d-971d5ab1a71a"
version = "0.1.17"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
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

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "67d3e75e8af8089ea34ce96974d5468d4a008ca6"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.15"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

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
git-tree-sha1 = "ab6083f09b3e617e34a956b43e9d51b824206932"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.1.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "b03a3b745aa49b566f128977a7dd1be8711c5e71"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.14"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadsX]]
deps = ["ArgCheck", "BangBang", "ConstructionBase", "InitialValues", "MicroCollections", "Referenceables", "Setfield", "SplittablesBase", "Transducers"]
git-tree-sha1 = "34e6bcf36b9ed5d56489600cf9f3c16843fa2aa2"
uuid = "ac1d9e8a-700a-412c-b207-f0111f4b6c0d"
version = "0.1.11"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c42fa452a60f022e9e087823b47e5a5f8adc53d5"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.75"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.4.0+0"

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
# ╠═b05dbced-8909-4e44-aee9-3665c38bad11
# ╟─ead1ece0-91a0-4f3a-983c-2e0763918655
# ╠═468b8932-1ff6-4810-9ad2-94b841d7afd9
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
# ╟─0bd6accd-415d-4097-862c-89ebcfaf3ccd
# ╟─ee5e67c8-8394-4b85-b655-092cb94779aa
# ╟─4d5ad9d1-2b12-4cd3-a4c1-c304017b089e
# ╟─da95f93f-fac2-4afa-8337-bdfddcfa150b
# ╠═59fdf6d8-7eea-45bf-b787-665ef17147ef
# ╠═61d04ccc-36a6-431e-9c09-06be4e07fa33
# ╟─71e2becb-e482-485a-992c-975c8744ea97
# ╟─fce1d1b3-1fe8-43ae-8a35-136d9b8fca5a
# ╟─452512e6-f7ec-423f-924e-f924d9fa0778
# ╟─d4443968-63d5-479f-8c38-6dfa7cc6f2f5
# ╟─d9772d78-2a93-4818-9af9-018573f99a62
# ╠═d00aab47-550c-4a1a-ae81-a5980a53368b
# ╟─3a134034-8e10-416c-ac99-1667f707fa63
# ╠═38bb492f-aaad-4758-bac5-f1dbea55ba48
# ╠═fdbad101-3f31-47f5-adb2-4e55bf192c54
# ╠═8b12a11c-5c41-461b-a286-79ad46313188
# ╟─c4275684-b330-462c-b68e-580ca9f06e32
# ╠═2e5466fe-17cf-47d9-a30b-021d7b2c9894
# ╟─9152fa0e-399a-4a4a-b632-2919bb15e279
# ╠═5b73562f-9576-4fe3-bafb-ca98771feac7
# ╠═368d1664-71f3-4070-bae0-8351243b9b14
# ╟─f36b2b62-79a4-43fd-8446-badccdcd5dd6
# ╠═fdbcda0f-a281-4bb0-a1a7-63295ac3af2c
# ╟─a0a0d4a6-c77e-4a13-ad3c-75c8c775ad8d
# ╠═3b610570-e0d8-49f5-be69-5f8ae1029a94
# ╟─2d788639-d5b6-4306-9b8c-34a4c09d13b8
# ╟─3dd0aa2b-d1d1-454a-a5f7-f5d0d11c96ae
# ╟─d1835c86-350d-491f-96ff-126216fae9f8
# ╟─1ac62ca5-bb62-4da9-98d3-14689d96ce04
# ╠═2bd0b39b-8ce4-4661-adac-059e31153503
# ╟─f699be96-8192-49ac-a425-301607c4934a
# ╟─164222d0-2676-49df-ab70-8e86ea912c6d
# ╟─b7e4b2ff-45ec-413f-b3b5-e00084dd40a8
# ╟─6663f2d0-d026-4437-96e8-48e71f02d620
# ╠═d9da774b-e3c9-4688-b9c4-15d80745f4d1
# ╟─c56e4af7-d44d-42fe-922f-6d07d1fa9296
# ╠═742944d6-6fd1-405b-a385-72f301c285bd
# ╠═73d237f6-fbc0-4387-b83c-50b6dd81b111
# ╟─a77b95d3-3d7e-44d5-bed4-1e95b0bb0e85
# ╟─6d12367e-b320-4b23-9608-4b1b6c41c968
# ╟─1350a0b4-c683-4987-8239-1691a464e899
# ╟─bb9f7d85-6922-45b7-abc3-d8efc49e60f9
# ╠═9435477e-08cd-47fe-88f2-4b74fb75305f
# ╟─7863463b-7045-4731-b570-6db000ee3149
# ╟─a50b5689-d186-4adb-b7c4-390270fb40ee
# ╟─9c1f440c-e8eb-44a4-9b60-15504cd7ec31
# ╟─7fbd73a1-95a2-42f5-bf9c-6d4eec5a9119
# ╟─48c90289-6c64-4906-bc3d-c7d0c9411c34
# ╠═6f2f83f1-d40f-4e90-9c78-110bffa4d17c
# ╟─58be6aa4-43b0-4dd0-bc9d-fd6ff927e11d
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
# ╠═a39e055d-7c16-4ada-9ac1-8981be6b2c5c
# ╠═e31adc26-70af-4b78-a2af-134915e55f71
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
