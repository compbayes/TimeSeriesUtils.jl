
"""
    spectrum(X; method = "pgram", plot = false)

R function estimates a spectral density using smoothed periodogram or fitted AR and plots it in R if `plot` is `true`.

> The spectrum is defined with scaling 1/frequency(x). This makes the spectral density a density over the range (-frequency(x)/2, +frequency(x)/2] [R documentation]

`method` can be `"pgram"` or `"ar"``.
returns a dict with keys including
- :freq (vector with linear frequencies between 0 and π)
- :spec (vector with the spectral density estimate)

Examples:
```@jldoctest
julia> X = rand(Normal(),100);
julia> spectrum(X; method = "pgram")
OrderedCollections.OrderedDict{Symbol, Any} with 16 entries:
  :freq      => [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1  …  0.41, 0.42, 0.43, 0.…
  :spec      => [0.0162793, 0.391037, 0.0750275, 1.10354, 0.170123, 0.379539, 0.640757, 3.54369, 0.…
  :coh       => nothing
  :phase     => nothing
  :kernel    => nothing
  :df        => 1.79159
  :bandwidth => 0.00288675
  :n_used    => 100
  ⋮          => ⋮
```
"""
function spectrum(X; method = "pgram", plot = false)
    # method can be "pgram" or "ar"
    R"""
        specdens = spectrum($X, method = $method, plot = $plot)
        specdens$spec = specdens$spec*(1/(pi))
    """
    @rget specdens
    return specdens
end

"""
    pspectrum(X; sampfreq = 1, niter = 3, plotflag = false)

R function multi-taper estimate of the spectral density.

Examples:
```@jldoctest
julia> X = randn(100);
julia> pspectrum(X)
⋮
OrderedCollections.OrderedDict{Symbol, Any} with 22 entries:
  :freq      => [0.0, 0.0102041, 0.0204082, 0.0306122, 0.0408163, 0.0510204, 0.0612245, 0.0714286, 0.0816327, 0.0918367  …  0.408163, 0.418367, 0.428571, 0.438776, 0.44898, 0.459184, 0.469388, 0.479592, 0.489796, 0.5]
  :spec      => [0.175972, 0.175629, 0.175015, 0.173849, 0.17223, 0.171751, 0.171284, 0.170648, 0.172062, 0.173692  …  0.235946, 0.235948, 0.236749, 0.238585, 0.239633, 0.240635, 0.241796, 0.242778, 0.244478, 0.245774]
  :pspec     => nothing
  :transfer  => nothing
  :coh       => nothing
  :phase     => nothing
  :kernel    => nothing
  :df        => 100.0
  :numfreq   => 50
  :bandwidth => 1.02
  ⋮          => ⋮
```
"""
function pspectrum(X; sampfreq = 1, niter = 3, plotflag = false)
    R"""
        suppressMessages(library(psd))
        specdens = pspectrum($X, x.frqsamp = $sampfreq, niter = $niter, plot = $plotflag)
        specdens$spec = specdens$spec*(1/(2*pi)) # Normalizing so integral from -pi to pi is the variance.
    """
    @rget specdens
    return specdens
end

"""
    pGram, ωFourier = pgram(X)

Computes the periodogram `pGram` at the `ωFourier` frequencies for each of the columns in `X`.

If X is vector, the returned periodogram `pGram` is a vector with length(ωFourier) elements.

If X is a matrix with r>1 columns, the returned periodogram `pGram` is a 3d array of dimensions (r, r, length(ωFourier)).

Examples:
```julia-repl
julia> pGram, ωFourier = pgram([1,2,5,8,5])
([14.03746598070517, 2.399901258445937, 0.05108486516925192], [0.0, 1.2566370614359172, 2.5132741228718345])
julia> pGram, ωFourier = pgram([1 3;2 5;5 7;8 4;5 5]); # two column matrix
julia> pGram
2×2×3 Array{Complex, 3}:
[:, :, 1] =
 14.0375+0.0im  16.0428+0.0im
 16.0428+0.0im  18.3346+0.0im

[:, :, 2] =
   2.3999-2.22045e-16im  0.398436-0.825562im
 0.398436+0.825562im     0.350141+0.0im

[:, :, 3] =
  0.0510849+3.46945e-18im  -0.0642101+0.11732im
 -0.0642101-0.11732im        0.350141+0.0im
```
"""
function pgram(X::Matrix)
    T = size(X,1)
    r = size(X,2)
    ωFourier = 2π*rfftfreq(T)
    DFT = mapslices(rfft, X, dims = 1)
    pGram = zeros(Complex, r, r, length(ωFourier)) # array of dims (p,p,(꜖T/2˩ + 1))
    for (j,ω) ∈ enumerate(ωFourier)
        pGram[:,:,j] = (1/(2π*T))*DFT[j,:]*DFT[j,:]' # FIXME: Diagonal is here only approx real. Make it real?
    end
    return pGram, ωFourier
end

function pgram(X::Vector)
    T = length(X)
    ωFourier = 2π*rfftfreq(T)
    pGram = (1/(2π*T))*abs.(rfft(X)).^2 # returns vector of length ꜖T/2˩ + 1
    return pGram, ωFourier
end

"""
    pGramDistribution(pGram, ωGrid, SpecDens, plotFig)

Plots the periodogram `pGram` over the vector of frequencies `ωGrid` and the exponential distribution implied by the spectral density model `SpecDens`.

# Examples
Plot the periodogram and spectral density for the AR(1) model:
```julia-repl
julia> pGram, ωFourier = pgram(simARMA([0.8], [0.0], 0.0, 1.0, 100));
julia> SpecDensDistr = pGramDistribution(pGram[2:end],
    ωFourier[2:end], ω -> SpecDensARMA.(ω, 0.8, 0.0, 1), true);
julia> SpecDensDistr[1] # The implied exponential distribution at ωFourier[2]
Exponential{Float64}(θ=3.6877929014993596)
```
"""
function pGramDistribution(pGram, ωGrid, SpecDens, plotFig)
    SpecDensDistr = Exponential.(SpecDens.(ωGrid))
    if plotFig
        fracAboveUpper = mean(pGram .> quantile.(SpecDensDistr, 0.975))
        fracBelowLower = mean(pGram .< quantile.(SpecDensDistr, 0.025))
        p = scatter(ωGrid, pGram, color = colors[1], markersize = 2, markerstrokecolor = :auto, label = "periodogram", xlabel = L"\omega", ylabel = L"f(\omega)")
        plot!(p, ωGrid, quantile.(SpecDensDistr, 0.025), color = colors[3], lw = 2,
            label = "95% bands")
        plot!(p, ωGrid, quantile.(SpecDensDistr, 0.500), color = colors[4], lw = 2,
            label = median)
        plot!(p, ωGrid, quantile.(SpecDensDistr, 0.975), color = colors[3], lw = 2,
            label = nothing)
        annotate!(2, ylims(p)[2]/2, text("Fraction obs <  2.5 % band is $(fracBelowLower) \nFraction obs > 97.5 % band is $(fracAboveUpper)", :black, :left, 6))
        display(p)
    end
    return SpecDensDistr
end


""" 
    ℓwhittle(pGram::Vector, ω, specDens) 

Compute Whittle approximation to the log-likelihood of a zero mean univariate time series with spectral density `specDens`. The periodogram data `pGram` should contain only the strictly positive frequencies.

The periodogram can be computed with [`pgram`](@ref).

# Examples
Compute the Whittle log-likelihood of the ARIMA model at the true parameters:
```julia-repl
julia> ϕ = 0.8; θ = 0.0; μ = 0; σ = 1; # Set up data generating AR(1) process.
julia> x = artsim(100, 0, 0, ϕ, θ, μ, σ); # simulate time series of length 100
julia> pGram, ωFourier = pgram(x); # ωFourier are the non-negative Fourier frequencies.
julia> ℓwhittle(pGram[2:end], ωFourier[2:end], ω -> SpecDensARMA.(ω, ϕ, θ, σ^2))
-143.3107687445708 # varies with random seed.
```
""" 
function ℓwhittle(pGram::Vector, ω, specDens)
    specDensEval = specDens.(ω)
    T = 2*length(pGram)
    return -T*log(2π)  - sum( log.(specDensEval) + pGram./specDensEval)
end




""" 
    simProcessSpectral(specDens, T, Δₜ) 

Simulates a continuous time process x(t) for t ∈ [0,T] from its (one-sided) spectral density specDens(ω) at time resolution Δₜ.

Uses FFT as described in Section 5.6 of [Lindgren et al 2014](https://doi.org/10.1201/b15922).

# Examples
Simulating from a mixture of Gamma spectral density:
```julia-repl
julia> d=Truncated(MixtureModel(Gamma, [(2, 0.1), (150,0.01)]), 0, π);
julia> ωgrid=0.01:0.01:π; p1=plot(ωgrid, 3*pdf.(d,ωgrid), xlab = "v", ylab = "f(v)")
julia> T = 10; Δₜ = 0.1; x = simProcessSpectral(ω -> 3*pdf.(d,ω), T, Δₜ);
julia> p2=plot(Δₜ:Δₜ:T, x, xlab = "t", ylab = "x(t)")
julia> plot(p1,p2, layout = (2,1), legend = nothing)
```
"""
function simProcessSpectral(specDens, T, Δₜ)
    N = ceil(Int,T/Δₜ) # Number of sample points in the time domain. 
    Δ = (2π)/T # Resolution in frequency domain. 
    ωGrid = 0:Δ:((N-1)*Δ)
    σ² = Δ .* specDens(ωGrid)
    z = .√σ² .* (randn(N) + im*randn(N))
    return real(fft(z))
end

""" 
    simProcessSpectral(specDens, ωSupport, T, Δₜ [, scale = 1])

Simulates a continuous time stochastic process x(t) for t ∈ [0,T] at time resolution `Δₜ` from its (one-sided) discrete spectral density `specDens` over a vector of frequencies `ωSupport`. If `specDens` is a probability density (e.g. from Distributions.jl), then `scale` is the energy (variance) of the process.

# Examples 
Simulating from a discretized mixture of Gamma spectral density:
```julia-repl
julia> specDens = Truncated(MixtureModel(Gamma, [(2, 0.1), (10, 0.1)]), 0, π);
julia> ωSupport = 0.01:0.25:π; T = 100; Δₜ = 1; scale = 2;
julia> p1 = plot(ωSupport, .√(scale*pdf.(specDens, ωSupport)), xlims = [0,π], 
    seriestype = :sticks, xlab = "v", ylab = "f(v)")
julia> x = simProcessSpectral(specDens, ωSupport, T, Δₜ, scale)
julia> p2 = plot(Δₜ:Δₜ:T, x, xlab = "t", ylab = "x(t)")
julia> plot(p1, p2, layout = (2,1), legend = nothing)
```
""" 
function simProcessSpectral(specDens, ωSupport, T, Δₜ, scale = 1)
    tGrid = Δₜ:Δₜ:T
    σs = .√(scale*pdf.(specDens, ωSupport))
    a = σs.*randn(length(ωSupport))
    b = σs.*randn(length(ωSupport))
    x = zeros(length(tGrid))
    for (k,ω) in enumerate(ωSupport)
        x = x + a[k]*cos.(ω*tGrid) + b[k]*sin.(ω*tGrid)
    end
    return x
end