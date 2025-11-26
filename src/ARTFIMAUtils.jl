""" 
    artfima(z; model, arimaOrder = [0,0,0], likAlg, fixd = nothing, lambdaMax = 3, dMax = 10) 

R function that ML estimates an ARTFIMA model for the time series `z`. 

""" 
function artfima(z; model, arimaOrder = [0,0,0], likAlg, fixd = nothing, lambdaMax = 3, dMax = 10)
    R"""
        suppressMessages(library(artfima))
        fittedModel = artfima($z, glp = $model, arimaOrder = $arimaOrder, likAlg = $likAlg, fixd = $fixd, b0 = NULL, lambdaMax = $lambdaMax, dMax = $dMax)
    """
    @rget fittedModel # Grabs fittedModel object from R and makes it a Julia dict
    return fittedModel
end


""" 
    artsim(n, d, λ, ϕ, θ, μ, σ) 

R function that simulates time series ´x´ from an ARTFIMA model. 

# Examples

Simulating n = 3 observations from ARFIMA(1, d=0.5, 0) with ϕ = 0.8 and σ = 1
```julia-repl
julia> artsim(3, 0.5, 0, 0.8, 0, 0, 1)
3-element Vector{Float64}:
 -28.22058711106737
 -27.70285365880599
 -26.99852776495729
```
"""
function artsim(n, d, λ, ϕ, θ, μ, σ) # No unicode in R-Windows 🤦‍♂️
    lambda = λ; phi = ϕ; theta = θ; mu = μ; sigma = σ; 
    R"""
        suppressMessages(library(artfima))
        x = artsim($n, $d, $lambda, $phi, $theta, $mu, $sigma)
    """
    @rget x
	return x
end



""" 
    artfima_pred(x, h; likAlg = "Whittle")

R function that predicts time series ´x´ using an ARTFIMA model. 

# Examples

Simulating n = 3 observations from ARFIMA(1, d=0.5, 0) with ϕ = 0.8 and σ = 1
```julia-repl
julia> x = artsim(100, 0.5, 0, 0.8, 0, 0, 1);
julia> preds = artfima_pred(x, 12; likAlg = "Whittle")
```
"""
function artfima_pred(x, h; likAlg = "Whittle")
    R"""
        suppressMessages(library(artfima))
        artfima.model <- artfima(ts($x), likAlg=$likAlg)
        preds = predict(artfima.model, n.ahead=$h)$Forecasts
    """
    @rget preds
	return preds[:]
end


""" 
    SpecDensARTFIMA(ω, ϕ, θ, d, λ, σ²) 

Compute spectral density for the univariate ARTFIMA model over domain ω ∈ [-π,π]. 

- ω is a radial frequency
- ϕ is a vector of AR coefficients
- θ is a vector of MA coefficients
- d is the fractional differenting parameter
- λ ≥ 0 is the tempering parameter 
- σ² is the noise variance

# Examples
The spectral density for an AR(1) process with unit noise variance is
```doctests 
julia> SpecDensARTFIMA(0.5, 0.9, 0, 0, 0, 1)
0.6909224383713601
```
""" 
function SpecDensARTFIMA(ω, ϕ, θ, d, λ, σ²)
    ARpoly =  Polynomial([1;-ϕ], :z)
    MApoly =  Polynomial([1;θ], :z) 
    specDens = (σ²/(2π))*(abs(MApoly(exp(-im*ω)))^2/abs(ARpoly(exp(-im*ω)))^2)*
		abs(1-exp(-(λ+im*ω)))^(-2*d)
	return specDens
end 



