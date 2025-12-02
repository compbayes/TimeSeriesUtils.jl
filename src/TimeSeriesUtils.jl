module TimeSeriesUtils

using Plots, Distributions, FFTW, Polynomials, LinearAlgebra
using LaTeXStrings
using RCall

# Installing some R package, if not already installed.
R"if (suppressMessages(!require('forecast'))) 
install.packages('forecast', repo = 'https://cloud.r-project.org')"

R"if (suppressMessages(!require('psd'))) 
install.packages('psd', repo = 'https://cloud.r-project.org')"

include("GeneralUtils.jl")
export ts, stl, mstl, nainterpret!, nainterpret

include("SpectralUtils.jl")
export spectrum, pspectrum, pgram, pGramDistribution
export ℓwhittle
export simProcessSpectral

include("ARIMAUtils.jl")
export ARMAacf, arma_reparam, inv_arma_reparam, arma_reparam_partials, sarma_reparam
export check_stationarity, sim_uniformAR
export Arima, simARMA, ℓARMA
export SpecDensARMA, SpecDensSARMA, SpecDensMultiSARMA, SpecDensARTFIMA

end
