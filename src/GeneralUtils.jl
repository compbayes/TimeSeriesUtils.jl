"""
    ts(data; frequency = 1, deltat = 1)

Creates a time series object in R with a certain `frequency` and observed `deltat` time steps a part

"""
function ts(data; frequency = 1, deltat = 1)
    R"""
        suppressMessages(library(forecast))
        tsObject = ts($data, frequency = 1, deltat = 1)
    """
    @rget tsObject
    return tsObject
end
   
"""
    stl(x; frequency = 1, swindow="periodic", twindow = nothing, robust=true)

R function that decompose a time series into seasonal, trend and irregular components using loess.

Examples:
```jldoctest
julia> T = 10*12; 
julia> x = 0.01*(1:T) .+ sin.((1:T)*(2pi/12)) .+ 0.2*randn(T)
julia> stl_object = stl(x, frequency = 12)
julia> stl_object
julia> plot(x, label ="time series", c = "black", legend = :topright)
julia> plot!(stl_object[:time_series][:,1], label = "seasonal", c = "red")
julia> plot!(stl_object[:time_series][:,2], label = "trend", c = "blue")
julia> plot!(stl_object[:time_series][:,3], label = "remainder", c = "green")
```

"""
function stl(x; frequency = 1, swindow="periodic", twindow = nothing, robust=true)
    # Inputs: time series vector x
    R"""
        suppressMessages(library(forecast))
        stlObject = stl(ts($x, frequency = $frequency), s.window=$swindow, 
            t.window=$twindow, robust=$robust)
    """
    @rget stlObject
    return stlObject
end

"""
    mstl(x; seasonal_periods, swindow = 7 .+ 4 .*(1:6))

R function that decompose a time series into seasonal, trend and irregular components using loess. Decompose a time series into seasonal, trend and remainder components. Allows for multiple seasonal components with different periods.

Examples:
```jldoctest
julia> T = 10*12; 
julia> x = 0.01*(1:T) .+ sin.((1:T)*(2pi/12)) .+ sin.((1:T)*(2pi/365.25)) .+ 0.2*randn(T)
julia> mstl_object = mstl(x; seasonal_periods = [12,365.25])
julia> mstl_object
julia> plot(x, label ="time series", c = "black", legend = :topright)
julia> plot!(mstl_object[:time_series][:,1], label = "seasonal", c = "red")
julia> plot!(mstl_object[:time_series][:,2], label = "trend", c = "blue")
julia> plot!(mstl_object[:time_series][:,3], label = "remainder", c = "green")
```

"""
function mstl(x; seasonal_periods, swindow = 7 .+ 4 .*(1:6), robust=true)
    # Inputs: time series vector x
    R"""
        suppressMessages(library(forecast))
        mstlObject = mstl(msts($x, seasonal.periods = $seasonal_periods), s.window = $swindow, robust = $robust)
        mstl_df = data.frame(mstlObject)
    """
    @rget mstl_df
    return mstl_df
end

"""
    nainterpret!(x; frequency = 1)

R function that interpolates missing values in a time series inplace.

See also [`nainterpret`](@ref) for an not in place version.

Examples:
```jldoctest
julia> nainterpret!([4, missing, 10])
3-element Vector{Float64}:
  4.0
  7.0
 10.0
```
"""
function nainterpret!(x; frequency = 1)
    R"""
        suppressMessages(library(forecast))
        x = na.interp(ts($x, frequency = $frequency))
    """
    @rget x
    return x
end

"""
    nainterpret(x; frequency = 1)

R function that interpolates missing values in a time series.

See also [`nainterpret!`](@ref) for an inplace version.

Examples:
```jldoctest
julia> y = nainterpret([4, missing, 10])
3-element Vector{Float64}:
  4.0
  7.0
 10.0
```
"""
function nainterpret(x; frequency = 1)
    R"""
        suppressMessages(library(forecast))
        y = na.interp(ts($x, frequency = $frequency))
    """
    @rget y
    return y
end

