#= MoM-SO

Patel, Utkarsh.
A Surface Admittance Approach For Fast Calculation of the Series
Impedance of Cables Including Skin, Proximity, and Ground Return
Effects. 2014. https://hdl.handle.net/1807/75765

Patel et al.
An Equivalent Surface Current Approach for the Computation of the Series
Impedance of Power Cables with Inclusion of Skin and Proximity Effects.
IEEE TRANSACTIONS ON POWER DELIVERY, VOL. 28, NO. 4, OCTOBER 2013
DOI 10.1109/TPWRD.2013.2267098
=#

using LinearAlgebra
using SpecialFunctions
using HCubature
using Plots

const μ_0 = 4e-7 * pi  # magnetic permibility
const ϵ_0 = 8.8541878128e-12  # electric permitivitty
const sqrt_2 = sqrt(2)
const exp_kelvin = exp(1im * 3/4 * pi)

# derivative of besselj(v,z) in respect to z
dJ_dz(v,z) = (besselj(v-1, z) - besselj(v+1, z)) / 2.0

# Example 1, p. 40, two solid round conductors
a = 10e-3  # radius, m
σ = 58e6  # conductivity, S/m
μ = μ_0
ϵ = ϵ_0
rdc = 1 / (σ * π * a^2)  # DC resistance, Ω/m
ldc = μ_0 / (8π)  # DC inductance, H/m
D_array = [100e-3, 25e-3]  # center-to-center spacing, m
D = D_array[1]  #TODO for D in D_array

nf = 100  # number of frequency points
freq = exp10.(range(0, 6, length=nf))  # frequency range, Hz

# %% Approximation formula 1 and 2
D_2a = D / (2a)
Rf1 = zeros(Float64, nf)  # Analytical (high-freq)
Lf1 = similar(Rf1)
Rf2 = similar(Rf1)  # Analytical (no proximity)
Lf2 = similar(Rf1)
for (i, f) in enumerate(freq)
    w = 2π * f
    δ = 1.0 / sqrt(π * f * μ_0 * σ)  # skin depth
    R_s = 1 / (σ * δ)
    R = R_s * D_2a / (π * a * sqrt((D_2a)^2 - 1))
    L_ext = μ_0 / π * acosh(D_2a)
    Rf1[i] = R
    Lf1[i] = L_ext

    ξ = sqrt_2 * a / δ
    b0 = besselj(0, ξ * exp_kelvin)
    ber = real(b0)
    bei = imag(b0)
    b1 = besselj(1, ξ * exp_kelvin)
    dber_dz = (imag(b1) + real(b1)) / sqrt_2
    dbei_dz = (imag(b1) - real(b1)) / sqrt_2
    zint = (ber + 1im * bei) / (dbei_dz - 1im * dber_dz) / (sqrt_2 * π * a * σ * δ)
    Z = 2 * zint + 1im * w * L_ext
    Rf2[i] = real(Z)
    Lf2[i] = imag(Z) / w
end

# %% MoM-SO
Np = 1  # number - 1 of coefficients of the Fourier expansion in space-domain (basis function)
Ps = 2  # number of solid conductors
Ph = 0  # number of hollow conductors
P = Ps + Ph  # total number of conductors
N = Ps * (2Np + 1) + Ph * 2*(2Np + 1)  # total number of equivalent current coefficients (2.36)

# generate all the RHS matrices of (2.61): Ys, G, U
G = zeros(ComplexF64, N, N)  # Green's matrix
ar = fill(a, P)
xp = [-D/2, D/2]
yp = [0.0, 0.0]
closed_form = false
for p = 1:P  # block p
    for q = 1:P  # block q
        for m = Np:-1:-Np  # line n
            for n = Np:-1:-Np  # line m
                m1 = m + Np + 1 + (2Np + 1) * (q - 1)
                n1 = n + Np + 1 + (2Np + 1) * (p - 1)
                if closed_form
                    xpq = xp[p] - xp[q]
                    ypq = yp[p] - yp[q]
                    dpq = sqrt(xpq^2 + ypq^2)
                    if p != q && n == 0 && m == 0
                        green = log(dpq) / (2π)
                    elseif p != q && n == 0 && m != 0
                        green = -1 / (4π * abs(m)) * (ar[p] / dpq)^abs(m) * (-(xpq - 1im * ypq) / dpq)^m
                    elseif p != q && n > 0 && m >= 1
                        green = 0
                    elseif p != q && n > 0 && m < 1
                        #FIXME what is dx and dy?
                        dx = xpq
                        dy = ypq
                        green = -π * ar[q]^n / (-ar[p])^m * binomial(n-m-1, -m)  * (dx - 1im * dy)^(-n+m) / (4π^2 * n)
                    elseif p != q && n < 0
                        m2 = -m + Np + 1 + (2Np + 1) * (q - 1)
                        n2 = -n + Np + 1 + (2Np + 1) * (p - 1)
                        green = conj(G[m2, n2])
                    elseif p == q && n == 0 && m == 0
                        green = log(ar[p]) / (2π)
                    elseif p == q && n == 0 && m != 0
                        green = 0
                    elseif p == q && n != 0 && m == n
                        green = -1 / (4π * abs(n))
                    elseif p == q && n != 0 && m != n
                        green = 0
                    end
                else
                    #=c1 = (p != q && n > 0 && m >= 1)
                    c2 = (p == q && m != 0)
                    c3 = (p == q && n != 0 && m != n)
                    if (c1 || c2 || c3)
                        green = 0
                    elseif (p != q && n < 0)
                        m2 = -m + Np + 1 + (2Np + 1) * (q - 1)
                        n2 = -n + Np + 1 + (2Np + 1) * (p - 1)
                        green = conj(G[m2, n2])
                    else=#
                        function green_integrand(θ)
                            r = zeros(2,2)
                            for (i, k) in enumerate([p, q])
                                r[1,i] = xp[k] + ar[k] * cos(θ[i])
                                r[2,i] = yp[k] + ar[k] * sin(θ[i])
                            end
                            return log(norm(r[:,1] .- r[:,2])) * exp(1im * (n * θ[2] - m * θ[1])) / 248.05021344239853
                        end
                        green, err = hcubature(green_integrand, [0.0, 0.0], [2π, 2π])
                        err > 1e-3 && @show err
                    #end
                end
                G[m1, n1] = green
            end
        end
    end
end

Rmom = zeros(Float64, nf)
Lmom = similar(Rmom)
Ys = zeros(ComplexF64, N, N)  # surface admittance

#= If conductor p is solid, then “1” is in the pth column, and the same row as
the coefficient J0(p) in J. Otherwise, if conductor p is hollow, then “1” is in the
pth column, and the same rows as the coefficients J0(p) and ~J0(p) in J. =#
U = zeros(Int, N, P)  # incidence matrix
for p = 1:Ps
    i = (-Np:Np) .+ (p - 1) * (2Np + 1) .+ (Np + 1)
    U[i,p] .= 1
end

#= Q is made up of ‘1’s and ‘0’s. On the i-th row of Q, ‘1’s are present in the
columns corresponding to the conductors which are part of the i-th line, and all
other columns are zero. =#
Q = [1 0; 0 1]  # incidence line-conductor

#= S is made up of ‘1’s, ‘0’s, and ‘-1’s. In the i-th row, we have a “1” in the
column corresponding to the active line’s line number, and “-1” in the column
corresponding to the line number of its return line.
-- it's wrong in the thesis; that results in S^T =#
S = [1; -1]  # incidence line-conductor-return
for i in eachindex(freq)
    w = 2π * freq[i]
    # (2.20) for solid conductor
    k_0 = w * sqrt(μ_0 * ϵ_0)  # wave number
    k = sqrt(w * μ * (w * ϵ - 1im * σ))
    for p = 1:P
        for n = -Np:Np
            n1 = n + Np + 1 + (2Np + 1) * (p - 1)
            #v = abs(n)
            v = n
            c1 = (k   * ar[p] * dJ_dz(v, k   * ar[p])) / (μ   * besselj(v, k   * ar[p]))
            c0 = (k_0 * ar[p] * dJ_dz(v, k_0 * ar[p])) / (μ_0 * besselj(v, k_0 * ar[p]))
            Ys[n1, n1] = 2π / (1im * w) * (c1 - c0)
        end
    end
    Zpart = inv(transpose(U) * inv(I - 1im * w * μ_0 * Ys * G) * Ys * U)
    Z = (transpose(S) * (Q * (Zpart) * transpose(Q)) * S)
    Rmom[i] = real(Z)
    Lmom[i] = imag(Z) / w
end

# %% Plots
begin
    p1 = plot(xlabel="Frequency [Hz]", ylabel="Resistance p.u.l. [Ω/m]", title="D = $(D*1e3) mm", legend=:topleft)
    plot!(p1, freq, Rf1, label="Analytical (high-freq)", color=:red, linestyle=:solid, xaxis=:log, yaxis=:log)
    plot!(p1, freq, Rf2, label="Analytical (no proximity)", color=:black, linestyle=:solid, markershape=:diamond, markersize=2)
    plot!(p1, freq, Rmom, label="MoM-SO", color=:blue, linestyle=:solid)
    display(p1)
end

ref = reshape([
    1.00; 10.21
    1.47; 10.21
    2.12; 10.21
    3.16; 10.21
    4.72; 10.21
    6.92; 10.21
    10.17; 10.21
    14.92; 10.21
    22.65; 10.21
    32.70; 10.20
    48.01; 10.20
    70.48; 10.18
    106.99; 10.14
    157.07; 10.08
    230.58; 9.99
    338.51; 9.87
    505.33; 9.75
    754.34; 9.65
    1089.09; 9.57
    1572.39; 9.50
    2426.92; 9.44
    3562.88; 9.39
    5230.55; 9.35
    7678.80; 9.32
    11655.72; 9.29
    16828.11; 9.27
    25120.66; 9.25
    37499.61; 9.23
    54140.58; 9.22
    80819.97; 9.21
    116684.93; 9.21
    177117.04; 9.20
    260019.70; 9.19
    381726.39; 9.19
    560399.98; 9.19
    836553.77; 9.18
], 2, 36)
ref[2,:] ./= 10

begin
    sc = 1e6
    p2 = plot(xlabel="Frequency [Hz]", ylabel="Inductance p.u.l. [μH/m]", title="D = $(D*1e3) mm", legend=:topright)
    plot!(p2, freq, Lf1*sc, label="Analytical (high-freq)", color=:red, linestyle=:solid, xaxis=:log)
    plot!(p2, freq, Lf2*sc, label="Analytical (no proximity)", color=:black, linestyle=:solid)
    plot!(p2, freq, Lmom*sc, label="MoM-SO", color=:blue, linestyle=:solid)
    scatter!(p2, ref[1,:], ref[2,:], label="MoM-SO - REF", color=:black, markershape=:x)
    display(p2)
end

