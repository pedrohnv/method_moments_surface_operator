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

# Example 2, p. 42, coaxial cable
a = 22e-3  # radius core, m
b = 39.5e-3  # radius inner sheath, m
c = 44e-3  # radius outer sheath, m
σ = 58e6  # conductivity, S/m
μ = μ_0
ϵ = ϵ_0

nf = 100  # number of frequency points
freq = exp10.(range(0, 6, length=nf))  # frequency range, Hz

# %% Approximation formula 1
Rf1 = zeros(Float64, nf)  # Analytical (Schelkunoff Model)
Lf1 = similar(Rf1)

for (i, f) in enumerate(freq)
    w = 2π * f
    jw = 1im * w
    L1 = μ / (2pi) * log(b/a)
    eta = sqrt(jw * μ / σ)
    gamma = sqrt(jw * μ * σ)
    za = eta / (2pi * a) * besseli(0, gamma * a) / besseli(1, gamma * a)
    num1 = besseli(0, gamma * b) * besselk(1, gamma * c)
    num2 = besselk(0, gamma * b) * besseli(1, gamma * c)
    den1 = besseli(1, gamma * c) * besselk(1, gamma * b)
    den2 = besselk(1, gamma * c) * besseli(1, gamma * b)
    zb = eta / (2pi * b) * (num1 + num2) / (den1 - den2)
    z = jw * L1 + za + zb
    Rf1[i] = real(z)
    Lf1[i] = imag(z) / w
end

# %% MoM-SO
Np = 1  # number - 1 of coefficients of the Fourier expansion in space-domain (basis function)
Ps = 1  # number of solid conductors
Ph = 2  # number of hollow conductors
P = Ps + Ph  # total number of conductors
N = Ps * (2Np + 1) + Ph * 2*(2Np + 1)  # total number of equivalent current coefficients (2.36)

# generate all the RHS matrices of (2.61): Ys, G, U
G = zeros(ComplexF64, N, N)  # Green's matrix
ar = [a, b, c]
xp = [0.0, 0.0, 0.0]
yp = [0.0, 0.0, 0.0]
closed_form = true
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
                    function green_integrand(θ)
                        r = zeros(2,2)
                        for (i, k) in enumerate([p, q])
                            r[1,i] = xp[k] + ar[k] * cos(θ[i])
                            r[2,i] = yp[k] + ar[k] * sin(θ[i])
                        end
                        return log(norm(r[:,1] .- r[:,2])) * exp(1im * (n * θ[2] - m * θ[1]))
                    end
                    c1 = (p != q && n > 0 && m >= 1)
                    c2 = (p == q && m != 0)
                    c3 = (p == q && n != 0 && m != n)
                    if (c1 || c2 || c3)
                        green = 0
                    elseif (p != q && n < 0)
                        m2 = -m + Np + 1 + (2Np + 1) * (q - 1)
                        n2 = -n + Np + 1 + (2Np + 1) * (p - 1)
                        green = conj(G[m2, n2])
                    else
                        green, err = hcubature(green_integrand, [0.0, 0.0], [2π, 2π])
                        green /= 248.05021344239853  # 8π^3
                    end
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
Q = [1 0; 0 1; 0 1]  # incidence line-conductor

#= S is made up of ‘1’s, ‘0’s, and ‘-1’s. In the i-th row, we have a “1” in the
column corresponding to the active line’s line number, and “-1” in the column
corresponding to the line number of its return line.
-- it's wrong in the thesis; that results in S^T =#
S = [1; -1; -1]  # incidence line-conductor-return
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
    plot!(p1, freq, Rf1, label="Analytical", color=:red, linestyle=:solid, xaxis=:log, yaxis=:log)
    plot!(p1, freq, Rmom, label="MoM-SO", color=:blue, linestyle=:solid)
    display(p1)
end


begin
    sc = 1e6
    p2 = plot(xlabel="Frequency [Hz]", ylabel="Inductance p.u.l. [μH/m]", title="D = $(D*1e3) mm", legend=:topright)
    plot!(p2, freq, Lf1*sc, label="Analytical", color=:red, linestyle=:solid, xaxis=:log)
    plot!(p2, freq, Lmom*sc, label="MoM-SO", color=:blue, linestyle=:solid)
    display(p2)
end
#= nem a fórmula analítica bate com a figura da dissertação. Tem algum valor de parâmetro errado =#
