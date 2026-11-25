# Theoretical moments at order 3 with the pruned state space (Dynare's pruned_state_space_system.m, order 3, no derivatives);
# port of pydyn's `_pruned_order3`. Literal: vec and reshape are column-major, as in MATLAB.

"`pruned_SS.commutation(n, m)`: reshape(kron(vec(eye(n)), eye(m)), n*m, n*m)."
_commutation_nm(n, m) = reshape(kron(vec(Matrix{Float64}(I, n, n)), Matrix{Float64}(I, m, m)), n * m, n * m)

"All perfect matchings of the positions `idx` (Isserlis' theorem)."
function _matchings(idx::Vector{Int})
    isempty(idx) && return [Tuple{Int,Int}[]]
    a = idx[1]
    out = Vector{Vector{Tuple{Int,Int}}}()
    for j in 2:length(idx)
        rest = vcat(idx[2:j-1], idx[j+1:end])
        for mt in _matchings(rest)
            push!(out, vcat([(a, idx[j])], mt))
        end
    end
    out
end

"E[u_i1 ... u_ik] for u ~ N(0, S) (k even), as a k-dimensional array."
function _gaussian_moment_tensor(S::AbstractMatrix, k::Int)
    n = size(S, 1)
    mts = _matchings(collect(1:k))
    out = zeros(ntuple(_ -> n, k))
    for I in CartesianIndices(out)
        t = Tuple(I)
        s = 0.0
        for mt in mts
            p = 1.0
            for (a, b) in mt
                p *= S[t[a], t[b]]
            end
            s += p
        end
        out[I] = s
    end
    out
end

const _THIRD_ORDER_KEYS = ("ghxx", "ghxu", "ghuu", "ghs2", "ghxxx", "ghxxu", "ghxuu", "ghuuu", "ghxss", "ghuss")

"""
Order-3 pruned-state-space moments for the variables `ivar`: `(E_y, Var_y, Corr_y, Corr_yi)`. `dr` holds the third-order
decision rules (`ghxxx`, `ghxxu`, `ghxuu`, `ghuuu`, `ghxss`, `ghuss` besides the second-order ones).
"""
function _pruned_order3(R::Rules, dr, Sigma, ivar, o)
    nar = o["ar"]
    dr = _as_dr(dr)
    all(k -> haskey(dr, k), _THIRD_ORDER_KEYS) || throw(ArgumentError("third-order decision rules (ghxxx, ghxxu, ghxuu, ghuuu, ghxss, ghuss) are needed"))
    x, u = R.nspred, R.nexo
    indx = R.index_states
    indy = R.inv_order_var[ivar]
    ny = length(indy)
    Yss = R.ys[R.order_var]
    g = Dict(k => Matrix{Float64}(k == "ghs2" ? reshape(Float64.(dr[k]), :, 1) : dr[k]) for k in _THIRD_ORDER_KEYS)
    hx, gx = R.ghx[indx, :], R.ghx[indy, :]
    hu, gu = R.ghu[indx, :], R.ghu[indy, :]
    hxx, gxx = g["ghxx"][indx, :], g["ghxx"][indy, :]
    hxu, gxu = g["ghxu"][indx, :], g["ghxu"][indy, :]
    huu, guu = g["ghuu"][indx, :], g["ghuu"][indy, :]
    hss, gss = g["ghs2"][indx, 1], g["ghs2"][indy, 1]
    hxxx, gxxx = g["ghxxx"][indx, :], g["ghxxx"][indy, :]
    hxxu, gxxu = g["ghxxu"][indx, :], g["ghxxu"][indy, :]
    hxuu, gxuu = g["ghxuu"][indx, :], g["ghxuu"][indy, :]
    huuu, guuu = g["ghuuu"][indx, :], g["ghuuu"][indy, :]
    hxss, gxss = g["ghxss"][indx, :], g["ghxss"][indy, :]
    huss, guss = g["ghuss"][indx, :], g["ghuss"][indy, :]
    E_uu = Matrix{Float64}(Sigma)
    vE_uu = vec(E_uu)
    col(v) = reshape(v, :, 1)
    row(v) = reshape(v, 1, :)

    # first order
    Var_z1, Schur_u = lyapunov_symm(hx, hu * E_uu * hu'; qz_criterium=o["qz_criterium"], complex_threshold=o["lyapunov_complex_threshold"])
    size(Schur_u, 2) > 0 && error("pruned_state_space_system: the pruned state space currently does not support unit roots at higher order.")
    E_xfxf = Var_z1
    vE_xfxf = vec(E_xfxf)
    hx_hx, hx_hu, hu_hu = kron(hx, hx), kron(hx, hu), kron(hu, hu)
    I_xx = Matrix{Float64}(I, x * x, x * x)
    K_x_x = _commutation_nm(x, x)
    invIx_hx = inv(Matrix{Float64}(I, x, x) - hx)
    QPuE4 = vec(_gaussian_moment_tensor(E_uu, 4))
    E_xfxf_uu = kron(E_xfxf, E_uu')

    # second-order system: z = [xf; xs; kron(xf,xf)], inov = [u; kron(u,u)-E_uu(:); kron(xf,u)]
    i1 = 1:x; i2 = x+1:2x; i3 = 2x+1:2x+x*x
    j1 = 1:u; j2 = u+1:u+u*u; j3 = u+u*u+1:u+u*u+x*u
    nz2, ni2 = 2x + x * x, u + u * u + x * u
    A = zeros(nz2, nz2)
    A[i1, i1] = hx
    A[i2, i2] = hx
    A[i2, i3] = 0.5 .* hxx
    A[i3, i3] = hx_hx
    B = zeros(nz2, ni2)
    B[i1, j1] = hu
    B[i2, j2] = 0.5 .* huu
    B[i2, j3] = hxu
    B[i3, j2] = hu_hu
    B[i3, j3] = (I_xx + K_x_x) * hx_hu
    c_xs = 0.5 .* hss .+ 0.5 .* (huu * vE_uu)
    Varinov = zeros(ni2, ni2)
    Varinov[j1, j1] = E_uu
    Varinov[j2, j2] = reshape(QPuE4, u * u, u * u) - vE_uu * vE_uu'
    Varinov[j3, j3] = E_xfxf_uu
    E_xs = invIx_hx * (0.5 .* (hxx * vE_xfxf) .+ c_xs)
    Om_z = B * Varinov * B'
    Var_z, err = disclyap_fast(A, Om_z; tol=o["lyapunov_doubling_tol"])
    if err
        println("PRUNED_STATE_SPACE_SYSTEM: error flag in disclyap_fast at order=2, use lyapunov_symm")
        Var_z, _ = lyapunov_symm(A, Om_z; qz_criterium=1.0, complex_threshold=o["lyapunov_complex_threshold"])
    end
    Var_z[i1, i2] .= 0; Var_z[i1, i3] .= 0; Var_z[i2, i1] .= 0; Var_z[i3, i1] .= 0

    # objects for order 3
    K_u_xx = _commutation_nm(u, x * x)
    K_u_ux = _commutation_nm(u, u * x)
    K_xx_x = _commutation_nm(x * x, x)
    hx_hss2 = kron(hx, col(0.5 .* hss))
    hu_hss2 = kron(hu, col(0.5 .* hss))
    hx_hxx2 = kron(hx, 0.5 .* hxx)
    hxx2_hu = kron(0.5 .* hxx, hu)
    hx_hxu = kron(hx, hxu)
    hxu_hu = kron(hxu, hu)
    hx_huu2 = kron(hx, 0.5 .* huu)
    hu_huu2 = kron(hu, 0.5 .* huu)
    hx_hx_hx = kron(hx, hx_hx)
    hx_hx_hu = kron(hx_hx, hu)
    hu_hx_hx = kron(hu, hx_hx)
    hu_hu_hu = kron(hu_hu, hu)
    hx_hu_hu = kron(hx, hu_hu)
    hu_hx_hu = kron(hu, hx_hu)
    invIxx_hx_hx = inv(Matrix{Float64}(I, x * x, x * x) - hx_hx)

    E_xsxs = Var_z[i2, i2] + E_xs * E_xs'
    E_xsxf_xf = Var_z[i2, i3] + E_xs * vE_xfxf'
    E_xf_xfxs = Var_z[i3, i2] + vE_xfxf * E_xs'
    E_xf_xfxf_xf = Var_z[i3, i3] + vE_xfxf * vE_xfxf'
    M = (hxx * reshape(K_xx_x * vec(E_xf_xfxs), x * x, x) * hx'
         + hxu * kron(col(E_xs), E_uu) * hu'
         + 1 / 6 * hxxx * reshape(E_xf_xfxf_xf, x^3, x) * hx'
         + 1 / 6 * huuu * reshape(QPuE4, u^3, u) * hu'
         + 3 / 6 * hxxu * kron(col(vE_xfxf), E_uu) * hu'
         + 3 / 6 * hxuu * kron(E_xfxf, col(vE_uu)) * hx'
         + 3 / 6 * hxss * E_xfxf * hx'
         + 3 / 6 * huss * E_uu * hu')
    E_xrdxf = reshape(invIxx_hx_hx * vec(M), x, x)
    QPuE6 = vec(_gaussian_moment_tensor(E_uu, 6))

    # third-order system
    # z = [xf; xs; kron(xf,xf); xrd; kron(xf,xs); kron(kron(xf,xf),xf)]
    # inov = [u; kron(u,u)-E_uu(:); kron(xf,u); kron(xs,u); kron(kron(xf,xf),u); kron(kron(xf,u),u); kron(kron(u,u),u)]
    i4 = nz2+1:nz2+x
    i5 = nz2+x+1:nz2+x+x*x
    i6 = nz2+x+x*x+1:nz2+x+x*x+x^3
    j4 = ni2+1:ni2+x*u
    j5 = ni2+x*u+1:ni2+x*u+x*x*u
    j6 = ni2+x*u+x*x*u+1:ni2+x*u+x*x*u+x*u*u
    j7 = ni2+x*u+x*x*u+x*u*u+1:ni2+x*u+x*x*u+x*u*u+u^3
    nz = last(i6)
    ni = last(j7)
    A = zeros(nz, nz)
    A[i1, i1] = hx
    A[i2, i2] = hx
    A[i2, i3] = 0.5 .* hxx
    A[i3, i3] = hx_hx
    A[i4, i1] = 3 / 6 .* hxss
    A[i4, i4] = hx
    A[i4, i5] = hxx
    A[i4, i6] = 1 / 6 .* hxxx
    A[i5, i1] = hx_hss2
    A[i5, i5] = hx_hx
    A[i5, i6] = hx_hxx2
    A[i6, i6] = hx_hx_hx
    B = zeros(nz, ni)
    B[i1, j1] = hu
    B[i2, j2] = 0.5 .* huu
    B[i2, j3] = hxu
    B[i3, j2] = hu_hu
    B[i3, j3] = (I_xx + K_x_x) * hx_hu
    B[i4, j1] = 3 / 6 .* huss
    B[i4, j4] = hxu
    B[i4, j5] = 3 / 6 .* hxxu
    B[i4, j6] = 3 / 6 .* hxuu
    B[i4, j7] = 1 / 6 .* huuu
    B[i5, j1] = hu_hss2
    B[i5, j4] = K_x_x * hx_hu
    B[i5, j5] = hx_hxu + K_x_x * hxx2_hu
    B[i5, j6] = hx_huu2 + K_x_x * hxu_hu
    B[i5, j7] = hu_huu2
    B[i6, j5] = hx_hx_hu + kron(hx, K_x_x * hx_hu) + hu_hx_hx * K_u_xx
    B[i6, j6] = hx_hu_hu + hu_hx_hu * K_u_ux + kron(hu, K_x_x * hx_hu) * K_u_ux
    B[i6, j7] = hu_hu_hu
    C = zeros(ny, nz)
    C[:, i1] = gx + 3 / 6 .* gxss
    C[:, i2] = gx
    C[:, i3] = 0.5 .* gxx
    C[:, i4] = gx
    C[:, i5] = gxx
    C[:, i6] = 1 / 6 .* gxxx
    D = zeros(ny, ni)
    D[:, j1] = gu + 3 / 6 .* guss
    D[:, j2] = 0.5 .* guu
    D[:, j3] = gxu
    D[:, j4] = gxu
    D[:, j5] = 3 / 6 .* gxxu
    D[:, j6] = 3 / 6 .* gxuu
    D[:, j7] = 1 / 6 .* guuu
    d = 0.5 .* gss .+ 0.5 .* (guu * vE_uu)

    E4_u_uuu = reshape(QPuE4, u, u^3)
    E4_uuu_u = reshape(QPuE4, u^3, u)
    Vi = zeros(ni, ni)
    Vi[j1, j1] = E_uu
    Vi[j1, j4] = kron(row(E_xs), E_uu)
    Vi[j1, j5] = kron(row(vE_xfxf), E_uu)
    Vi[j1, j7] = E4_u_uuu
    Vi[j2, j2] = reshape(QPuE4, u * u, u * u) - vE_uu * vE_uu'
    Vi[j3, j3] = E_xfxf_uu
    Vi[j4, j1] = kron(col(E_xs), E_uu)
    Vi[j4, j4] = kron(E_xsxs, E_uu)
    Vi[j4, j5] = kron(E_xsxf_xf, E_uu)
    Vi[j4, j7] = kron(col(E_xs), E4_u_uuu)
    Vi[j5, j1] = kron(col(vE_xfxf), E_uu)
    Vi[j5, j4] = kron(E_xf_xfxs, E_uu)
    Vi[j5, j5] = kron(E_xf_xfxf_xf, E_uu)
    Vi[j5, j7] = kron(col(vE_xfxf), E4_u_uuu)
    Vi[j6, j6] = kron(E_xfxf, reshape(QPuE4, u * u, u * u))
    Vi[j7, j1] = E4_uuu_u
    Vi[j7, j4] = kron(row(E_xs), E4_uuu_u)
    Vi[j7, j5] = kron(row(vE_xfxf), E4_uuu_u)
    Vi[j7, j7] = reshape(QPuE6, u^3, u^3)
    Varinov = Vi

    function inovzlag(hxi)
        E = zeros(ni, nz)
        E[j6, i1] = kron(hxi * E_xfxf, col(vE_uu))
        E[j6, i4] = kron(hxi * E_xrdxf', col(vE_uu))
        E[j6, i5] = kron(hxi * reshape(K_xx_x * vec(E_xsxf_xf), x, x * x), col(vE_uu))
        E[j6, i6] = kron(hxi * reshape(E_xf_xfxf_xf, x, x^3), col(vE_uu))
        E
    end

    E_inovzlag1 = inovzlag(Matrix{Float64}(I, x, x))
    BinA = B * E_inovzlag1 * A'
    Om_z = B * Varinov * B' + BinA + BinA'
    Var_z, err = disclyap_fast(A, Om_z; tol=o["lyapunov_doubling_tol"])
    if err
        println("PRUNED_STATE_SPACE_SYSTEM: error flag in disclyap_fast at order=3, use lyapunov_symm")
        Var_z, _ = lyapunov_symm(A, Om_z; qz_criterium=1.0, complex_threshold=o["lyapunov_complex_threshold"])
    end
    for (a_, b_) in ((i1, i2), (i1, i3), (i2, i1), (i2, i4), (i2, i5), (i2, i6), (i3, i1), (i3, i4), (i3, i5),
                     (i3, i6), (i4, i2), (i4, i3), (i5, i2), (i5, i3), (i6, i2), (i6, i3))
        Var_z[a_, b_] .= 0
    end

    Var_y = C * Var_z * C' + D * E_inovzlag1 * C' + C * E_inovzlag1' * D' + D * Varinov * D'
    Var_y[abs.(Var_y) .< 1e-12] .= 0.0
    sd = _sqrt_nan(diag(Var_y))
    sy = sd * sd'
    Corr_y = Var_y ./ sy
    Corr_yi = Matrix{Float64}[]
    tmp = A * E_inovzlag1' * D' + B * Varinov * D'
    Ai = Matrix{Float64}(I, nz, nz)
    hxi = Matrix{Float64}(I, x, x)
    E_inovzlagi = E_inovzlag1
    Var_zi = Var_z
    for _ in 1:nar
        Var_zi = A * Var_zi + B * E_inovzlagi
        hxi = hx * hxi
        E_inovzlagi = inovzlag(hxi)
        Var_yi = C * Var_zi * C' + C * Ai * tmp + D * E_inovzlagi * C'
        push!(Corr_yi, Var_yi ./ sy)
        Ai = Ai * A
    end
    E_z = vcat(zeros(x), E_xs, vE_xfxf, zeros(x), zeros(x * x), zeros(x^3))
    E_y = Yss[indy] .+ C * E_z .+ d
    E_y, Var_y, Corr_y, Corr_yi
end
