# Dulmage-Mendelsohn permutation, a port of CSparse's cs_dmperm (Tim Davis), which is what MATLAB's and Octave's
# `[p,q,r,s] = dmperm(A)` return. Dynare's `dynare_solve` (solve_algo = 2 and 4) solves the static model block by block
# in the order this decomposition gives, so the same matching and the same blocks are needed to reproduce which
# variables of an under-determined steady state it moves (docs/REVIEW2.md, R01).

"CSC pattern of a matrix (0-based row indices sorted within each column), as Octave builds it from a full matrix."
function _csc_pattern(A::AbstractMatrix)
    m, n = size(A)
    Ap = zeros(Int, n + 1)
    Ai = Int[]
    for j in 1:n
        for i in 1:m
            A[i, j] != 0 && push!(Ai, i - 1)
        end
        Ap[j+1] = length(Ai)
    end
    m, n, Ap, Ai
end

function _transpose_pattern(m, n, Ap, Ai)
    Cp = zeros(Int, m + 1)
    for p in eachindex(Ai)
        Cp[Ai[p]+2] += 1
    end
    for i in 1:m
        Cp[i+1] += Cp[i]
    end
    w = copy(Cp[1:m])
    Ci = zeros(Int, length(Ai))
    for j in 0:n-1
        for p in Ap[j+1]+1:Ap[j+2]
            q = w[Ai[p]+1]; w[Ai[p]+1] += 1
            Ci[q+1] = j
        end
    end
    Cp, Ci
end

# cs_augment: find an augmenting path starting at column k
function _cs_augment!(k, Ap, Ai, jmatch, cheap, w, js, is, ps)
    found = false
    p = 0
    i = -1
    head = 0
    js[1] = k
    while head >= 0
        j = js[head+1]
        if w[j+1] != k
            w[j+1] = k
            p = cheap[j+1]
            while p < Ap[j+2] && !found
                i = Ai[p+1]
                found = jmatch[i+1] == -1
                p += 1
            end
            cheap[j+1] = p
            if found
                is[head+1] = i
                break
            end
            ps[head+1] = Ap[j+1]
        end
        p = ps[head+1]
        while p < Ap[j+2]
            i = Ai[p+1]
            if w[jmatch[i+1]+1] == k
                p += 1
                continue
            end
            ps[head+1] = p + 1
            is[head+1] = i
            head += 1
            js[head+1] = jmatch[i+1]
            break
        end
        p == Ap[j+2] && (head -= 1)
    end
    if found
        for q in head:-1:0
            jmatch[is[q+1]+1] = js[q+1]
        end
    end
    nothing
end

"cs_maxtrans: `(jmatch, imatch)`, jmatch[i] = column matched to row i (or -1), imatch[j] = row matched to column j; 0-based values."
function _cs_maxtrans(m, n, Ap, Ai)
    w = zeros(Int, m)
    k = 0
    n2 = 0
    for j in 0:n-1
        n2 += Ap[j+1] < Ap[j+2]
        for p in Ap[j+1]+1:Ap[j+2]
            w[Ai[p]+1] = 1
            k += (j == Ai[p])
        end
    end
    if k == min(m, n)                      # diagonal zero-free
        jmatch = fill(-1, m); imatch = fill(-1, n)
        for i in 0:k-1
            jmatch[i+1] = i; imatch[i+1] = i
        end
        return jmatch, imatch
    end
    m2 = sum(w)
    transposed = m2 < n2
    if transposed
        Cp, Ci = _transpose_pattern(m, n, Ap, Ai)
        mm, nn = n, m                      # C = A'
    else
        Cp, Ci = Ap, Ai
        mm, nn = m, n
    end
    jm = fill(-1, mm)                      # jmatch of C: column matched to each row of C
    cheap = copy(Cp[1:nn])
    wk = fill(-1, nn)
    js = zeros(Int, nn); is = zeros(Int, nn); ps = zeros(Int, nn)
    for kk in 0:nn-1
        _cs_augment!(kk, Cp, Ci, jm, cheap, wk, js, is, ps)
    end
    im = fill(-1, nn)
    for i in 0:mm-1
        jm[i+1] >= 0 && (im[jm[i+1]+1] = i)
    end
    transposed ? (im, jm) : (jm, im)
end

# cs_bfs
function _cs_bfs!(Ap, Ai, n, wi, wj, queue, imatch, jmatch, mark)
    head = 0; tail = 0
    for j in 0:n-1
        imatch[j+1] >= 0 && continue
        wj[j+1] = 0
        tail += 1; queue[tail] = j
    end
    tail == 0 && return
    while head < tail
        head += 1
        j = queue[head]
        for p in Ap[j+1]+1:Ap[j+2]
            i = Ai[p]
            wi[i+1] >= 0 && continue
            wi[i+1] = mark
            j2 = jmatch[i+1]
            wj[j2+1] >= 0 && continue
            wj[j2+1] = mark
            tail += 1; queue[tail] = j2
        end
    end
    nothing
end

# cs_dfs with separate recursion and output stacks; `marked` plays the role of CS_MARK
function _cs_dfs!(j0, Gp, Gi, top, out, marked, stack, pstack)
    head = 0
    stack[1] = j0
    while head >= 0
        j = stack[head+1]
        if !marked[j+1]
            marked[j+1] = true
            pstack[head+1] = Gp[j+1]
        end
        done = true
        p2 = Gp[j+2]
        p = pstack[head+1]
        while p < p2
            i = Gi[p+1]
            if marked[i+1]
                p += 1
                continue
            end
            pstack[head+1] = p
            head += 1
            stack[head+1] = i
            done = false
            break
        end
        if done
            head -= 1
            top -= 1
            out[top+1] = j
        end
    end
    top
end

# cs_scc on a square pattern: `(p, r, nb)`
function _cs_scc(n, Ap, Ai)
    ATp, ATi = _transpose_pattern(n, n, Ap, Ai)
    xi = zeros(Int, n)
    p = zeros(Int, n)
    r = zeros(Int, n + 1)
    stack = zeros(Int, n + 1); pstack = zeros(Int, n + 1)
    marked = falses(n)
    top = n
    for i in 0:n-1
        marked[i+1] || (top = _cs_dfs!(i, Ap, Ai, top, xi, marked, stack, pstack))
    end
    fill!(marked, false)
    top = n
    nb = n
    for k in 0:n-1
        i = xi[k+1]
        marked[i+1] && continue
        r[nb+1] = top
        nb -= 1
        top = _cs_dfs!(i, ATp, ATi, top, p, marked, stack, pstack)
    end
    r[nb+1] = 0
    for k in nb:n
        r[k-nb+1] = r[k+1]
    end
    nb = n - nb
    Blk = zeros(Int, n)
    for b in 0:nb-1
        for k in r[b+1]:r[b+2]-1
            Blk[p[k+1]+1] = b
        end
    end
    rcopy = copy(r[1:nb+1])
    for i in 0:n-1
        b = Blk[i+1]
        p[rcopy[b+1]+1] = i
        rcopy[b+1] += 1
    end
    p, r, nb
end

"""
    dmperm(A) -> (p, q, r, s)

Dulmage-Mendelsohn decomposition of the non-zero pattern of `A`, as MATLAB's `[p,q,r,s] = dmperm(A)` (1-based): `A[p,q]`
is block upper triangular, block k being rows `p[r[k]:r[k+1]-1]` and columns `q[s[k]:s[k+1]-1]`. Port of CSparse's
`cs_dmperm` (seed 0).
"""
function dmperm(A::AbstractMatrix)
    m, n, Ap, Ai = _csc_pattern(A)
    jmatch, imatch = _cs_maxtrans(m, n, Ap, Ai)
    p = zeros(Int, m); q = zeros(Int, n)
    r = zeros(Int, m + 6); s = zeros(Int, n + 6)
    cc = zeros(Int, 5); rr = zeros(Int, 5)
    wi = fill(-1, m); wj = fill(-1, n)
    # coarse decomposition
    _cs_bfs!(Ap, Ai, n, wi, wj, q, imatch, jmatch, 1)
    Tp, Ti = _transpose_pattern(m, n, Ap, Ai)
    _cs_bfs!(Tp, Ti, m, wj, wi, p, jmatch, imatch, 3)
    # cs_unmatched(n, wj, q, cc, 0)
    kr = cc[1]
    for j in 0:n-1
        wj[j+1] == 0 && (kr += 1; q[kr] = j)
    end
    cc[2] = kr
    function matched(set, mark)
        kc = cc[set+1]; krr = rr[set]
        for j in 0:n-1
            wj[j+1] == mark || continue
            krr += 1; p[krr] = imatch[j+1]
            kc += 1; q[kc] = j
        end
        cc[set+2] = kc
        rr[set+1] = krr
    end
    matched(1, 1); matched(2, -1); matched(3, 3)
    kr = rr[4]
    for i in 0:m-1
        wi[i+1] == 0 && (kr += 1; p[kr] = i)
    end
    rr[5] = kr
    # fine decomposition: C = A(R2, C2)
    pinv = zeros(Int, m)
    for k in 1:m
        pinv[p[k]+1] = k - 1
    end
    nc = cc[4] - cc[3]
    Cp = zeros(Int, nc + 1); Ci = Int[]
    for jj in 0:nc-1
        j = q[cc[3]+jj+1]
        for pp in Ap[j+1]+1:Ap[j+2]
            i = pinv[Ai[pp]+1]
            (i >= rr[2] && i < rr[3]) && push!(Ci, i - rr[2])
        end
        Cp[jj+2] = length(Ci)
    end
    ps, rs, nb1 = _cs_scc(nc, Cp, Ci)
    wj2 = [q[ps[k]+cc[3]+1] for k in 1:nc]
    for k in 1:nc
        q[k+cc[3]] = wj2[k]
    end
    wi2 = [p[ps[k]+rr[2]+1] for k in 1:nc]
    for k in 1:nc
        p[k+rr[2]] = wi2[k]
    end
    nb2 = 0
    r[1] = s[1] = 0
    if cc[3] > 0
        nb2 += 1
    end
    for k in 0:nb1-1
        r[nb2+1] = rs[k+1] + rr[2]
        s[nb2+1] = rs[k+1] + cc[3]
        nb2 += 1
    end
    if rr[3] < m
        r[nb2+1] = rr[3]
        s[nb2+1] = cc[4]
        nb2 += 1
    end
    r[nb2+1] = m
    s[nb2+1] = n
    p .+ 1, q .+ 1, r[1:nb2+1] .+ 1, s[1:nb2+1] .+ 1
end
