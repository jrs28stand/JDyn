# Minimal FFT (no FFTW dependency): iterative radix-2 for powers of two, Bluestein's chirp-z for other lengths.
# Used by the filtered theoretical moments (th_autocovariances.m's real(ifft(...)) over the frequency grid).

function _fft_pow2!(x::Vector{ComplexF64}, sign::Int)
    n = length(x)
    n <= 1 && return x
    j = 0                                   # bit-reversal permutation
    for i in 1:n-1
        bit = n >> 1
        while j & bit != 0
            j ⊻= bit; bit >>= 1
        end
        j ⊻= bit
        if i < j; x[i+1], x[j+1] = x[j+1], x[i+1]; end
    end
    tw = [cis(sign * 2pi * k / n) for k in 0:n÷2-1]       # exact twiddles, no recurrence error growth
    len = 2
    while len <= n
        half, step = len >> 1, n ÷ len
        for s in 1:len:n
            @inbounds for k in 0:half-1
                u = x[s+k]; v = x[s+k+half] * tw[k*step+1]
                x[s+k] = u + v; x[s+k+half] = u - v
            end
        end
        len <<= 1
    end
    x
end

"Unnormalised DFT, sum_k x[k] exp(sign 2 pi i k t / n) for t = 0..n-1 (sign = +1: inverse transform times n)."
function _dft(x::AbstractVector{<:Number}, sign::Int)
    n = length(x)
    xc = Vector{ComplexF64}(x)
    ispow2(n) && return _fft_pow2!(xc, sign)
    n <= 1 && return xc
    m = nextpow(2, 2n - 1)
    # chirp w[k] = exp(sign i pi k^2 / n), k^2 reduced mod 2n for accuracy
    w = [cis(sign * pi * (mod(k * k, 2n)) / n) for k in 0:n-1]
    a = zeros(ComplexF64, m); b = zeros(ComplexF64, m)
    for k in 0:n-1
        a[k+1] = xc[k+1] * w[k+1]
    end
    b[1] = conj(w[1])
    for k in 1:n-1
        b[k+1] = conj(w[k+1]); b[m-k+1] = conj(w[k+1])
    end
    _fft_pow2!(a, -1); _fft_pow2!(b, -1)
    a .*= b
    _fft_pow2!(a, +1)
    [a[k+1] / m * w[k+1] for k in 0:n-1]
end
